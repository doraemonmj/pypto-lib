# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#
# v2: single InCore kernel merging gate/up (phase 1) and down (phase 2), both
# SPMD-split on the N axis. Phase 2's reduction axis is INTER (== phase 1's N),
# so each phase-2 block needs the *full* ffn tensor across all phase-1 blocks.
# That requires a grid-wide synchronization between the two phases, which the
# framework does not yet expose. The sync point is marked by `_grid_sync()`
# below as a PLACEHOLDER (currently an identity no-op) until a real barrier
# primitive is available.

from __future__ import annotations

import pypto.language as pl

BATCH = 16
HIDDEN = 7168
INTERMEDIATE = 3072  # example/test shape; wse-ffn production shape may vary

M_TILE = 16
P1_N_TILE = 128
P1_K_TILE = 128
P2_N_TILE = 128
P2_K_TILE = 128

# Legacy aliases retained for the golden reference.
K_CHUNK = P1_K_TILE
FFN_OUT_CHUNK = P1_N_TILE


def _grid_sync(ffn_scratch):
    """PLACEHOLDER grid-wide barrier.

    Phase 2 needs every phase-1 block's contribution to `ffn_scratch` before
    it can start its INTER-axis reduction. A real implementation must block
    each SPMD block until all blocks have finished writing `ffn_scratch`.
    No such primitive exists yet, so this is a no-op identity pass-through.
    """
    # TODO: replace with a real grid-wide barrier once the framework exposes one.
    return ffn_scratch


def build_wse_ffn_program(
    batch: int = BATCH,
    hidden_size: int = HIDDEN,
    intermediate_size: int = INTERMEDIATE,
):
    BATCH_SIZE = batch
    HIDDEN_SIZE = hidden_size
    INTER_SIZE = intermediate_size

    M_BLOCKS = BATCH_SIZE // M_TILE
    # Phase 1: N along INTER, K along HIDDEN.
    P1_N_BLOCKS = (INTER_SIZE  + P1_N_TILE - 1) // P1_N_TILE
    P1_K_BLOCKS = (HIDDEN_SIZE + P1_K_TILE - 1) // P1_K_TILE
    # Phase 2: N along HIDDEN, K along INTER.
    P2_N_BLOCKS = (HIDDEN_SIZE + P2_N_TILE - 1) // P2_N_TILE
    P2_K_BLOCKS = (INTER_SIZE  + P2_K_TILE - 1) // P2_K_TILE

    # Single merged grid must cover both phases' block counts.
    P1_GRID = M_BLOCKS * P1_N_BLOCKS
    P2_GRID = M_BLOCKS * P2_N_BLOCKS
    GRID = max(P1_GRID, P2_GRID)

    @pl.program
    class WseFFN:
        # Merged InCore kernel: each block runs phase-1 (if in P1 range),
        # then the grid sync, then phase-2 (if in P2 range).
        @pl.function(type=pl.FunctionType.InCore)
        def kernel_ffn(
            self,
            post_norm: pl.Tensor[[BATCH_SIZE, HIDDEN_SIZE], pl.BF16],
            w_gate: pl.Tensor[[HIDDEN_SIZE, INTER_SIZE], pl.BF16],
            w_up: pl.Tensor[[HIDDEN_SIZE, INTER_SIZE], pl.BF16],
            w_down: pl.Tensor[[INTER_SIZE, HIDDEN_SIZE], pl.BF16],
            ffn_scratch: pl.InOut[pl.Tensor[[BATCH_SIZE, INTER_SIZE], pl.BF16]],
            out: pl.Out[pl.Tensor[[BATCH_SIZE, HIDDEN_SIZE], pl.FP32]],
        ) -> pl.Tensor[[BATCH_SIZE, HIDDEN_SIZE], pl.FP32]:
            idx = pl.tile.get_block_idx()

            # ---- Phase 1: gate/up, SPMD-split on N=INTER ----
            p1_m0 = (idx // P1_N_BLOCKS) * M_TILE
            p1_n0 = (idx %  P1_N_BLOCKS) * P1_N_TILE

            x_full = pl.load(post_norm, [p1_m0, 0], [M_TILE, HIDDEN_SIZE], target_memory=pl.MemorySpace.Mat)

            x0 = pl.slice(x_full, [M_TILE, P1_K_TILE], [0, 0])
            wg0 = pl.load(w_gate, [0, p1_n0], [P1_K_TILE, P1_N_TILE], target_memory=pl.MemorySpace.Mat)
            gate_acc = pl.matmul(x0, wg0)
            for kb in pl.range(1, P1_K_BLOCKS):
                k0 = kb * P1_K_TILE
                xi = pl.slice(x_full, [M_TILE, P1_K_TILE], [0, k0])
                wg = pl.load(w_gate, [k0, p1_n0], [P1_K_TILE, P1_N_TILE], target_memory=pl.MemorySpace.Mat)
                gate_acc = pl.matmul_acc(gate_acc, xi, wg)
            gate_vec = pl.move(gate_acc, target_memory=pl.MemorySpace.Vec, blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.none_box)

            x1 = pl.slice(x_full, [M_TILE, P1_K_TILE], [0, 0])
            wu0 = pl.load(w_up, [0, p1_n0], [P1_K_TILE, P1_N_TILE], target_memory=pl.MemorySpace.Mat)
            up_acc = pl.matmul(x1, wu0)
            for kb in pl.range(1, P1_K_BLOCKS):
                k0 = kb * P1_K_TILE
                xj = pl.slice(x_full, [M_TILE, P1_K_TILE], [0, k0])
                wu = pl.load(w_up, [k0, p1_n0], [P1_K_TILE, P1_N_TILE], target_memory=pl.MemorySpace.Mat)
                up_acc = pl.matmul_acc(up_acc, xj, wu)

            sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_vec)), 1.0))
            gate_silu = pl.mul(gate_vec, sigmoid)
            up_vec = pl.move(up_acc, target_memory=pl.MemorySpace.Vec, blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.none_box)
            ffn_chunk = pl.mul(gate_silu, up_vec)
            ffn_bf16 = pl.cast(ffn_chunk, target_type=pl.BF16)
            ffn_scratch = pl.store(ffn_bf16, [p1_m0, p1_n0], ffn_scratch)

            # ---- Grid-wide sync (PLACEHOLDER) ----
            # Every block must have written its INTER column band before any
            # block reads the full ffn for phase 2.
            ffn_scratch = _grid_sync(ffn_scratch)

            # ---- Phase 2: down, SPMD-split on N=HIDDEN, K=INTER ----
            p2_m0 = (idx // P2_N_BLOCKS) * M_TILE
            p2_n0 = (idx %  P2_N_BLOCKS) * P2_N_TILE

            ffn_full = pl.load(ffn_scratch, [p2_m0, 0], [M_TILE, INTER_SIZE], target_memory=pl.MemorySpace.Mat)

            ffn0 = pl.slice(ffn_full, [M_TILE, P2_K_TILE], [0, 0])
            wd0 = pl.load(w_down, [0, p2_n0], [P2_K_TILE, P2_N_TILE], target_memory=pl.MemorySpace.Mat)
            down_acc = pl.matmul(ffn0, wd0)
            for ob in pl.range(1, P2_K_BLOCKS):
                k0 = ob * P2_K_TILE
                ffn_i = pl.slice(ffn_full, [M_TILE, P2_K_TILE], [0, k0])
                wd_i = pl.load(w_down, [k0, p2_n0], [P2_K_TILE, P2_N_TILE], target_memory=pl.MemorySpace.Mat)
                down_acc = pl.matmul_acc(down_acc, ffn_i, wd_i)
            out = pl.store(down_acc, [p2_m0, p2_n0], out)
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def wse_ffn(
            self,
            post_norm: pl.Tensor[[BATCH_SIZE, HIDDEN_SIZE], pl.BF16],
            w_gate: pl.Tensor[[HIDDEN_SIZE, INTER_SIZE], pl.BF16],
            w_up: pl.Tensor[[HIDDEN_SIZE, INTER_SIZE], pl.BF16],
            w_down: pl.Tensor[[INTER_SIZE, HIDDEN_SIZE], pl.BF16],
            out: pl.Out[pl.Tensor[[BATCH_SIZE, HIDDEN_SIZE], pl.FP32]],
        ) -> pl.Tensor[[BATCH_SIZE, HIDDEN_SIZE], pl.FP32]:
            ffn_scratch = pl.create_tensor([BATCH_SIZE, INTER_SIZE], dtype=pl.BF16)

            # Single SPMD dispatch over the merged grid.
            with pl.spmd(GRID):
                out = self.kernel_ffn(post_norm, w_gate, w_up, w_down, ffn_scratch, out)

            return out

    return WseFFN


def build_tensor_specs(
    batch: int = BATCH,
    hidden_size: int = HIDDEN,
    intermediate_size: int = INTERMEDIATE,
):
    import torch  # type: ignore[import]
    from golden import TensorSpec

    return [
        TensorSpec(
            "post_norm", [batch, hidden_size], torch.bfloat16,
            init_value=lambda: torch.rand(batch, hidden_size) - 0.5,
        ),
        TensorSpec(
            "w_gate", [hidden_size, intermediate_size], torch.bfloat16,
            init_value=lambda: (torch.rand(hidden_size, intermediate_size) - 0.5) / (hidden_size ** 0.5),
        ),
        TensorSpec(
            "w_up", [hidden_size, intermediate_size], torch.bfloat16,
            init_value=lambda: (torch.rand(hidden_size, intermediate_size) - 0.5) / (hidden_size ** 0.5),
        ),
        TensorSpec(
            "w_down", [intermediate_size, hidden_size], torch.bfloat16,
            init_value=lambda: (torch.rand(intermediate_size, hidden_size) - 0.5) / (intermediate_size ** 0.5),
        ),
        TensorSpec("out", [batch, hidden_size], torch.float32, is_output=True),
    ]


def golden_wse_ffn(tensors):
    """PyTorch reference for wse-ffn SwiGLU feed-forward network.

    Implements SwiGLU gate + up projections followed by down projection.
    Chunked accumulation order matches the hardware kernel to minimize BF16/FP32 drift.
    """
    import torch

    post_norm = tensors["post_norm"]
    w_gate = tensors["w_gate"]
    w_up = tensors["w_up"]
    w_down = tensors["w_down"]

    batch = post_norm.shape[0]
    hidden_size = post_norm.shape[1]
    inter_size = w_gate.shape[1]

    k_chunk = K_CHUNK
    ffn_out_chunk = FFN_OUT_CHUNK

    ffn_bf16 = torch.zeros(batch, inter_size, dtype=torch.bfloat16)
    for o0 in range(0, inter_size, ffn_out_chunk):
        gate_acc = torch.zeros(batch, ffn_out_chunk, dtype=torch.float32)
        up_acc = torch.zeros(batch, ffn_out_chunk, dtype=torch.float32)
        for k0 in range(0, hidden_size, k_chunk):
            post_chunk = post_norm[:, k0:k0 + k_chunk].float()
            gate_acc += post_chunk @ w_gate[k0:k0 + k_chunk, o0:o0 + ffn_out_chunk].float()
            up_acc += post_chunk @ w_up[k0:k0 + k_chunk, o0:o0 + ffn_out_chunk].float()
        sigmoid = torch.reciprocal(torch.exp(-gate_acc) + 1.0)
        ffn_bf16[:, o0:o0 + ffn_out_chunk] = (gate_acc * sigmoid * up_acc).bfloat16()

    out = torch.zeros(batch, hidden_size, dtype=torch.float32)
    for d0 in range(0, hidden_size, k_chunk):
        down_acc = torch.zeros(batch, k_chunk, dtype=torch.float32)
        for o0 in range(0, inter_size, ffn_out_chunk):
            down_acc += ffn_bf16[:, o0:o0 + ffn_out_chunk].float() @ w_down[o0:o0 + ffn_out_chunk, d0:d0 + k_chunk].float()
        out[:, d0:d0 + k_chunk] = down_acc

    tensors["out"][:] = out


if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

    from golden import run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    args = parser.parse_args()

    result = run(
        program=build_wse_ffn_program(),
        specs=build_tensor_specs(),
        golden_fn=golden_wse_ffn,
        compile_cfg=dict(dump_passes=True),
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
        ),
        rtol=3e-3,
        atol=5e-3,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
