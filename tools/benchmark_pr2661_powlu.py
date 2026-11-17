#!/usr/bin/env python3
"""Time the PR 2661 fully fused PowLU and dPowLU wrappers on SM100."""

from __future__ import annotations

import argparse
import statistics

import torch
from cuda.bindings import driver as cuda
from cudnn import grouped_gemm_dglu_wrapper_sm100, grouped_gemm_glu_wrapper_sm100
from gemm.cutedsl.test_grouped_gemm_dglu import (
    _test_grouped_gemm_dglu_dense_wrapper,
)
from gemm.cutedsl.test_grouped_gemm_glu import (
    _test_grouped_gemm_glu_dense_wrapper,
)


class _Config:
    @staticmethod
    def getoption(name, default=None):
        return True if name == "--skip-ref" else default


class _Request:
    config = _Config()


def _time(function, warmup: int, iterations: int, samples: int) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    timings = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            function()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end) / iterations)
    return statistics.median(timings)


def _forward(args, stream):
    inputs, _, cfg = _test_grouped_gemm_glu_dense_wrapper(
        ab_dtype=torch.float8_e4m3fn,
        c_dtype=torch.bfloat16,
        d_dtype=torch.float8_e4m3fn,
        cd_major="n",
        acc_dtype=torch.float32,
        mma_tiler_mn=(256, 256),
        cluster_shape_mn=(2, 1),
        sf_vec_size=32,
        sf_dtype=torch.float8_e8m0fnu,
        vector_f32=False,
        discrete_col_sfd=False,
        request=_Request(),
        cfg_overrides={
            "n": 2 * args.ffn_hidden,
            "k": args.hidden,
            "l": args.experts,
            "group_m_list": [args.tokens_per_expert] * args.experts,
        },
        act_func="powlu",
        power_m=args.power_m,
        skip_reference=True,
    )

    def run():
        return grouped_gemm_glu_wrapper_sm100(
            a_tensor=inputs["a_tensor"],
            sfa_tensor=inputs["sfa_tensor"],
            padded_offsets=inputs["padded_offsets_tensor"],
            alpha_tensor=inputs["alpha_tensor"],
            b_tensor=inputs["b_tensor"],
            sfb_tensor=inputs["sfb_tensor"],
            norm_const_tensor=inputs["norm_const_tensor"],
            prob_tensor=inputs["prob_tensor"],
            acc_dtype=cfg["acc_dtype"],
            c_dtype=cfg["c_dtype"],
            d_dtype=cfg["d_dtype"],
            cd_major=cfg["cd_major"],
            mma_tiler_mn=cfg["mma_tiler_mn"],
            cluster_shape_mn=cfg["cluster_shape_mn"],
            sf_vec_size=cfg["sf_vec_size"],
            vector_f32=cfg["vector_f32"],
            m_aligned=cfg["m_aligned"],
            discrete_col_sfd=cfg["discrete_col_sfd"],
            act_func="powlu",
            power_m=args.power_m,
            current_stream=stream,
        )

    return run


def _backward(args, stream):
    inputs, outputs, cfg = _test_grouped_gemm_dglu_dense_wrapper(
        ab_dtype=torch.float8_e4m3fn,
        c_dtype=torch.bfloat16,
        d_dtype=torch.float8_e4m3fn,
        b_major="k",
        cd_major="n",
        acc_dtype=torch.float32,
        mma_tiler_mn=(256, 256),
        cluster_shape_mn=(2, 1),
        sf_vec_size=32,
        sf_dtype=torch.float8_e8m0fnu,
        vector_f32=False,
        discrete_col_sfd=False,
        request=_Request(),
        cfg_overrides={
            "n": args.ffn_hidden,
            "k": args.hidden,
            "l": args.experts,
            "group_m_list": [args.tokens_per_expert] * args.experts,
        },
        act_func="dpowlu",
        power_m=args.power_m,
    )
    dprob = outputs["dprob_tensor"]

    def run():
        dprob.zero_()
        return grouped_gemm_dglu_wrapper_sm100(
            a_tensor=inputs["a_tensor"],
            c_tensor=inputs["c_tensor"],
            sfa_tensor=inputs["sfa_tensor"],
            padded_offsets=inputs["padded_offsets_tensor"],
            alpha_tensor=inputs["alpha_tensor"],
            beta_tensor=inputs["beta_tensor"],
            prob_tensor=inputs["prob_tensor"],
            dprob_tensor=dprob,
            b_tensor=inputs["b_tensor"],
            sfb_tensor=inputs["sfb_tensor"],
            norm_const_tensor=inputs["norm_const_tensor"],
            acc_dtype=cfg["acc_dtype"],
            d_dtype=cfg["d_dtype"],
            cd_major=cfg["cd_major"],
            mma_tiler_mn=cfg["mma_tiler_mn"],
            cluster_shape_mn=cfg["cluster_shape_mn"],
            sf_vec_size=cfg["sf_vec_size"],
            vector_f32=cfg["vector_f32"],
            m_aligned=cfg["m_aligned"],
            discrete_col_sfd=cfg["discrete_col_sfd"],
            act_func="dpowlu",
            power_m=args.power_m,
            current_stream=stream,
        )

    return run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--tokens-per-expert", type=int, default=8192)
    parser.add_argument("--hidden", type=int, default=384)
    parser.add_argument("--ffn-hidden", type=int, default=5120)
    parser.add_argument("--power-m", type=float, default=2.5)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--samples", type=int, default=3)
    args = parser.parse_args()

    torch.manual_seed(1234)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    forward = _forward(args, stream)
    backward = _backward(args, stream)
    forward_ms = _time(forward, args.warmup, args.iterations, args.samples)
    backward_ms = _time(backward, args.warmup, args.iterations, args.samples)
    print(
        f"M={args.experts * args.tokens_per_expert} hidden={args.hidden} "
        f"ffn_hidden={args.ffn_hidden} power_m={args.power_m}"
    )
    print(f"forward_ms={forward_ms:.6f}")
    print(f"backward_ms={backward_ms:.6f}")
    print(f"total_ms={forward_ms + backward_ms:.6f}")


if __name__ == "__main__":
    main()
