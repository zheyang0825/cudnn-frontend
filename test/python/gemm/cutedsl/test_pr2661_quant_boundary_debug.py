"""Diagnostic seed sweep for the PR 2661 PowLU MXFP8 forward case."""

import pytest
import torch

from gemm.cutedsl.test_grouped_gemm_glu import (
    _test_grouped_gemm_glu_dense_wrapper,
)
from gemm.cutedsl.test_grouped_gemm_swiglu_utils import (
    run_grouped_gemm_swiglu_ref,
)


@pytest.mark.parametrize("seed", range(32))
@pytest.mark.L0
def test_powlu_mxfp8_seed_sweep(request, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    _test_grouped_gemm_glu_dense_wrapper(
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
        request=request,
        act_func="powlu",
        power_m=2.5,
    )


@pytest.mark.L0
def test_powlu_mxfp8_seed4_strict(request):
    torch.manual_seed(4)
    torch.cuda.manual_seed_all(4)
    _test_grouped_gemm_glu_dense_wrapper(
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
        request=request,
        act_func="powlu",
        power_m=2.5,
    )


@pytest.mark.parametrize("seed", range(64))
@pytest.mark.L0
def test_swiglu_mxfp8_seed_sweep(request, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    _test_grouped_gemm_glu_dense_wrapper(
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
        request=request,
        act_func="swiglu",
    )


@pytest.mark.L0
def test_powlu_mxfp8_seed4_diagnostics(request):
    torch.manual_seed(4)
    torch.cuda.manual_seed_all(4)
    inputs, outputs, cfg = _test_grouped_gemm_glu_dense_wrapper(
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
        request=request,
        act_func="powlu",
        power_m=2.5,
        skip_reference=True,
    )
    reference = run_grouped_gemm_swiglu_ref(
        a_ref=inputs["a_ref"].float(),
        b_ref=inputs["b_ref"].float(),
        sfa_ref=inputs["sfa_ref"].float(),
        sfb_ref=inputs["sfb_ref"].float(),
        alpha_tensor=inputs["alpha_tensor"],
        prob_tensor=inputs["prob_tensor"],
        aligned_group_m_list=inputs["aligned_group_m_list"],
        valid_m=inputs["valid_m"],
        generate_sfd=True,
        norm_const_tensor=inputs["norm_const_tensor"],
        c_dtype=cfg["c_dtype"],
        d_dtype=cfg["d_dtype"],
        sf_vec_size=cfg["sf_vec_size"],
        sf_dtype=cfg["sf_dtype"],
        act_func="powlu",
        power_m=2.5,
    )
    actual = outputs["d_tensor"].float()
    expected = reference["d_ref"].to(torch.float8_e4m3fn).float()
    mismatch = torch.nonzero(actual != expected)
    assert mismatch.shape == (1, 3)
    row, output_col, batch = (int(value) for value in mismatch[0])

    gate_col = (output_col // 32) * 64 + output_col % 32
    up_col = gate_col + 32
    c_kernel = outputs["c_tensor"].float()
    c_ref = reference["c_ref"].float()
    prob = inputs["prob_tensor"][row, 0, batch].float()

    def exact_powlu(gate, up):
        safe_gate = torch.clamp(gate, min=torch.finfo(torch.float32).tiny)
        positive = torch.pow(safe_gate, 2.5 / (torch.sqrt(safe_gate) + 1.0))
        base = torch.where(gate > 0, positive, gate)
        return base * torch.sigmoid(gate) * up * prob

    print(
        "diagnostic",
        dict(
            index=(row, output_col, batch),
            actual=float(actual[row, output_col, batch]),
            expected=float(expected[row, output_col, batch]),
            c_kernel_gate=float(c_kernel[row, gate_col, batch]),
            c_ref_gate=float(c_ref[row, gate_col, batch]),
            c_kernel_up=float(c_kernel[row, up_col, batch]),
            c_ref_up=float(c_ref[row, up_col, batch]),
            exact_from_kernel_c=float(
                exact_powlu(
                    c_kernel[row, gate_col, batch],
                    c_kernel[row, up_col, batch],
                )
            ),
            exact_from_ref_c=float(
                exact_powlu(
                    c_ref[row, gate_col, batch],
                    c_ref[row, up_col, batch],
                )
            ),
            normalized_ref=float(reference["d_ref"][row, output_col, batch]),
            prob=float(prob),
        ),
    )
