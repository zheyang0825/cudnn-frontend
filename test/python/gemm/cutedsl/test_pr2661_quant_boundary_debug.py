"""Diagnostic seed sweep for the PR 2661 PowLU MXFP8 forward case."""

import pytest
import torch

from gemm.cutedsl.test_grouped_gemm_glu import (
    _test_grouped_gemm_glu_dense_wrapper,
)


@pytest.mark.parametrize("seed", range(32))
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
