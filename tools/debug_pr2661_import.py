#!/usr/bin/env python3
"""Print the full import failure for the PR 2661 overlay environment."""

import traceback


try:
    from cudnn import grouped_gemm_glu_wrapper_sm100  # noqa: F401
except Exception:
    traceback.print_exc()
    raise

print("grouped_gemm_glu_wrapper_sm100 import succeeded")
