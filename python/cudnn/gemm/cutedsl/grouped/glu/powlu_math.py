# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PowLU math helpers used by grouped-GLU numerical experiments."""

from cutlass._mlir.dialects import llvm
from cutlass.cute.typing import Float32
from cutlass.cutlass_dsl import T, dsl_user_op


@dsl_user_op
def nv_powf_f32(value, exponent, *, loc=None, ip=None):
    """Positive-finite FTZ core of CUDA libdevice ``__nv_powf``."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [
                Float32(value).ir_value(loc=loc, ip=ip),
                Float32(exponent).ir_value(loc=loc, ip=ip),
            ],
            """{
            .reg .pred p<4>;
            .reg .b32 xabs, mantissa, normalized, exponent_bits;
            .reg .b32 exponent_f, normalized_half, exponent_plus_one;
            .reg .b32 delta, denominator, inverse, twice_delta;
            .reg .b32 ratio, ratio2, polynomial, temporary;
            .reg .b32 residual0, residual1, log_core, log_error;
            .reg .b32 log_hi_part, log_lo_part, log_hi, log_lo;
            .reg .b32 product_hi, product_error, product;
            .reg .b32 exp_input, scaled, integer, abs_integer;
            .reg .b32 integer_limit, signed_limit, reduced_integer;
            .reg .b32 remainder, fraction, scale_bits, exp2_value;
            .reg .b32 exp_value, correction, corrected;

            abs.ftz.f32 xabs, $1;
            and.b32 mantissa, xabs, 8388607;
            or.b32 normalized, mantissa, 1065353216;
            shr.u32 exponent_bits, xabs, 23;
            cvt.rn.f32.u32 exponent_f, exponent_bits;
            add.f32 exponent_f, exponent_f, 0fC2FE0000;
            setp.gt.f32 p0, normalized, 0f3FB504F3;
            mul.f32 normalized_half, normalized, 0f3F000000;
            add.f32 exponent_plus_one, exponent_f, 0f3F800000;
            selp.f32 exponent_f, exponent_plus_one, exponent_f, p0;
            selp.f32 normalized, normalized_half, normalized, p0;
            add.f32 delta, normalized, 0fBF800000;
            add.f32 denominator, normalized, 0f3F800000;
            rcp.approx.ftz.f32 inverse, denominator;
            add.f32 twice_delta, delta, delta;
            mul.f32 ratio, inverse, twice_delta;
            mul.f32 ratio2, ratio, ratio;
            mov.b32 polynomial, 0f3C4CAF63;
            mov.b32 temporary, 0f3B18F0FE;
            fma.rn.ftz.f32 polynomial, temporary, ratio2, polynomial;
            mov.b32 temporary, 0f3DAAAABD;
            fma.rn.ftz.f32 polynomial, polynomial, ratio2, temporary;
            mul.rn.ftz.f32 polynomial, polynomial, ratio2;
            mul.rn.ftz.f32 polynomial, polynomial, ratio;
            neg.f32 temporary, inverse;
            fma.rn.f32 residual0, temporary, twice_delta, delta;
            add.f32 residual0, residual0, residual0;
            neg.f32 temporary, ratio;
            fma.rn.ftz.f32 residual0, temporary, delta, residual0;
            mul.rn.ftz.f32 residual0, inverse, residual0;
            fma.rn.f32 log_core, inverse, twice_delta, polynomial;
            neg.f32 temporary, log_core;
            fma.rn.f32 residual1, inverse, twice_delta, temporary;
            add.f32 residual1, polynomial, residual1;
            add.f32 residual1, residual0, residual1;
            add.f32 temporary, log_core, residual1;
            sub.f32 log_error, log_core, temporary;
            add.f32 log_error, residual1, log_error;
            mov.b32 log_hi_part, 0f3F317200;
            mul.rn.ftz.f32 log_hi_part, exponent_f, log_hi_part;
            mov.b32 log_lo_part, 0f35BFBE8E;
            mul.rn.ftz.f32 log_lo_part, exponent_f, log_lo_part;
            add.f32 log_hi, log_hi_part, temporary;
            sub.f32 residual0, log_hi_part, log_hi;
            add.f32 residual0, temporary, residual0;
            add.f32 log_error, log_error, residual0;
            add.f32 log_lo_part, log_lo_part, log_error;
            add.f32 temporary, log_hi, log_lo_part;
            sub.f32 residual0, log_hi, temporary;
            add.f32 log_lo, log_lo_part, residual0;
            mov.b32 log_hi, temporary;

            mul.rn.ftz.f32 product_hi, $2, log_hi;
            neg.f32 temporary, product_hi;
            fma.rn.ftz.f32 product_error, $2, log_hi, temporary;
            fma.rn.ftz.f32 product_error, $2, log_lo, product_error;
            mul.rn.ftz.f32 product_error, product_error, 0f3F800000;
            add.rn.ftz.f32 product, product_hi, product_error;
            setp.eq.s32 p1, product, 1118925336;
            selp.f32 exp_input, 0f42B17217, product, p1;
            mul.rn.ftz.f32 scaled, exp_input, 0f3FB8AA3B;
            cvt.rzi.f32.f32 integer, scaled;
            abs.ftz.f32 abs_integer, integer;
            setp.gt.f32 p2, abs_integer, 0f42FC0000;
            mov.b32 integer_limit, 0f42FC0000;
            copysign.f32 signed_limit, integer, integer_limit;
            selp.f32 reduced_integer, signed_limit, integer, p2;
            fma.rn.ftz.f32 remainder, reduced_integer, 0fBF317218, exp_input;
            fma.rn.ftz.f32 remainder, reduced_integer, 0f3102E308, remainder;
            mul.f32 fraction, remainder, 0f3FB8AA3B;
            add.f32 scale_bits, reduced_integer, 0f4B40007F;
            shl.b32 scale_bits, scale_bits, 23;
            ex2.approx.ftz.f32 exp2_value, fraction;
            mul.f32 exp_value, exp2_value, scale_bits;
            neg.f32 temporary, product;
            add.rn.ftz.f32 correction, product_hi, temporary;
            add.rn.ftz.f32 correction, correction, product_error;
            add.f32 corrected, correction, 0f37000000;
            selp.f32 correction, corrected, correction, p1;
            fma.rn.ftz.f32 corrected, exp_value, correction, exp_value;
            setp.eq.f32 p3, exp_value, 0f7F800000;
            selp.f32 $0, 0f7F800000, corrected, p3;
            }""",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )
