// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef AOCOMMON_AVX256_MATRIX_COMPLEX_DOUBLE_2_X_2_H
#define AOCOMMON_AVX256_MATRIX_COMPLEX_DOUBLE_2_X_2_H

/**
 * @file Implements a 2x2 Matrix with complex double values.
 *
 * This class is based on @ref aocommon::MC2x2 but uses AVX-256 instructions.
 *
 * @note The class only implements a subset of the matrix operations. Other
 * operations will be added on a when-needed basis.
 *
 * @todo Move this to aocommon then the class has matured further.
 */

#include "common/avx256/VectorComplexDouble2.h"

#include <aocommon/matrix2x2.h>

#include <array>
#include <cassert>
#include <complex>
#include <ostream>

#if defined(__AVX2__)

#include <immintrin.h>

namespace aocommon::Avx256 {

class MaxtrixComplexDouble2x2 {
 public:
  /* implicit */ MaxtrixComplexDouble2x2(
      std::array<VectorComplexDouble2, 2> data) noexcept
      : data_{data} {}

  explicit MaxtrixComplexDouble2x2(std::complex<double> a,
                                   std::complex<double> b,
                                   std::complex<double> c,
                                   std::complex<double> d) noexcept
      : data_{{VectorComplexDouble2{a, b}, VectorComplexDouble2{c, d}}} {}

  explicit MaxtrixComplexDouble2x2(const std::complex<float> matrix[4]) noexcept
      : data_{{VectorComplexDouble2{std::addressof(matrix[0])},
               VectorComplexDouble2{std::addressof(matrix[2])}}} {}

  explicit MaxtrixComplexDouble2x2(
      const std::complex<double> matrix[4]) noexcept
      : data_{{VectorComplexDouble2{std::addressof(matrix[0])},
               VectorComplexDouble2{std::addressof(matrix[2])}}} {}

  explicit MaxtrixComplexDouble2x2(const aocommon::MC2x2& matrix) noexcept
      : MaxtrixComplexDouble2x2(matrix.Data()) {}

  [[nodiscard]] std::complex<double> operator[](size_t index) const noexcept {
    assert(index < 4 && "Index out of bounds.");
    size_t array = index / 2;
    index %= 2;
    return data_[array][index];
  }

  [[nodiscard]] MaxtrixComplexDouble2x2 Conjugate() const noexcept {
    return std::array<VectorComplexDouble2, 2>{data_[0].Conjugate(),
                                               data_[1].Conjugate()};
  }

  [[nodiscard]] MaxtrixComplexDouble2x2 Transpose() const noexcept {
    // Note the compiler uses intrinsics without assistance.
    return MaxtrixComplexDouble2x2{(*this)[0], (*this)[2], (*this)[1],
                                   (*this)[3]};
  }

  [[nodiscard]] MaxtrixComplexDouble2x2 HermitianTranspose() const noexcept {
    return Transpose().Conjugate();
  }

  /// @returns the Frobenius norm of the matrix.
  [[nodiscard]] double Norm() const noexcept {
    // Norm Matrix Complex 2x2
    // Norm(a) + Norm(b) + Norm(c) + Norm(d)
    //
    // Norm is using C++'s definition of std::complex<T>::norm(). This norm is
    // also known as the 'field norm' or 'absolute square'. Norm is defined as
    // a.re * a.re + a.im * a.im
    //
    // Note if we want to do this accoring to the rules above some shuffing
    // needs to be done. Instead we can consider the underlaying data an array
    // of 8 doubles. Then Norm becomes
    //
    // -- 7
    // \.
    //  .  a[n] * a[n]
    // /
    // -- n = 0
    //
    // and no shuffling in needed instead use the following algorithm
    //
    // hi = data_[0]
    // lo = data_[1]
    //
    // hi = hi * hi
    // lo = lo * lo
    //
    // tmp = hi + lo
    // ret = std::accumulate(&tmp[0], &tmp[4], 0.0); // not possible in C++
    //
    // instead of calculating tmp as described it can be done by
    // hi = lo * lo + hi

    __m256d hi = static_cast<__m256d>(data_[0]);
    __m256d lo = static_cast<__m256d>(data_[1]);

    hi *= hi;
    hi = _mm256_fmadd_pd(lo, lo, hi);

    // Summing the 4 elements in hi can be simply done by
    // return hi[0] + hi[1] + hi[2] + hi[3]
    //
    // however this is slow, it's more efficient to permutate the data and use
    // vector adding. The instruction set has a hadd operation, but this is
    // slow too. Instead use register permutations and additons. The entries
    // marked with - in the table mean we don't care about the contents. The
    // result will be stored in hi[0]:
    //
    // hi | a             | b     | c | d |
    // lo | c             | d     | - | - |
    //    --------------------------------- +
    // hi | a + c         | b + d | - | - |
    // lo | b + d         | -     | - | - |
    //    --------------------------------- +
    // hi | a + c + b + d | -     | - | - |

    lo = _mm256_permute4x64_pd(hi, 0b11'10);
    hi += lo;

    __m128d ret = _mm256_castpd256_pd128(hi);
    ret += _mm_permute_pd(ret, 0b01);
    return ret[0];
  }

  MaxtrixComplexDouble2x2& operator+=(MaxtrixComplexDouble2x2 value) noexcept {
    data_[0] += value.data_[0];
    data_[1] += value.data_[1];
    return *this;
  }

  MaxtrixComplexDouble2x2& operator-=(MaxtrixComplexDouble2x2 value) noexcept {
    data_[0] -= value.data_[0];
    data_[1] -= value.data_[1];
    return *this;
  }

  [[nodiscard]] friend MaxtrixComplexDouble2x2 operator+(
      MaxtrixComplexDouble2x2 lhs, MaxtrixComplexDouble2x2 rhs) noexcept {
    return lhs += rhs;
  }

  [[nodiscard]] friend MaxtrixComplexDouble2x2 operator-(
      MaxtrixComplexDouble2x2 lhs, MaxtrixComplexDouble2x2 rhs) noexcept {
    return lhs -= rhs;
  }

  [[nodiscard]] friend MaxtrixComplexDouble2x2 operator*(
      MaxtrixComplexDouble2x2 lhs, MaxtrixComplexDouble2x2 rhs) noexcept {
    // The 2x2 matrix multiplication is done using the following algorithm.

    // High:
    // ret.a = lhs.a * rhs.a + lhs.b * rhs.c
    // ret.b = lhs.a * rhs.b + lhs.b * rhs.d
    //       | hc1   | hc2   | hc3   | hc4   |
    //       | hs1           | hs2           |

    // Low:
    // ret.c = lhs.c * rhs.a + lhs.d * rhs.c
    // ret.d = lhs.c * rhs.b + lhs.d * rhs.d
    //       | lc1   | lc2   | lc3   | lc4   |
    //       | ls1           | ls2           |

    // High:
    VectorComplexDouble2 hc1{lhs[0], lhs[0]};
    VectorComplexDouble2 hc2{rhs[0], rhs[1]};
    VectorComplexDouble2 hs1 = hc1 * hc2;

    VectorComplexDouble2 hc3{lhs[1], lhs[1]};
    VectorComplexDouble2 hc4{rhs[2], rhs[3]};
    VectorComplexDouble2 hs2 = hc3 * hc4;

    VectorComplexDouble2 hr = hs1 + hs2;

    // Low:
    VectorComplexDouble2 lc1{lhs[2], lhs[2]};
    VectorComplexDouble2 lc2{rhs[0], rhs[1]};
    VectorComplexDouble2 ls1 = lc1 * lc2;

    VectorComplexDouble2 lc3{lhs[3], lhs[3]};
    VectorComplexDouble2 lc4{rhs[2], rhs[3]};
    VectorComplexDouble2 ls2 = lc3 * lc4;

    VectorComplexDouble2 lr = ls1 + ls2;

    return std::array<VectorComplexDouble2, 2>{hr, lr};
  }

  [[nodiscard]] friend bool operator==(MaxtrixComplexDouble2x2 lhs,
                                       MaxtrixComplexDouble2x2 rhs) noexcept {
    return lhs.data_ == rhs.data_;
  }

  friend std::ostream& operator<<(std::ostream& output,
                                  MaxtrixComplexDouble2x2 value) {
    output << "[{" << value[0] << ", " << value[1] << "}, {" << value[2] << ", "
           << value[3] << "}]";
    return output;
  }

 private:
  std::array<VectorComplexDouble2, 2> data_;
};

/// MC2x2Base compatibility wrapper.
inline MaxtrixComplexDouble2x2 HermTranspose(
    MaxtrixComplexDouble2x2 matrix) noexcept {
  return matrix.HermitianTranspose();
}

/**
 * MC2x2Base compatibility wrapper.
 *
 * @returns the Frobenius norm of the matrix.
 */
inline double Norm(MaxtrixComplexDouble2x2 matrix) noexcept {
  return matrix.Norm();
}

}  // namespace aocommon::Avx256

#endif  // defined(__AVX2__)

#endif  // AOCOMMON_AVX256_MATRIX_COMPLEX_DOUBLE_2_X_2_H