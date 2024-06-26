// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include "common/avx256/MatrixComplexDouble2x2.h"

#include <type_traits>

#include <boost/test/unit_test.hpp>

#if defined(__AVX2__)

// operator[] is tested in other tests.

BOOST_AUTO_TEST_SUITE(MatrixComplexDouble2x2)

BOOST_AUTO_TEST_CASE(constructor_4_complex_doubles) {
  static_assert(
      std::is_nothrow_constructible_v<
          aocommon::Avx256::MatrixComplexDouble2x2, std::complex<double>,
          std::complex<double>, std::complex<double>, std::complex<double>>);

  const aocommon::Avx256::MatrixComplexDouble2x2 result{
      std::complex<double>{-1.0, 1.0}, std::complex<double>{3.75, -3.75},
      std::complex<double>{99.0, -99.0}, std::complex<double>{1.5, -1.5}};

  BOOST_TEST(result[0] == (std::complex<double>{-1.0, 1.0}));
  BOOST_TEST(result[1] == (std::complex<double>{3.75, -3.75}));
  BOOST_TEST(result[2] == (std::complex<double>{99.0, -99.0}));
  BOOST_TEST(result[3] == (std::complex<double>{1.5, -1.5}));
}

BOOST_AUTO_TEST_CASE(constructor_complex_float_pointer) {
  static_assert(
      std::is_nothrow_constructible_v<aocommon::Avx256::MatrixComplexDouble2x2,
                                      const std::complex<float>[4]>);
  const std::complex<float> input[] = {
      {-1.0, 1.0}, {3.75, -3.75}, {99.0, -99.0}, {1.5, -1.5}};
  const aocommon::Avx256::MatrixComplexDouble2x2 result{input};

  BOOST_TEST(result[0] == (std::complex<double>{-1.0, 1.0}));
  BOOST_TEST(result[1] == (std::complex<double>{3.75, -3.75}));
  BOOST_TEST(result[2] == (std::complex<double>{99.0, -99.0}));
  BOOST_TEST(result[3] == (std::complex<double>{1.5, -1.5}));
}

BOOST_AUTO_TEST_CASE(constructor_complex_double_pointer) {
  static_assert(
      std::is_nothrow_constructible_v<aocommon::Avx256::MatrixComplexDouble2x2,
                                      const std::complex<double>[4]>);
  const std::complex<double> input[] = {
      {-1.0, 1.0}, {3.75, -3.75}, {99.0, -99.0}, {1.5, -1.5}};
  const aocommon::Avx256::MatrixComplexDouble2x2 result{input};

  BOOST_TEST(result[0] == (std::complex<double>{-1.0, 1.0}));
  BOOST_TEST(result[1] == (std::complex<double>{3.75, -3.75}));
  BOOST_TEST(result[2] == (std::complex<double>{99.0, -99.0}));
  BOOST_TEST(result[3] == (std::complex<double>{1.5, -1.5}));
}

BOOST_AUTO_TEST_CASE(constructor_MC2x2) {
  static_assert(
      std::is_nothrow_constructible_v<aocommon::Avx256::MatrixComplexDouble2x2,
                                      const aocommon::MC2x2&>);
  const aocommon::MC2x2 input{
      {-1.0, 1.0}, {3.75, -3.75}, {99.0, -99.0}, {1.5, -1.5}};
  const aocommon::Avx256::MatrixComplexDouble2x2 result{input};

  BOOST_TEST(result[0] == (std::complex<double>{-1.0, 1.0}));
  BOOST_TEST(result[1] == (std::complex<double>{3.75, -3.75}));
  BOOST_TEST(result[2] == (std::complex<double>{99.0, -99.0}));
  BOOST_TEST(result[3] == (std::complex<double>{1.5, -1.5}));
}

BOOST_AUTO_TEST_CASE(conjugate) {
  static_assert(noexcept(aocommon::Avx256::MatrixComplexDouble2x2{
      static_cast<const std::complex<double>*>(nullptr)}
                             .Conjugate()));

  BOOST_CHECK_EQUAL((aocommon::Avx256::MatrixComplexDouble2x2{
                        {1.0, 2.0}, {10, 11}, {100, 101}, {1000, 1001}}
                         .Conjugate()),
                    (aocommon::Avx256::MatrixComplexDouble2x2{
                        {1.0, -2.0}, {10, -11}, {100, -101}, {1000, -1001}}));

  BOOST_CHECK_EQUAL((aocommon::Avx256::MatrixComplexDouble2x2{
                        {-1.0, 2.0}, {10, 11}, {100, 101}, {1000, 1001}}
                         .Conjugate()),
                    (aocommon::Avx256::MatrixComplexDouble2x2{
                        {-1.0, -2.0}, {10, -11}, {100, -101}, {1000, -1001}}));
  BOOST_CHECK_EQUAL((aocommon::Avx256::MatrixComplexDouble2x2{
                        {-1.0, -2.0}, {10, 11}, {100, 101}, {1000, 1001}}
                         .Conjugate()),
                    (aocommon::Avx256::MatrixComplexDouble2x2{
                        {-1.0, 2.0}, {10, -11}, {100, -101}, {1000, -1001}}));

  BOOST_CHECK_EQUAL((aocommon::Avx256::MatrixComplexDouble2x2{
                        {-1.0, -2.0}, {-10, 11}, {100, 101}, {1000, 1001}}
                         .Conjugate()),
                    (aocommon::Avx256::MatrixComplexDouble2x2{
                        {-1.0, 2.0}, {-10, -11}, {100, -101}, {1000, -1001}}));
  BOOST_CHECK_EQUAL((aocommon::Avx256::MatrixComplexDouble2x2{
                        {-1.0, -2.0}, {-10, -11}, {100, 101}, {1000, 1001}}
                         .Conjugate()),
                    (aocommon::Avx256::MatrixComplexDouble2x2{
                        {-1.0, 2.0}, {-10, 11}, {100, -101}, {1000, -1001}}));

  BOOST_CHECK_EQUAL((aocommon::Avx256::MatrixComplexDouble2x2{
                        {-1.0, -2.0}, {-10, 11}, {-100, 101}, {1000, 1001}}
                         .Conjugate()),
                    (aocommon::Avx256::MatrixComplexDouble2x2{
                        {-1.0, 2.0}, {-10, -11}, {-100, -101}, {1000, -1001}}));
  BOOST_CHECK_EQUAL((aocommon::Avx256::MatrixComplexDouble2x2{
                        {-1.0, -2.0}, {-10, -11}, {-100, -101}, {1000, 1001}}
                         .Conjugate()),
                    (aocommon::Avx256::MatrixComplexDouble2x2{
                        {-1.0, 2.0}, {-10, 11}, {-100, 101}, {1000, -1001}}));

  BOOST_CHECK_EQUAL(
      (aocommon::Avx256::MatrixComplexDouble2x2{
          {-1.0, -2.0}, {-10, 11}, {-100, 101}, {-1000, 1001}}
           .Conjugate()),
      (aocommon::Avx256::MatrixComplexDouble2x2{
          {-1.0, 2.0}, {-10, -11}, {-100, -101}, {-1000, -1001}}));
  BOOST_CHECK_EQUAL((aocommon::Avx256::MatrixComplexDouble2x2{
                        {-1.0, -2.0}, {-10, -11}, {-100, -101}, {-1000, -1001}}
                         .Conjugate()),
                    (aocommon::Avx256::MatrixComplexDouble2x2{
                        {-1.0, 2.0}, {-10, 11}, {-100, 101}, {-1000, 1001}}));
}

BOOST_AUTO_TEST_CASE(transpose) {
  static_assert(noexcept(aocommon::Avx256::MatrixComplexDouble2x2{
      static_cast<const std::complex<double>*>(nullptr)}
                             .Transpose()));
  const aocommon::MC2x2 input{
      std::complex<double>{1.0, 2.0}, std::complex<double>{10, 11},
      std::complex<double>{100, 101}, std::complex<double>{1000, 1001}};

  const aocommon::Avx256::MatrixComplexDouble2x2 expected{input.Transpose()};

  const aocommon::Avx256::MatrixComplexDouble2x2 result =
      aocommon::Avx256::MatrixComplexDouble2x2{input}.Transpose();

  BOOST_CHECK_EQUAL(result, expected);
}

BOOST_AUTO_TEST_CASE(hermitian_transpose) {
  static_assert(noexcept(aocommon::Avx256::MatrixComplexDouble2x2{
      static_cast<const std::complex<double>*>(nullptr)}
                             .HermitianTranspose()));
  const std::vector<aocommon::MC2x2> inputs{
      aocommon::MC2x2{
          std::complex<double>{1.0, 2.0}, std::complex<double>{10, 11},
          std::complex<double>{100, 101}, std::complex<double>{1000, 1001}},
      aocommon::MC2x2{
          std::complex<double>{-1.0, 2.0}, std::complex<double>{10, 11},
          std::complex<double>{100, 101}, std::complex<double>{1000, 1001}},
      aocommon::MC2x2{
          std::complex<double>{-1.0, -2.0}, std::complex<double>{10, 11},
          std::complex<double>{100, 101}, std::complex<double>{1000, 1001}},
      aocommon::MC2x2{
          std::complex<double>{-1.0, -2.0}, std::complex<double>{-10, 11},
          std::complex<double>{100, 101}, std::complex<double>{1000, 1001}},
      aocommon::MC2x2{
          std::complex<double>{-1.0, -2.0}, std::complex<double>{-10, -11},
          std::complex<double>{100, 101}, std::complex<double>{1000, 1001}},
      aocommon::MC2x2{
          std::complex<double>{-1.0, -2.0}, std::complex<double>{-10, -11},
          std::complex<double>{-100, 101}, std::complex<double>{1000, 1001}},
      aocommon::MC2x2{
          std::complex<double>{-1.0, -2.0}, std::complex<double>{-10, -11},
          std::complex<double>{-100, -101}, std::complex<double>{1000, 1001}},
      aocommon::MC2x2{
          std::complex<double>{-1.0, -2.0}, std::complex<double>{-10, -11},
          std::complex<double>{-100, -101}, std::complex<double>{-1000, 1001}},
      aocommon::MC2x2{std::complex<double>{-1.0, -2.0},
                      std::complex<double>{-10, -11},
                      std::complex<double>{-100, -101},
                      std::complex<double>{-1000, -1001}}};

  for (const auto& input : inputs) {
    const aocommon::Avx256::MatrixComplexDouble2x2 expected{
        input.HermTranspose()};

    const aocommon::Avx256::MatrixComplexDouble2x2 result =
        aocommon::Avx256::MatrixComplexDouble2x2{input}.HermitianTranspose();

    BOOST_CHECK_EQUAL(result, expected);
  }
}

BOOST_AUTO_TEST_CASE(invert) {
  static_assert(noexcept(aocommon::Avx256::MatrixComplexDouble2x2{
      static_cast<const std::complex<double>*>(nullptr)}
                             .Invert()));
  const aocommon::MC2x2 input{
      std::complex<double>{1.0, 2.0}, std::complex<double>{10, 11},
      std::complex<double>{100, 101}, std::complex<double>{1000, 1001}};

  const aocommon::Avx256::MatrixComplexDouble2x2 expected{[&] {
    aocommon::MC2x2 result{input};
    bool b = result.Invert();
    BOOST_REQUIRE(b);
    return result;
  }()};

  const aocommon::Avx256::MatrixComplexDouble2x2 result{[&] {
    aocommon::Avx256::MatrixComplexDouble2x2 result{input};
    bool b = result.Invert();
    BOOST_REQUIRE(b);
    return result;
  }()};

  BOOST_CHECK_CLOSE(result[0].real(), expected[0].real(), 1e-3);
  BOOST_CHECK_CLOSE(result[0].imag(), expected[0].imag(), 1e-3);
  BOOST_CHECK_CLOSE(result[1].real(), expected[1].real(), 1e-3);
  BOOST_CHECK_CLOSE(result[1].imag(), expected[1].imag(), 1e-3);
  BOOST_CHECK_CLOSE(result[2].real(), expected[2].real(), 1e-3);
  BOOST_CHECK_CLOSE(result[2].imag(), expected[2].imag(), 1e-3);
  BOOST_CHECK_CLOSE(result[3].real(), expected[3].real(), 1e-3);
  BOOST_CHECK_CLOSE(result[3].imag(), expected[3].imag(), 1e-3);
}

BOOST_AUTO_TEST_CASE(norm) {
  static_assert(noexcept(aocommon::Avx256::MatrixComplexDouble2x2{
      static_cast<const std::complex<double>*>(nullptr)}
                             .Norm()));
  const std::vector<aocommon::MC2x2> inputs{
      aocommon::MC2x2{
          std::complex<double>{1.0, 2.0}, std::complex<double>{10, 11},
          std::complex<double>{100, 101}, std::complex<double>{1000, 1001}},
      aocommon::MC2x2{
          std::complex<double>{-1.0, 2.0}, std::complex<double>{10, 11},
          std::complex<double>{100, 101}, std::complex<double>{1000, 1001}},
      aocommon::MC2x2{
          std::complex<double>{-1.0, -2.0}, std::complex<double>{10, 11},
          std::complex<double>{100, 101}, std::complex<double>{1000, 1001}},
      aocommon::MC2x2{
          std::complex<double>{-1.0, -2.0}, std::complex<double>{-10, 11},
          std::complex<double>{100, 101}, std::complex<double>{1000, 1001}},
      aocommon::MC2x2{
          std::complex<double>{-1.0, -2.0}, std::complex<double>{-10, -11},
          std::complex<double>{100, 101}, std::complex<double>{1000, 1001}},
      aocommon::MC2x2{
          std::complex<double>{-1.0, -2.0}, std::complex<double>{-10, -11},
          std::complex<double>{-100, 101}, std::complex<double>{1000, 1001}},
      aocommon::MC2x2{
          std::complex<double>{-1.0, -2.0}, std::complex<double>{-10, -11},
          std::complex<double>{-100, -101}, std::complex<double>{1000, 1001}},
      aocommon::MC2x2{
          std::complex<double>{-1.0, -2.0}, std::complex<double>{-10, -11},
          std::complex<double>{-100, -101}, std::complex<double>{-1000, 1001}},
      aocommon::MC2x2{std::complex<double>{-1.0, -2.0},
                      std::complex<double>{-10, -11},
                      std::complex<double>{-100, -101},
                      std::complex<double>{-1000, -1001}}};

  for (const auto& input : inputs) {
    const double expected = aocommon::Norm(input);

    const double result =
        aocommon::Avx256::MatrixComplexDouble2x2{input}.Norm();

    BOOST_CHECK_EQUAL(result, expected);
  }
}

BOOST_AUTO_TEST_CASE(trace) {
  static_assert(noexcept(aocommon::Avx256::MatrixComplexDouble2x2{
      static_cast<const std::complex<double>*>(nullptr)}
                             .Trace()));
  const std::vector<aocommon::MC2x2> inputs{
      aocommon::MC2x2{
          std::complex<double>{1.0, 2.0}, std::complex<double>{10, 11},
          std::complex<double>{100, 101}, std::complex<double>{1000, 1001}},
      aocommon::MC2x2{
          std::complex<double>{-1.0, 2.0}, std::complex<double>{10, 11},
          std::complex<double>{100, 101}, std::complex<double>{1000, 1001}},
      aocommon::MC2x2{
          std::complex<double>{-1.0, -2.0}, std::complex<double>{10, 11},
          std::complex<double>{100, 101}, std::complex<double>{1000, 1001}},
      aocommon::MC2x2{
          std::complex<double>{-1.0, -2.0}, std::complex<double>{-10, 11},
          std::complex<double>{100, 101}, std::complex<double>{1000, 1001}},
      aocommon::MC2x2{
          std::complex<double>{-1.0, -2.0}, std::complex<double>{-10, -11},
          std::complex<double>{100, 101}, std::complex<double>{1000, 1001}},
      aocommon::MC2x2{
          std::complex<double>{-1.0, -2.0}, std::complex<double>{-10, -11},
          std::complex<double>{-100, 101}, std::complex<double>{1000, 1001}},
      aocommon::MC2x2{
          std::complex<double>{-1.0, -2.0}, std::complex<double>{-10, -11},
          std::complex<double>{-100, -101}, std::complex<double>{1000, 1001}},
      aocommon::MC2x2{
          std::complex<double>{-1.0, -2.0}, std::complex<double>{-10, -11},
          std::complex<double>{-100, -101}, std::complex<double>{-1000, 1001}},
      aocommon::MC2x2{std::complex<double>{-1.0, -2.0},
                      std::complex<double>{-10, -11},
                      std::complex<double>{-100, -101},
                      std::complex<double>{-1000, -1001}}};

  for (const auto& input : inputs) {
    const std::complex<double> expected = aocommon::Trace(input);

    const std::complex<double> result =
        aocommon::Avx256::MatrixComplexDouble2x2{input}.Trace();

    BOOST_CHECK_EQUAL(result, expected);
  }
}

BOOST_AUTO_TEST_CASE(assign_to) {
  static_assert(noexcept(aocommon::Avx256::MatrixComplexDouble2x2{
      static_cast<const std::complex<double>*>(nullptr)}
                             .AssignTo(nullptr)));
  const aocommon::Avx256::MatrixComplexDouble2x2 input{
      std::complex<double>{-1.0, 1.0}, std::complex<double>{3.75, -3.75},
      std::complex<double>{99.0, -99.0}, std::complex<double>{1.5, -1.5}};

  std::vector<std::complex<double>> result(6);
  input.AssignTo(std::addressof(result[1]));

  BOOST_TEST(result[0] == (std::complex<double>{0.0, 0.0}));
  BOOST_TEST(result[1] == (std::complex<double>{-1.0, 1.0}));
  BOOST_TEST(result[2] == (std::complex<double>{3.75, -3.75}));
  BOOST_TEST(result[3] == (std::complex<double>{99.0, -99.0}));
  BOOST_TEST(result[4] == (std::complex<double>{1.5, -1.5}));
  BOOST_TEST(result[5] == (std::complex<double>{0.0, 0.0}));
}

BOOST_AUTO_TEST_CASE(Zero) {
  static_assert(noexcept(aocommon::Avx256::MatrixComplexDouble2x2::Zero()));

  BOOST_CHECK_EQUAL(
      aocommon::Avx256::MatrixComplexDouble2x2{aocommon::MC2x2::Zero()},
      aocommon::Avx256::MatrixComplexDouble2x2::Zero());
}

BOOST_AUTO_TEST_CASE(Unity) {
  static_assert(noexcept(aocommon::Avx256::MatrixComplexDouble2x2::Unity()));

  BOOST_CHECK_EQUAL(
      aocommon::Avx256::MatrixComplexDouble2x2{aocommon::MC2x2::Unity()},
      aocommon::Avx256::MatrixComplexDouble2x2::Unity());
}

BOOST_AUTO_TEST_CASE(NaN) {
  static_assert(noexcept(aocommon::Avx256::MatrixComplexDouble2x2::NaN()));

  const aocommon::Avx256::MatrixComplexDouble2x2 value =
      aocommon::Avx256::MatrixComplexDouble2x2::NaN();

  BOOST_TEST(std::isnan(value[0].real()));
  BOOST_TEST(std::isnan(value[0].imag()));
  BOOST_TEST(std::isnan(value[1].real()));
  BOOST_TEST(std::isnan(value[1].imag()));
  BOOST_TEST(std::isnan(value[2].real()));
  BOOST_TEST(std::isnan(value[2].imag()));
  BOOST_TEST(std::isnan(value[3].real()));
  BOOST_TEST(std::isnan(value[3].imag()));
}

BOOST_AUTO_TEST_CASE(operator_plus_equal) {
  aocommon::Avx256::MatrixComplexDouble2x2 r{
      {1.0, 2.0}, {10, 11}, {100, 101}, {1000, 1001}};

  const aocommon::Avx256::MatrixComplexDouble2x2 value{
      {4, 8}, {40, 44}, {400, 404}, {4000, 4004}};

  r += value;
  static_assert(noexcept(r += value));

  BOOST_TEST(r[0] == (std::complex<double>{5.0, 10.0}));
  BOOST_TEST(r[1] == (std::complex<double>{50.0, 55.0}));
  BOOST_TEST(r[2] == (std::complex<double>{500., 505.0}));
  BOOST_TEST(r[3] == (std::complex<double>{5000., 5005.0}));
}

BOOST_AUTO_TEST_CASE(operator_minus_equal) {
  aocommon::Avx256::MatrixComplexDouble2x2 r{
      {1.0, 2.0}, {10, 11}, {100, 101}, {1000, 1001}};

  const aocommon::Avx256::MatrixComplexDouble2x2 value{
      {4, 8}, {40, 44}, {400, 404}, {4000, 4004}};

  r -= value;
  static_assert(noexcept(r -= value));

  BOOST_TEST(r[0] == (std::complex<double>{-3, -6}));
  BOOST_TEST(r[1] == (std::complex<double>{-30, -33}));
  BOOST_TEST(r[2] == (std::complex<double>{-300, -303}));
  BOOST_TEST(r[3] == (std::complex<double>{-3000, -3003}));
}

BOOST_AUTO_TEST_CASE(operator_plus) {
  const aocommon::Avx256::MatrixComplexDouble2x2 lhs{
      {1.0, 2.0}, {10, 11}, {100, 101}, {1000, 1001}};

  const aocommon::Avx256::MatrixComplexDouble2x2 rhs{
      {4, 8}, {40, 44}, {400, 404}, {4000, 4004}};

  const aocommon::Avx256::MatrixComplexDouble2x2 r = lhs + rhs;
  static_assert(noexcept(lhs + rhs));

  BOOST_TEST(r[0] == (std::complex<double>{5.0, 10.0}));
  BOOST_TEST(r[1] == (std::complex<double>{50.0, 55.0}));
  BOOST_TEST(r[2] == (std::complex<double>{500., 505.0}));
  BOOST_TEST(r[3] == (std::complex<double>{5000., 5005.0}));
}

BOOST_AUTO_TEST_CASE(operator_minus) {
  const aocommon::Avx256::MatrixComplexDouble2x2 lhs{
      {1.0, 2.0}, {10, 11}, {100, 101}, {1000, 1001}};

  const aocommon::Avx256::MatrixComplexDouble2x2 rhs{
      {4, 8}, {40, 44}, {400, 404}, {4000, 4004}};

  const aocommon::Avx256::MatrixComplexDouble2x2 r = lhs - rhs;
  static_assert(noexcept(lhs - rhs));

  BOOST_TEST(r[0] == (std::complex<double>{-3, -6}));
  BOOST_TEST(r[1] == (std::complex<double>{-30, -33}));
  BOOST_TEST(r[2] == (std::complex<double>{-300, -303}));
  BOOST_TEST(r[3] == (std::complex<double>{-3000, -3003}));
}

BOOST_AUTO_TEST_CASE(multiply) {
  {
    const aocommon::Avx256::MatrixComplexDouble2x2 lhs{
        {1.0, 2.0}, {10, 11}, {100, 101}, {1000, 1001}};

    const aocommon::Avx256::MatrixComplexDouble2x2 rhs{
        {4, 8}, {40, 44}, {400, 404}, {4000, 4004}};

    static_assert(noexcept(lhs * rhs));
    const aocommon::Avx256::MatrixComplexDouble2x2 r = lhs * rhs;

    BOOST_CHECK_CLOSE(r[0].real(), -456, 1e-6);
    BOOST_CHECK_CLOSE(r[0].imag(), 8456, 1e-6);
    BOOST_CHECK_CLOSE(r[1].real(), -4092, 1e-6);
    BOOST_CHECK_CLOSE(r[1].imag(), 84164, 1e-6);
    BOOST_CHECK_CLOSE(r[2].real(), -4812, 1e-6);
    BOOST_CHECK_CLOSE(r[2].imag(), 805604, 1e-6);
    BOOST_CHECK_CLOSE(r[3].real(), -8448, 1e-6);
    BOOST_CHECK_CLOSE(r[3].imag(), 8016440, 1e-6);
  }

  // Emulate Matrix * Diagonal Matrix
  {
    const aocommon::Avx256::MatrixComplexDouble2x2 lhs{
        {1.0, 2.0}, {10, 11}, {100, 101}, {1000, 1001}};

    const aocommon::Avx256::MatrixComplexDouble2x2 rhs{
        {4, 8}, {0, 0}, {0, 0}, {4000, 4004}};

    const aocommon::Avx256::MatrixComplexDouble2x2 r = lhs * rhs;

    BOOST_CHECK_CLOSE(r[0].real(), -12, 1e-6);
    BOOST_CHECK_CLOSE(r[0].imag(), 16, 1e-6);
    BOOST_CHECK_CLOSE(r[1].real(), -4044, 1e-6);
    BOOST_CHECK_CLOSE(r[1].imag(), 84040, 1e-6);
    BOOST_CHECK_CLOSE(r[2].real(), -408, 1e-6);
    BOOST_CHECK_CLOSE(r[2].imag(), 1204, 1e-6);
    BOOST_CHECK_CLOSE(r[3].real(), -8004, 1e-6);
    BOOST_CHECK_CLOSE(r[3].imag(), 8008000, 1e-6);
  }

  // Emulate Diagonal Matrix * Matrix
  {
    const aocommon::Avx256::MatrixComplexDouble2x2 lhs{
        {1.0, 2.0}, {0, 0}, {0, 0}, {1000, 1001}};

    const aocommon::Avx256::MatrixComplexDouble2x2 rhs{
        {4, 8}, {40, 44}, {400, 404}, {4000, 4004}};

    static_assert(noexcept(lhs * rhs));
    const aocommon::Avx256::MatrixComplexDouble2x2 r = lhs * rhs;

    BOOST_CHECK_CLOSE(r[0].real(), -12, 1e-6);
    BOOST_CHECK_CLOSE(r[0].imag(), 16, 1e-6);
    BOOST_CHECK_CLOSE(r[1].real(), -48, 1e-6);
    BOOST_CHECK_CLOSE(r[1].imag(), 124, 1e-6);
    BOOST_CHECK_CLOSE(r[2].real(), -4404, 1e-6);
    BOOST_CHECK_CLOSE(r[2].imag(), 804400, 1e-6);
    BOOST_CHECK_CLOSE(r[3].real(), -8004, 1e-6);
    BOOST_CHECK_CLOSE(r[3].imag(), 8008000, 1e-6);
  }
}

BOOST_AUTO_TEST_CASE(operator_multiply_equal) {
  {
    aocommon::Avx256::MatrixComplexDouble2x2 value{
        {1.0, 2.0}, {10, 11}, {100, 101}, {1000, 1001}};

    const aocommon::Avx256::MatrixComplexDouble2x2 rhs{
        {4, 8}, {40, 44}, {400, 404}, {4000, 4004}};

    static_assert(noexcept(value *= rhs));
    value *= rhs;

    BOOST_CHECK_CLOSE(value[0].real(), -456, 1e-6);
    BOOST_CHECK_CLOSE(value[0].imag(), 8456, 1e-6);
    BOOST_CHECK_CLOSE(value[1].real(), -4092, 1e-6);
    BOOST_CHECK_CLOSE(value[1].imag(), 84164, 1e-6);
    BOOST_CHECK_CLOSE(value[2].real(), -4812, 1e-6);
    BOOST_CHECK_CLOSE(value[2].imag(), 805604, 1e-6);
    BOOST_CHECK_CLOSE(value[3].real(), -8448, 1e-6);
    BOOST_CHECK_CLOSE(value[3].imag(), 8016440, 1e-6);
  }

  // Emulate Matrix * Diagonal Matrix
  {
    aocommon::Avx256::MatrixComplexDouble2x2 value{
        {1.0, 2.0}, {10, 11}, {100, 101}, {1000, 1001}};

    const aocommon::Avx256::MatrixComplexDouble2x2 rhs{
        {4, 8}, {0, 0}, {0, 0}, {4000, 4004}};

    value *= rhs;

    BOOST_CHECK_CLOSE(value[0].real(), -12, 1e-6);
    BOOST_CHECK_CLOSE(value[0].imag(), 16, 1e-6);
    BOOST_CHECK_CLOSE(value[1].real(), -4044, 1e-6);
    BOOST_CHECK_CLOSE(value[1].imag(), 84040, 1e-6);
    BOOST_CHECK_CLOSE(value[2].real(), -408, 1e-6);
    BOOST_CHECK_CLOSE(value[2].imag(), 1204, 1e-6);
    BOOST_CHECK_CLOSE(value[3].real(), -8004, 1e-6);
    BOOST_CHECK_CLOSE(value[3].imag(), 8008000, 1e-6);
  }

  // Emulate Diagonal Matrix * Matrix
  {
    aocommon::Avx256::MatrixComplexDouble2x2 value{
        {1.0, 2.0}, {0, 0}, {0, 0}, {1000, 1001}};

    const aocommon::Avx256::MatrixComplexDouble2x2 rhs{
        {4, 8}, {40, 44}, {400, 404}, {4000, 4004}};

    value *= rhs;

    BOOST_CHECK_CLOSE(value[0].real(), -12, 1e-6);
    BOOST_CHECK_CLOSE(value[0].imag(), 16, 1e-6);
    BOOST_CHECK_CLOSE(value[1].real(), -48, 1e-6);
    BOOST_CHECK_CLOSE(value[1].imag(), 124, 1e-6);
    BOOST_CHECK_CLOSE(value[2].real(), -4404, 1e-6);
    BOOST_CHECK_CLOSE(value[2].imag(), 804400, 1e-6);
    BOOST_CHECK_CLOSE(value[3].real(), -8004, 1e-6);
    BOOST_CHECK_CLOSE(value[3].imag(), 8008000, 1e-6);
  }
}

BOOST_AUTO_TEST_CASE(multiply_matrix_and_value) {
  const aocommon::Avx256::MatrixComplexDouble2x2 lhs{
      {4, 8}, {40, 44}, {400, 404}, {4000, 4004}};

  const std::complex<double> rhs{1.0, 2.0};

  static_assert(noexcept(lhs * rhs));
  const aocommon::Avx256::MatrixComplexDouble2x2 r = lhs * rhs;

  BOOST_CHECK_CLOSE(r[0].real(), -12, 1e-6);
  BOOST_CHECK_CLOSE(r[0].imag(), 16, 1e-6);
  BOOST_CHECK_CLOSE(r[1].real(), -48, 1e-6);
  BOOST_CHECK_CLOSE(r[1].imag(), 124, 1e-6);
  BOOST_CHECK_CLOSE(r[2].real(), -408, 1e-6);
  BOOST_CHECK_CLOSE(r[2].imag(), 1204, 1e-6);
  BOOST_CHECK_CLOSE(r[3].real(), -4008, 1e-6);
  BOOST_CHECK_CLOSE(r[3].imag(), 12004, 1e-6);
}

BOOST_AUTO_TEST_CASE(multiply_value_and_matrix) {
  const std::complex<double> lhs{1.0, 2.0};

  const aocommon::Avx256::MatrixComplexDouble2x2 rhs{
      {4, 8}, {40, 44}, {400, 404}, {4000, 4004}};

  static_assert(noexcept(lhs * rhs));
  const aocommon::Avx256::MatrixComplexDouble2x2 r = lhs * rhs;

  BOOST_CHECK_CLOSE(r[0].real(), -12, 1e-6);
  BOOST_CHECK_CLOSE(r[0].imag(), 16, 1e-6);
  BOOST_CHECK_CLOSE(r[1].real(), -48, 1e-6);
  BOOST_CHECK_CLOSE(r[1].imag(), 124, 1e-6);
  BOOST_CHECK_CLOSE(r[2].real(), -408, 1e-6);
  BOOST_CHECK_CLOSE(r[2].imag(), 1204, 1e-6);
  BOOST_CHECK_CLOSE(r[3].real(), -4008, 1e-6);
  BOOST_CHECK_CLOSE(r[3].imag(), 12004, 1e-6);
}

BOOST_AUTO_TEST_CASE(multiply_diagonal_matrix_and_matrix) {
  const aocommon::Avx256::DiagonalMatrixComplexDouble2x2 lhs{{1.0, 2.0},
                                                             {1000, 1001}};

  const aocommon::Avx256::MatrixComplexDouble2x2 rhs{
      {4, 8}, {40, 44}, {400, 404}, {4000, 4004}};

  static_assert(noexcept(lhs * rhs));
  const aocommon::Avx256::MatrixComplexDouble2x2 r = lhs * rhs;

  BOOST_CHECK_CLOSE(r[0].real(), -12, 1e-6);
  BOOST_CHECK_CLOSE(r[0].imag(), 16, 1e-6);
  BOOST_CHECK_CLOSE(r[1].real(), -48, 1e-6);
  BOOST_CHECK_CLOSE(r[1].imag(), 124, 1e-6);
  BOOST_CHECK_CLOSE(r[2].real(), -4404, 1e-6);
  BOOST_CHECK_CLOSE(r[2].imag(), 804400, 1e-6);
  BOOST_CHECK_CLOSE(r[3].real(), -8004, 1e-6);
  BOOST_CHECK_CLOSE(r[3].imag(), 8008000, 1e-6);
}

BOOST_AUTO_TEST_CASE(multiply_matrix_and_diagonal_matrix) {
  const aocommon::Avx256::MatrixComplexDouble2x2 lhs{
      {1.0, 2.0}, {10, 11}, {100, 101}, {1000, 1001}};

  const aocommon::Avx256::DiagonalMatrixComplexDouble2x2 rhs{{4, 8},
                                                             {4000, 4004}};

  static_assert(noexcept(lhs * rhs));
  const aocommon::Avx256::MatrixComplexDouble2x2 r = lhs * rhs;

  BOOST_CHECK_CLOSE(r[0].real(), -12, 1e-6);
  BOOST_CHECK_CLOSE(r[0].imag(), 16, 1e-6);
  BOOST_CHECK_CLOSE(r[1].real(), -4044, 1e-6);
  BOOST_CHECK_CLOSE(r[1].imag(), 84040, 1e-6);
  BOOST_CHECK_CLOSE(r[2].real(), -408, 1e-6);
  BOOST_CHECK_CLOSE(r[2].imag(), 1204, 1e-6);
  BOOST_CHECK_CLOSE(r[3].real(), -8004, 1e-6);
  BOOST_CHECK_CLOSE(r[3].imag(), 8008000, 1e-6);
}

BOOST_AUTO_TEST_CASE(equal) {
  static_assert(
      noexcept(aocommon::Avx256::MatrixComplexDouble2x2{
                   static_cast<const std::complex<double>*>(nullptr)} ==
               aocommon::Avx256::MatrixComplexDouble2x2{
                   static_cast<const std::complex<double>*>(nullptr)}));

  BOOST_TEST((aocommon::Avx256::MatrixComplexDouble2x2{
                  {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}} ==
              aocommon::Avx256::MatrixComplexDouble2x2{
                  {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}}));
  BOOST_TEST(!(aocommon::Avx256::MatrixComplexDouble2x2{
                   {42.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}} ==
               aocommon::Avx256::MatrixComplexDouble2x2{
                   {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}}));
  BOOST_TEST(!(aocommon::Avx256::MatrixComplexDouble2x2{
                   {0.0, 42.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}} ==
               aocommon::Avx256::MatrixComplexDouble2x2{
                   {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}}));
  BOOST_TEST(!(aocommon::Avx256::MatrixComplexDouble2x2{
                   {0.0, 0.0}, {42.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}} ==
               aocommon::Avx256::MatrixComplexDouble2x2{
                   {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}}));
  BOOST_TEST(!(aocommon::Avx256::MatrixComplexDouble2x2{
                   {0.0, 0.0}, {0.0, 42.0}, {0.0, 0.0}, {0.0, 0.0}} ==
               aocommon::Avx256::MatrixComplexDouble2x2{
                   {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}}));
  BOOST_TEST(!(aocommon::Avx256::MatrixComplexDouble2x2{
                   {0.0, 0.0}, {0.0, 0.0}, {42.0, 0.0}, {0.0, 0.0}} ==
               aocommon::Avx256::MatrixComplexDouble2x2{
                   {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}}));
  BOOST_TEST(!(aocommon::Avx256::MatrixComplexDouble2x2{
                   {0.0, 0.0}, {0.0, 0.0}, {0.0, 42.0}, {0.0, 0.0}} ==
               aocommon::Avx256::MatrixComplexDouble2x2{
                   {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}}));
  BOOST_TEST(!(aocommon::Avx256::MatrixComplexDouble2x2{
                   {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {42.0, 0.0}} ==
               aocommon::Avx256::MatrixComplexDouble2x2{
                   {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}}));
  BOOST_TEST(!(aocommon::Avx256::MatrixComplexDouble2x2{
                   {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 42.0}} ==
               aocommon::Avx256::MatrixComplexDouble2x2{
                   {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}}));
}

BOOST_AUTO_TEST_CASE(output) {
  const aocommon::Avx256::MatrixComplexDouble2x2 input{
      std::complex<double>{-1.0, 1.0}, std::complex<double>{3.75, -3.75},
      std::complex<double>{99.0, -99.0}, std::complex<double>{1.5, -1.5}};

  std::stringstream result;
  result << input;

  BOOST_CHECK_EQUAL(result.str(),
                    "[{(-1,1), (3.75,-3.75)}, {(99,-99), (1.5,-1.5)}]");
}

BOOST_AUTO_TEST_CASE(non_member_herm_transpose) {
  static_assert(noexcept(
      aocommon::Avx256::HermTranspose(aocommon::Avx256::MatrixComplexDouble2x2{
          static_cast<const std::complex<double>*>(nullptr)})));
  const std::vector<aocommon::Avx256::MatrixComplexDouble2x2> inputs{
      aocommon::Avx256::MatrixComplexDouble2x2{
          std::complex<double>{1.0, 2.0}, std::complex<double>{10, 11},
          std::complex<double>{100, 101}, std::complex<double>{1000, 1001}},
      aocommon::Avx256::MatrixComplexDouble2x2{
          std::complex<double>{-1.0, 2.0}, std::complex<double>{10, 11},
          std::complex<double>{100, 101}, std::complex<double>{1000, 1001}},
      aocommon::Avx256::MatrixComplexDouble2x2{
          std::complex<double>{-1.0, -2.0}, std::complex<double>{10, 11},
          std::complex<double>{100, 101}, std::complex<double>{1000, 1001}},
      aocommon::Avx256::MatrixComplexDouble2x2{
          std::complex<double>{-1.0, -2.0}, std::complex<double>{-10, 11},
          std::complex<double>{100, 101}, std::complex<double>{1000, 1001}},
      aocommon::Avx256::MatrixComplexDouble2x2{
          std::complex<double>{-1.0, -2.0}, std::complex<double>{-10, -11},
          std::complex<double>{100, 101}, std::complex<double>{1000, 1001}},
      aocommon::Avx256::MatrixComplexDouble2x2{
          std::complex<double>{-1.0, -2.0}, std::complex<double>{-10, -11},
          std::complex<double>{-100, 101}, std::complex<double>{1000, 1001}},
      aocommon::Avx256::MatrixComplexDouble2x2{
          std::complex<double>{-1.0, -2.0}, std::complex<double>{-10, -11},
          std::complex<double>{-100, -101}, std::complex<double>{1000, 1001}},
      aocommon::Avx256::MatrixComplexDouble2x2{
          std::complex<double>{-1.0, -2.0}, std::complex<double>{-10, -11},
          std::complex<double>{-100, -101}, std::complex<double>{-1000, 1001}},
      aocommon::Avx256::MatrixComplexDouble2x2{
          std::complex<double>{-1.0, -2.0}, std::complex<double>{-10, -11},
          std::complex<double>{-100, -101},
          std::complex<double>{-1000, -1001}}};

  for (const auto& input : inputs)
    BOOST_CHECK_EQUAL(input.HermitianTranspose(),
                      aocommon::Avx256::HermTranspose(input));
}

BOOST_AUTO_TEST_CASE(non_member_norm) {
  static_assert(
      noexcept(aocommon::Avx256::Norm(aocommon::Avx256::MatrixComplexDouble2x2{
          static_cast<const std::complex<double>*>(nullptr)})));
  const std::vector<aocommon::Avx256::MatrixComplexDouble2x2> inputs{
      aocommon::Avx256::MatrixComplexDouble2x2{
          std::complex<double>{1.0, 2.0}, std::complex<double>{10, 11},
          std::complex<double>{100, 101}, std::complex<double>{1000, 1001}},
      aocommon::Avx256::MatrixComplexDouble2x2{
          std::complex<double>{-1.0, 2.0}, std::complex<double>{10, 11},
          std::complex<double>{100, 101}, std::complex<double>{1000, 1001}},
      aocommon::Avx256::MatrixComplexDouble2x2{
          std::complex<double>{-1.0, -2.0}, std::complex<double>{10, 11},
          std::complex<double>{100, 101}, std::complex<double>{1000, 1001}},
      aocommon::Avx256::MatrixComplexDouble2x2{
          std::complex<double>{-1.0, -2.0}, std::complex<double>{-10, 11},
          std::complex<double>{100, 101}, std::complex<double>{1000, 1001}},
      aocommon::Avx256::MatrixComplexDouble2x2{
          std::complex<double>{-1.0, -2.0}, std::complex<double>{-10, -11},
          std::complex<double>{100, 101}, std::complex<double>{1000, 1001}},
      aocommon::Avx256::MatrixComplexDouble2x2{
          std::complex<double>{-1.0, -2.0}, std::complex<double>{-10, -11},
          std::complex<double>{-100, 101}, std::complex<double>{1000, 1001}},
      aocommon::Avx256::MatrixComplexDouble2x2{
          std::complex<double>{-1.0, -2.0}, std::complex<double>{-10, -11},
          std::complex<double>{-100, -101}, std::complex<double>{1000, 1001}},
      aocommon::Avx256::MatrixComplexDouble2x2{
          std::complex<double>{-1.0, -2.0}, std::complex<double>{-10, -11},
          std::complex<double>{-100, -101}, std::complex<double>{-1000, 1001}},
      aocommon::Avx256::MatrixComplexDouble2x2{
          std::complex<double>{-1.0, -2.0}, std::complex<double>{-10, -11},
          std::complex<double>{-100, -101},
          std::complex<double>{-1000, -1001}}};

  for (const auto& input : inputs) BOOST_CHECK_EQUAL(input.Norm(), 2022428.0);
}

BOOST_AUTO_TEST_CASE(non_member_trace) {
  static_assert(
      noexcept(aocommon::Avx256::Trace(aocommon::Avx256::MatrixComplexDouble2x2{
          static_cast<const std::complex<double>*>(nullptr)})));
  const std::vector<aocommon::Avx256::MatrixComplexDouble2x2> inputs{
      aocommon::Avx256::MatrixComplexDouble2x2{
          std::complex<double>{1.0, 2.0}, std::complex<double>{10, 11},
          std::complex<double>{100, 101}, std::complex<double>{1000, 1001}},
      aocommon::Avx256::MatrixComplexDouble2x2{
          std::complex<double>{-1.0, 2.0}, std::complex<double>{10, 11},
          std::complex<double>{100, 101}, std::complex<double>{1000, 1001}},
      aocommon::Avx256::MatrixComplexDouble2x2{
          std::complex<double>{-1.0, -2.0}, std::complex<double>{10, 11},
          std::complex<double>{100, 101}, std::complex<double>{1000, 1001}},
      aocommon::Avx256::MatrixComplexDouble2x2{
          std::complex<double>{-1.0, -2.0}, std::complex<double>{-10, 11},
          std::complex<double>{100, 101}, std::complex<double>{1000, 1001}},
      aocommon::Avx256::MatrixComplexDouble2x2{
          std::complex<double>{-1.0, -2.0}, std::complex<double>{-10, -11},
          std::complex<double>{100, 101}, std::complex<double>{1000, 1001}},
      aocommon::Avx256::MatrixComplexDouble2x2{
          std::complex<double>{-1.0, -2.0}, std::complex<double>{-10, -11},
          std::complex<double>{-100, 101}, std::complex<double>{1000, 1001}},
      aocommon::Avx256::MatrixComplexDouble2x2{
          std::complex<double>{-1.0, -2.0}, std::complex<double>{-10, -11},
          std::complex<double>{-100, -101}, std::complex<double>{1000, 1001}},
      aocommon::Avx256::MatrixComplexDouble2x2{
          std::complex<double>{-1.0, -2.0}, std::complex<double>{-10, -11},
          std::complex<double>{-100, -101}, std::complex<double>{-1000, 1001}},
      aocommon::Avx256::MatrixComplexDouble2x2{
          std::complex<double>{-1.0, -2.0}, std::complex<double>{-10, -11},
          std::complex<double>{-100, -101},
          std::complex<double>{-1000, -1001}}};

  for (const auto& input : inputs)
    BOOST_CHECK_EQUAL(Trace(input), input.Trace());
}

BOOST_AUTO_TEST_SUITE_END()

#endif  // defined(__AVX2__)
