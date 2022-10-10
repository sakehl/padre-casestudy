// Copyright (C) 2022 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include "../../ApplyBeam.h"

#include <boost/test/unit_test.hpp>

#include "../../../common/ParameterSet.h"

#include "mock/MockInput.h"

using dp3::steps::ApplyBeam;
using dp3::steps::Step;

BOOST_AUTO_TEST_SUITE(apply_beam)

BOOST_AUTO_TEST_CASE(fields_defaults) {
  dp3::common::ParameterSet parset;
  dp3::steps::MockInput input;
  const ApplyBeam apply_beam(&input, parset, "");
  BOOST_TEST(apply_beam.getRequiredFields() == Step::kDataField);
  BOOST_TEST(apply_beam.getProvidedFields() == Step::kDataField);
}

BOOST_AUTO_TEST_CASE(fields_updateweights) {
  dp3::common::ParameterSet parset;
  parset.add("updateweights", "true");
  dp3::steps::MockInput input;
  const ApplyBeam updates_weights(&input, parset, "");

  const dp3::common::Fields kExpectedFields =
      Step::kDataField | Step::kWeightsField;
  BOOST_TEST(updates_weights.getRequiredFields() == kExpectedFields);
  BOOST_TEST(updates_weights.getProvidedFields() == kExpectedFields);
}

BOOST_AUTO_TEST_SUITE_END()