// Copyright (C) 2022 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include "../../MSUpdater.h"

#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>

#include "../../MSReader.h"
#include "../../NullStep.h"
#include "../../../common/ParameterSet.h"

#include "../../../common/test/unit/fixtures/fDirectory.h"
#include "mock/MockInput.h"
#include "mock/ThrowStep.h"
#include "tStepCommon.h"

using dp3::steps::MSReader;
using dp3::steps::MSUpdater;
using dp3::steps::NullStep;
using dp3::steps::Step;

namespace {

const std::string kInputMs = "../tNDPPP-generic.MS";
const std::string kCopyMs = "tNDPPP-generic-copy.MS";
const std::complex<float> kDataAdjustment{42.0, 43.0};
const float kWeightAdjustment{42.0};

const std::string kInputDataColumnName = "input_data";
const std::string kInputFlagColumnName = "input_flags";
const std::string kInputWeightColumnName = "input_weights";

/**
 * Test input class that returns fixed column names.
 */
class TestInput : public dp3::steps::MockInput {
 public:
  const std::string& dataColumnName() const override {
    return kInputDataColumnName;
  }

  const std::string& flagColumnName() const override {
    return kInputFlagColumnName;
  }

  const std::string& weightColumnName() const override {
    return kInputWeightColumnName;
  }
};

void TestFields(const std::string& data_column_name,
                const std::string& flag_column_name,
                const std::string& weight_column_name,
                const dp3::common::Fields expected_required_fields) {
  TestInput input;
  dp3::common::ParameterSet parset;
  parset.add("datacolumn", data_column_name);
  parset.add("flagcolumn", flag_column_name);
  parset.add("weightcolumn", weight_column_name);
  const MSUpdater updater(&input, "test_msupdater.ms", parset, "");
  BOOST_TEST(updater.getRequiredFields() == expected_required_fields);
  BOOST_TEST(updater.getProvidedFields() == dp3::common::Fields());
}

/**
 * Fixture for copying an input MS into a temporary directory.
 */
class FixtureCopyInput : public dp3::common::test::FixtureDirectory {
 public:
  FixtureCopyInput() : FixtureDirectory() {
    casacore::Table(kInputMs).deepCopy(kCopyMs, casacore::Table::New);
  }
};

/**
 * Sets all flags to false, and sets each 42nd value to true.
 */
void SetTestFlags(casacore::Array<bool>& flags) {
  for (std::size_t i = 0; i < flags.size(); ++i) {
    flags.data()[i] = ((i % 42) == 0);
  }
}

/**
 * Test class that changes values in the input buffer:
 * - Adds kDataAdjustment to all data values.
 * - Adds kWeightAdjustment to all weight values.
 * - Apply SetTestFlags to the flags.
 */
class TestAdjust : public dp3::steps::test::ThrowStep {
 public:
  TestAdjust(MSReader& reader) : reader_(reader) {}

  void updateInfo(const dp3::base::DPInfo& info) override {
    Step::updateInfo(info);
  }

  bool process(const dp3::base::DPBuffer& buffer) override {
    dp3::base::DPBuffer adjusted = buffer;
    dp3::common::NSTimer timer;
    reader_.fetchWeights(buffer, adjusted, timer);

    adjusted.getData() += kDataAdjustment;
    SetTestFlags(adjusted.getFlags());
    adjusted.getWeights() += kWeightAdjustment;

    getNextStep()->process(adjusted);
    return true;
  }

  void finish() override {}  // do nothing

 private:
  MSReader& reader_;
};

}  // namespace

BOOST_AUTO_TEST_SUITE(msupdater)

BOOST_AUTO_TEST_CASE(fields) {
  TestFields(kInputDataColumnName, kInputFlagColumnName, kInputWeightColumnName,
             dp3::common::Fields());
  TestFields("custom_data", kInputFlagColumnName, kInputWeightColumnName,
             Step::kDataField);
  TestFields(kInputDataColumnName, "custom_flag", kInputWeightColumnName,
             Step::kFlagsField);
  TestFields(kInputDataColumnName, kInputFlagColumnName, "custom_weight",
             Step::kWeightsField);
  TestFields("all_columns", "have", "different_names",
             Step::kDataField | Step::kFlagsField | Step::kWeightsField);
}

BOOST_DATA_TEST_CASE_F(
    FixtureCopyInput, update_fields,
    boost::unit_test::data::make({dp3::common::Fields(), Step::kDataField,
                                  Step::kFlagsField, Step::kWeightsField,
                                  Step::kDataField | Step::kFlagsField |
                                      Step::kWeightsField}),
    fields_to_write) {
  const casacore::MS original_ms(kInputMs);
  const casacore::MS updated_ms(kCopyMs);
  const dp3::common::ParameterSet parset;

  // Read, adjust and write data using the MSUpdater.
  {
    auto reader = std::make_shared<MSReader>(updated_ms, parset, "");
    auto adjust = std::make_shared<TestAdjust>(*reader);
    auto updater =
        std::make_shared<MSUpdater>(reader.get(), kCopyMs, parset, "");
    reader->setReadData();
    reader->setFieldsToRead(Step::kDataField | Step::kFlagsField |
                            Step::kWeightsField);
    updater->SetFieldsToWrite(fields_to_write);
    dp3::steps::test::Execute(
        {reader, adjust, updater, std::make_shared<NullStep>()});
  }

  // Check that the fields in fields_to_write are adjusted and that the
  // other fields did not change.
  const casacore::String data_column_name =
      casacore::MS::columnName(casacore::MS::DATA);
  const casacore::String flag_column_name =
      casacore::MS::columnName(casacore::MS::FLAG);
  const casacore::String weight_column_name =
      casacore::MS::columnName(casacore::MS::WEIGHT_SPECTRUM);

  // Read one time slot at a time using these iterators.
  casacore::TableIterator original_time_iterator(
      original_ms, casacore::Block<casacore::String>(1, "TIME"));
  casacore::TableIterator updated_time_iterator(
      updated_ms, casacore::Block<casacore::String>(1, "TIME"));

  while (!original_time_iterator.pastEnd()) {
    BOOST_TEST_REQUIRE(!updated_time_iterator.pastEnd());
    casacore::Table original_table = original_time_iterator.table();
    casacore::Table updated_table = updated_time_iterator.table();

    casacore::Array<casacore::Complex> original_data;
    casacore::Array<casacore::Complex> updated_data;
    casacore::Array<bool> original_flags;
    casacore::Array<bool> updated_flags;
    casacore::Array<float> original_weights;
    casacore::Array<float> updated_weights;
    casacore::ArrayColumn<casacore::Complex>(original_table, data_column_name)
        .getColumn(original_data);
    casacore::ArrayColumn<casacore::Complex>(updated_table, data_column_name)
        .getColumn(updated_data);
    casacore::ArrayColumn<bool>(original_table, flag_column_name)
        .getColumn(original_flags);
    casacore::ArrayColumn<bool>(updated_table, flag_column_name)
        .getColumn(updated_flags);
    casacore::ArrayColumn<float>(original_table, weight_column_name)
        .getColumn(original_weights);
    casacore::ArrayColumn<float>(updated_table, weight_column_name)
        .getColumn(updated_weights);

    BOOST_TEST(original_data.shape() == updated_data.shape());
    BOOST_TEST(original_flags.shape() == updated_flags.shape());
    BOOST_TEST(original_weights.shape() == updated_weights.shape());

    if (fields_to_write.Data()) {
      BOOST_TEST(casacore::allNear(original_data + kDataAdjustment,
                                   updated_data, 1.0e-6));
    } else {
      BOOST_TEST(casacore::allEQ(original_data, updated_data));
    }

    if (fields_to_write.Flags()) SetTestFlags(original_flags);
    BOOST_TEST(casacore::allEQ(original_flags, updated_flags));

    if (fields_to_write.Weights()) {
      BOOST_TEST(casacore::allNear(original_weights + kWeightAdjustment,
                                   updated_weights, 1.0e-6));
    } else {
      BOOST_TEST(casacore::allEQ(original_weights, updated_weights));
    }

    original_time_iterator.next();
    updated_time_iterator.next();
  }
  BOOST_TEST(updated_time_iterator.pastEnd());
}

BOOST_AUTO_TEST_SUITE_END()
