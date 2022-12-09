// Copyright (C) 2021 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include "../../OnePredict.h"

#include <regex>

#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>

#include <dp3/base/DP3.h>

#include "../../../common/ParameterSet.h"
#include "../../ApplyCal.h"
#include "../../NullStep.h"

#include "tPredict.h"
#include "H5ParmFixture.h"
#include "mock/MockInput.h"

using dp3::steps::OnePredict;
using dp3::steps::Step;

namespace {

// Constants copy pasted from steps/test/unit/tIDGPredict.cc
constexpr unsigned int kNCorr = 4;
constexpr unsigned int kNChan = 5;
const std::vector<std::size_t> kChannelCounts(kNChan, 1);
constexpr double kStartTime = 0.0;
constexpr double kInterval = 1.0;
constexpr std::size_t kNBaselines = 3;

class OnePredictFixture {
 public:
  OnePredictFixture() : predict_() {
    dp3::common::ParameterSet parset;
    parset.add("fixture.sourcedb", dp3::steps::test::kPredictSourceDB);
    predict_ = std::make_shared<OnePredict>(parset, "fixture.",
                                            std::vector<std::string>());
    predict_->setNextStep(std::make_shared<dp3::steps::NullStep>());
    SetInfo();
  }

  void SetInfo() {
    const unsigned int kNChannels = 1;
    dp3::base::DPInfo info(1, kNChannels);
    info.setTimes(0.5, 9.5, 1.0);

    const std::vector<int> kAnt1{0, 0, 1};
    const std::vector<int> kAnt2{1, 2, 2};
    const std::vector<std::string> kAntNames{"ant0", "ant1", "ant2"};
    const std::vector<double> kAntDiam(3, 1.0);
    const std::vector<casacore::MPosition> kAntPos(3);
    info.setAntennas(kAntNames, kAntDiam, kAntPos, kAnt1, kAnt2);

    std::vector<double> chan_freqs(kNChannels, 10.0e6);
    std::vector<double> chan_widths(kNChannels, 3.0e6);

    info.setChannels(std::move(chan_freqs), std::move(chan_widths));
    predict_->setInfo(info);
  }

 protected:
  std::shared_ptr<dp3::steps::OnePredict> predict_;
};
}  // namespace

BOOST_AUTO_TEST_SUITE(onepredict)

BOOST_FIXTURE_TEST_CASE(constructor, OnePredictFixture) {
  // Nothing to do: The fixture calls the constructor.
}

BOOST_FIXTURE_TEST_CASE(getfirstdirection, OnePredictFixture) {
  const dp3::base::Direction first_direction = predict_->GetFirstDirection();

  BOOST_CHECK_CLOSE(first_direction.ra,
                    dp3::steps::test::kExpectedFirstDirection.ra, 1.0e-3);
  BOOST_CHECK_CLOSE(first_direction.dec,
                    dp3::steps::test::kExpectedFirstDirection.dec, 1.0e-3);
}

BOOST_FIXTURE_TEST_CASE(fields_defaults, OnePredictFixture) {
  BOOST_TEST(predict_->getRequiredFields() == Step::kUvwField);
  BOOST_TEST(predict_->getProvidedFields() == Step::kDataField);
}

BOOST_DATA_TEST_CASE(fields_add_subtract,
                     boost::unit_test::data::make({"add", "subtract"}),
                     operation) {
  dp3::common::ParameterSet parset;
  parset.add("sourcedb", dp3::steps::test::kPredictSourceDB);
  parset.add("operation", operation);
  const OnePredict predict(parset, "", {});
  BOOST_TEST(predict.getRequiredFields() ==
             (Step::kDataField | Step::kUvwField));
  BOOST_TEST(predict.getProvidedFields() == Step::kDataField);
}

BOOST_FIXTURE_TEST_CASE(fields_applycal, dp3::steps::test::H5ParmFixture) {
  dp3::common::ParameterSet parset;
  parset.add("sourcedb", dp3::steps::test::kPredictSourceDB);
  parset.add("applycal.parmdb", kParmDb);
  parset.add("applycal.correction", kSoltabName);
  const OnePredict predict(parset, "", {});

  // OnePredict uses ApplyCal which has a OneApplyCal sub-step as next step.
  const dp3::steps::ApplyCal apply_cal(parset, "applycal.", true);

  const dp3::common::Fields apply_cal_required =
      dp3::base::GetChainRequiredFields(
          std::make_shared<dp3::steps::ApplyCal>(apply_cal));
  // TODO(AST-1033) Determine ApplyCal provided fields using generic DP3
  // functions.
  const dp3::common::Fields apply_cal_provided =
      apply_cal.getNextStep()->getProvidedFields();
  BOOST_TEST(predict.getRequiredFields() ==
             (apply_cal_required | Step::kUvwField));
  BOOST_TEST(predict.getProvidedFields() ==
             (apply_cal_provided | Step::kDataField));
}

BOOST_DATA_TEST_CASE_F(dp3::steps::test::H5ParmFixture,
                       fields_applycal_add_subtract,
                       boost::unit_test::data::make({"add", "subtract"}),
                       operation) {
  dp3::common::ParameterSet parset;
  parset.add("sourcedb", dp3::steps::test::kPredictSourceDB);
  parset.add("applycal.parmdb", kParmDb);
  parset.add("applycal.correction", kSoltabName);
  parset.add("operation", operation);
  const OnePredict predict(parset, "", {});

  // When operation is "add" or "subtract", OnePredict only combines the
  // required fields of its ApplyCal sub-step.
  const dp3::steps::ApplyCal apply_cal(parset, "applycal.", true);

  const dp3::common::Fields apply_cal_required =
      dp3::base::GetChainRequiredFields(
          std::make_shared<dp3::steps::ApplyCal>(apply_cal));

  BOOST_TEST(predict.getRequiredFields() ==
             (apply_cal_required | Step::kUvwField));
  BOOST_TEST(predict.getProvidedFields() == Step::kDataField);
}

/**
 * Create a buffer with artifical data values.
 * @param time Start time for the buffer.
 * @param interval Interval duration for the buffer.
 * @param n_baselines Number of baselines in the buffer.
 * @param base_value Base value for the data values, for distinguising buffers.
 *        For distinguishing baselines, this function adds baseline_nr * 100.0.
 *        When the buffer represents averaged data, the base_value should be
 *        the total of the base values of the original buffers.
 *        This function divides the base_value by the supplied weight so the
 *        caller does not have to do that division.
 * @param channel_counts List for generating channel data.
 *        For input buffers, this list should contain a 1 for each channel.
 *        When generating expected output data, this list should contain the
 *        number of averaged input buffers for each output buffer.
 * @param weight Weight value for the data values in the buffer.
 *
 * @note The function has been copied from @ref steps/test/unit/tIDGPredict.cc.
 */
static dp3::base::DPBuffer CreateBuffer(
    const double time, const double interval, std::size_t n_baselines,
    const std::vector<std::size_t>& channel_counts, const float base_value,
    const float weight = 1.0) {
  casacore::Cube<casacore::Complex> data(kNCorr, channel_counts.size(),
                                         n_baselines);
  casacore::Cube<bool> flags(data.shape(), false);
  casacore::Cube<float> weights(data.shape(), weight);
  casacore::Cube<bool> full_res_flags(channel_counts.size(), 1, n_baselines,
                                      false);
  casacore::Matrix<double> uvw(3, n_baselines);
  for (std::size_t bl = 0; bl < n_baselines; ++bl) {
    // Base value for this baseline.
    const float bl_value = (bl * 100.0) + (base_value / weight);

    std::size_t chan = 0;
    float chan_value = bl_value;  // Base value for a group of channels.
    for (std::size_t ch_count : channel_counts) {
      // For each channel, increase chan_base by 10.0.
      // When ch_count == 1, 'value' should equal chan_base.
      // When ch_count > 1, 'value' should be the average for multiple channels.
      const float value = chan_value + 5.0 * (ch_count - 1);
      for (unsigned int corr = 0; corr < kNCorr; ++corr) {
        data(corr, chan, bl) = value + corr;
        weights(corr, chan, bl) *= ch_count;
      }
      ++chan;
      chan_value += ch_count * 10.0;
    }
    uvw(0, bl) = bl_value + 0.0;
    uvw(1, bl) = bl_value + 1.0;
    uvw(2, bl) = bl_value + 2.0;
  }

  dp3::base::DPBuffer buffer;
  buffer.setTime(time);
  buffer.setExposure(interval);
  buffer.setData(data);
  buffer.setWeights(weights);
  buffer.setFlags(flags);
  buffer.setFullResFlags(full_res_flags);
  buffer.setUVW(uvw);
  return buffer;
}

BOOST_FIXTURE_TEST_CASE(showTimings, OnePredictFixture) {
  dp3::common::NSTimer timer;
  {
    const dp3::common::NSTimer::StartStop scoped_timer(timer);
    predict_->process(CreateBuffer(kStartTime * kInterval, kInterval,
                                   kNBaselines, kChannelCounts, 0.));
  }

  std::stringstream sstr;
  // Ensure the test doesn't depend on the system's locale settings.
  sstr.imbue(std::locale::classic());
  predict_->showTimings(sstr, timer.getElapsed());
  const std::string output = sstr.str();

  {
    // The output percentage is between "  0.x" and "100.x".
    // Percentages above the 100% aren't validated.
    const std::regex regex{
        R"(  (1[0-9]| [ 0-9])[0-9]\.[0-9]% \([ 0-9]{5} [m ]s\) OnePredict fixture.\n)"
        R"(          (1[0-9]| [ 0-9])[0-9]\.[0-9]% \([ 0-9]{5} [m ]s\) of it spent in predict\n)"
        R"(          (1[0-9]| [ 0-9])[0-9]\.[0-9]% \([ 0-9]{5} [m ]s\) of it spent in apply beam\n)"};
    BOOST_CHECK(std::regex_match(output.begin(), output.end(), regex));
  }
  {
    // At the moment no beam is applied so the percentages are fixed.
    // TODO Add an additional test to test with a beam applied.
    const std::regex regex{
        R"(  (1[0-9]| [ 0-9])[0-9]\.[0-9]% \([ 0-9]{5} [m ]s\) OnePredict fixture.\n)"
        R"(          100\.0% \([ 0-9]{5} [m ]s\) of it spent in predict\n)"
        R"(            0\.0% \(    0 ms\) of it spent in apply beam\n)"};
    BOOST_CHECK(std::regex_match(output.begin(), output.end(), regex));
  }
}

BOOST_AUTO_TEST_SUITE_END()
