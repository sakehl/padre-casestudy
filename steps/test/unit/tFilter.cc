// tFilter.cc: Test program for class Filter
// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later
//
// @author Ger van Diepen

#include "../../Filter.h"

#include <casacore/casa/Arrays/ArrayMath.h>
#include <casacore/casa/Arrays/ArrayLogical.h>
#include <casacore/casa/BasicMath/Math.h>

#include <boost/test/unit_test.hpp>

#include <dp3/base/DPBuffer.h>
#include <dp3/base/DPInfo.h>

#include "tStepCommon.h"
#include "mock/MockInput.h"
#include "mock/ThrowStep.h"
#include "../../../common/ParameterSet.h"
#include "../../../common/StringTools.h"

using dp3::base::DPBuffer;
using dp3::base::DPInfo;
using dp3::common::ParameterSet;
using dp3::steps::Filter;
using dp3::steps::Step;

BOOST_AUTO_TEST_SUITE(filter)

BOOST_AUTO_TEST_CASE(fields_default) {
  const ParameterSet parset;
  const Filter filter(parset, "");
  BOOST_CHECK_EQUAL(filter.getProvidedFields(), dp3::common::Fields());
  BOOST_CHECK_EQUAL(filter.getRequiredFields(), dp3::common::Fields());
}

BOOST_AUTO_TEST_CASE(fields_channel_selection) {
  // Test fields when enabling channel filtering.
  ParameterSet parset;
  parset.add("startchan", "4");
  parset.add("nchan", "2");
  const Filter channel_only(parset, "");
  const dp3::common::Fields kExpectedFields =
      Step::kDataField | Step::kFlagsField | Step::kWeightsField |
      Step::kFullResFlagsField;
  BOOST_CHECK_EQUAL(channel_only.getRequiredFields(), kExpectedFields);
  BOOST_CHECK_EQUAL(channel_only.getProvidedFields(), kExpectedFields);

  ParameterSet parset_remove_ant = parset;
  parset_remove_ant.add("remove", "true");
  const Filter remove_ant(parset_remove_ant, "");
  const dp3::common::Fields kExpectedFieldsRemove =
      kExpectedFields | Step::kUvwField;
  BOOST_CHECK_EQUAL(remove_ant.getRequiredFields(), kExpectedFieldsRemove);
  BOOST_CHECK_EQUAL(remove_ant.getProvidedFields(), kExpectedFieldsRemove);
}

BOOST_AUTO_TEST_CASE(fields_baseline_selection) {
  ParameterSet parset;
  parset.add("baseline", "42");
  const Filter baseline_only(parset, "");

  const dp3::common::Fields kExpectedFields =
      Step::kDataField | Step::kFlagsField | Step::kWeightsField |
      Step::kFullResFlagsField | Step::kUvwField;
  BOOST_CHECK_EQUAL(baseline_only.getRequiredFields(), kExpectedFields);
  BOOST_CHECK_EQUAL(baseline_only.getProvidedFields(), kExpectedFields);

  parset.add("startchan", "4");
  parset.add("nchan", "2");
  const Filter baseline_and_channel(parset, "");
  BOOST_CHECK_EQUAL(baseline_and_channel.getRequiredFields(), kExpectedFields);
  BOOST_CHECK_EQUAL(baseline_and_channel.getProvidedFields(), kExpectedFields);
}

// Simple class to generate input arrays.
// It can only set all flags to true or all false.
// Weights are always 1.
// It can be used with different nr of times, channels, etc.
class TestInput : public dp3::steps::MockInput {
 public:
  TestInput(int ntime, int nbl, int nchan, int ncorr, bool flag)
      : itsCount(0),
        itsNTime(ntime),
        itsNBl(nbl),
        itsNChan(nchan),
        itsNCorr(ncorr),
        itsFlag(flag) {
    // Define start time 0.5 (= 3 - 0.5*5) and time interval 5.
    info().init(ncorr, 0, nchan, ntime, 0.5, 5.0, std::string());
    // Fill the baseline stations; use 4 stations.
    // So they are called 00 01 02 03 10 11 12 13 20, etc.
    std::vector<int> ant1(nbl);
    std::vector<int> ant2(nbl);
    int st1 = 0;
    int st2 = 0;
    for (int i = 0; i < nbl; ++i) {
      ant1[i] = st1;
      ant2[i] = st2;
      if (++st2 == 4) {
        st2 = 0;
        if (++st1 == 4) {
          st1 = 0;
        }
      }
    }
    std::vector<std::string> antNames{"rs01.s01", "rs02.s01", "cs01.s01",
                                      "cs01.s02"};
    // Define their positions (more or less WSRT RT0-3).
    std::vector<casacore::MPosition> antPos(4);
    casacore::Vector<double> vals(3);
    vals[0] = 3828763;
    vals[1] = 442449;
    vals[2] = 5064923;
    antPos[0] = casacore::MPosition(
        casacore::Quantum<casacore::Vector<double>>(vals, "m"),
        casacore::MPosition::ITRF);
    vals[0] = 3828746;
    vals[1] = 442592;
    vals[2] = 5064924;
    antPos[1] = casacore::MPosition(
        casacore::Quantum<casacore::Vector<double>>(vals, "m"),
        casacore::MPosition::ITRF);
    vals[0] = 3828729;
    vals[1] = 442735;
    vals[2] = 5064925;
    antPos[2] = casacore::MPosition(
        casacore::Quantum<casacore::Vector<double>>(vals, "m"),
        casacore::MPosition::ITRF);
    vals[0] = 3828713;
    vals[1] = 442878;
    vals[2] = 5064926;
    antPos[3] = casacore::MPosition(
        casacore::Quantum<casacore::Vector<double>>(vals, "m"),
        casacore::MPosition::ITRF);
    std::vector<double> antDiam(4, 70.);
    info().set(antNames, antDiam, antPos, ant1, ant2);
    // Define the frequencies.
    std::vector<double> chanWidth(nchan, 100000.);
    std::vector<double> chanFreqs(nchan);
    for (int i = 0; i < nchan; i++) {
      chanFreqs.push_back(1050000. + i * 100000.);
    }
    info().set(std::move(chanFreqs), std::move(chanWidth));
  }

 private:
  virtual bool process(const DPBuffer&) {
    // Stop when all times are done.
    if (itsCount == itsNTime) {
      return false;
    }
    DPBuffer buf;
    buf.setTime(itsCount * 5 + 2);
    buf.setExposure(0.1 * (itsCount + 1));
    casacore::Cube<casacore::Complex> data(itsNCorr, itsNChan, itsNBl);
    for (int i = 0; i < int(data.size()); ++i) {
      data.data()[i] =
          casacore::Complex(i + itsCount * 10, i - 1000 + itsCount * 6);
    }
    buf.setData(data);
    casacore::Cube<float> weights(data.shape());
    buf.setWeights(weights);
    indgen(weights);
    casacore::Cube<bool> flags(data.shape());
    flags = itsFlag;
    // Set part of the flags to another value.
    flags(casacore::IPosition(3, 0), flags.shape() - 1,
          casacore::IPosition(3, 1, 3, 4)) = !itsFlag;
    buf.setFlags(flags);
    // The fullRes flags are a copy of the XX flags, but differently shaped.
    // Assume they are averaged for 2 chan, 2 time.
    casacore::Cube<bool> fullResFlags;
    if (itsNCorr == 4) {
      fullResFlags =
          flags.copy().reform(casacore::IPosition(3, 2 * itsNChan, 2, itsNBl));
    } else {
      fullResFlags.resize(casacore::IPosition(3, itsNChan, 1, itsNBl));
      fullResFlags = true;
    }
    buf.setFullResFlags(fullResFlags);
    casacore::Matrix<double> uvw(3, itsNBl);
    indgen(uvw, double(itsCount * 100));
    buf.setUVW(uvw);
    getNextStep()->process(buf);
    ++itsCount;
    return true;
  }

  virtual void finish() { getNextStep()->finish(); }
  virtual void updateInfo(const DPInfo&) {
    // Use timeInterval=5
    info().init(itsNCorr, 0, itsNChan, itsNTime, 100, 5, std::string());
    // Define the frequencies.
    std::vector<double> chanFreqs;
    std::vector<double> chanWidth(itsNChan, 100000.);
    for (int i = 0; i < itsNChan; i++) {
      chanFreqs.push_back(1050000. + i * 100000.);
    }
    info().set(std::move(chanFreqs), std::move(chanWidth));
  }
  int itsCount, itsNTime, itsNBl, itsNChan, itsNCorr;
  bool itsFlag;
};

// Class to check result of averaging TestInput.
class TestOutput : public dp3::steps::test::ThrowStep {
 public:
  TestOutput(int ntime, int nbl, int nchan, int ncorr, int nblout, int stchan,
             int nchanOut, bool flag)
      : itsCount(0),
        itsNTime(ntime),
        itsNBl(nbl),
        itsNChan(nchan),
        itsNCorr(ncorr),
        itsNBlOut(nblout),
        itsStChan(stchan),
        itsNChanOut(nchanOut),
        itsFlag(flag) {}

 private:
  virtual bool process(const DPBuffer& buf) {
    // Fill expected result in similar way as TestInput.
    casacore::Cube<casacore::Complex> data(itsNCorr, itsNChan, itsNBl);
    for (int i = 0; i < int(data.size()); ++i) {
      data.data()[i] =
          casacore::Complex(i + itsCount * 10, i - 1000 + itsCount * 6);
    }
    casacore::Cube<float> weights(data.shape());
    indgen(weights);
    casacore::Cube<bool> flags(data.shape());
    flags = itsFlag;
    // Set part of the flags to another value.
    flags(casacore::IPosition(3, 0), flags.shape() - 1,
          casacore::IPosition(3, 1, 3, 4)) = !itsFlag;
    // The fullRes flags are a copy of the XX flags, but differently shaped.
    // Assume they are averaged for 2 chan, 2 time.
    casacore::Cube<bool> fullResFlags;
    if (itsNCorr == 4) {
      fullResFlags =
          flags.copy().reform(casacore::IPosition(3, 2 * itsNChan, 2, itsNBl));
    } else {
      fullResFlags.resize(casacore::IPosition(3, itsNChan, 1, itsNBl));
      fullResFlags = true;
    }
    casacore::Matrix<double> uvw(3, itsNBl);
    indgen(uvw, double(itsCount * 100));
    casacore::Slicer slicer(
        casacore::IPosition(3, 0, itsStChan, 0),
        casacore::IPosition(3, itsNCorr, itsNChanOut, itsNBlOut));
    // Check the expected result.
    BOOST_CHECK(allEQ(buf.getData(), data(slicer)));
    BOOST_CHECK(allEQ(buf.getFlags(), flags(slicer)));
    BOOST_CHECK(allEQ(buf.getWeights(), weights(slicer)));
    BOOST_CHECK(
        allEQ(buf.getUVW(), uvw(casacore::IPosition(2, 0, 0),
                                casacore::IPosition(2, 2, itsNBlOut - 1))));
    if (itsNCorr == 4) {
      BOOST_CHECK(
          allEQ(buf.getFullResFlags(),
                fullResFlags(casacore::Slicer(
                    casacore::IPosition(3, itsStChan * 2, 0, 0),
                    casacore::IPosition(3, 2 * itsNChanOut, 2, itsNBlOut)))));
    } else {
      BOOST_CHECK(
          allEQ(buf.getFullResFlags(),
                fullResFlags(casacore::Slicer(
                    casacore::IPosition(3, itsStChan, 0, 0),
                    casacore::IPosition(3, itsNChanOut, 1, itsNBlOut)))));
    }
    BOOST_CHECK(casacore::near(buf.getTime(), itsCount * 5. + 2));
    BOOST_CHECK(casacore::near(buf.getExposure(), 0.1 * (itsCount + 1)));
    ++itsCount;
    return true;
  }

  virtual void finish() {}
  virtual void updateInfo(const DPInfo& info) {
    BOOST_CHECK_EQUAL(itsNChan, int(info.origNChan()));
    BOOST_CHECK_EQUAL(itsNChanOut, int(info.nchan()));
    BOOST_CHECK_EQUAL(itsNBlOut, int(info.nbaselines()));
    BOOST_CHECK_EQUAL(itsNTime, int(info.ntime()));
    BOOST_CHECK_EQUAL(5., info.timeInterval());
    BOOST_CHECK_EQUAL(1, int(info.nchanAvg()));
    BOOST_CHECK_EQUAL(1, int(info.ntimeAvg()));
  }

  int itsCount;
  int itsNTime, itsNBl, itsNChan, itsNCorr, itsNBlOut, itsStChan, itsNChanOut;
  bool itsFlag;
};

void TestChannelsOnly(int ntime, int nbl, int nchan, int ncorr, int startchan,
                      int nchanout, bool flag) {
  auto in = std::make_shared<TestInput>(ntime, nbl, nchan, ncorr, flag);
  ParameterSet parset;
  parset.add("startchan", std::to_string(startchan));
  parset.add("nchan", std::to_string(nchanout) + "+nchan-nchan");
  auto filter = std::make_shared<Filter>(parset, "");
  auto out = std::make_shared<TestOutput>(ntime, nbl, nchan, ncorr, nbl,
                                          startchan, nchanout, flag);
  dp3::steps::test::Execute({in, filter, out});
}

void TestChannelsAndBaselines(int ntime, int nbl, int nchan, int ncorr,
                              int startchan, int nchanout, bool flag) {
  BOOST_CHECK(nbl <=
              4);  // otherwise baseline selection removes more than the first

  auto in = std::make_shared<TestInput>(ntime, nbl, nchan, ncorr, flag);
  ParameterSet parset;
  parset.add("startchan", std::to_string(startchan) + "+nchan-nchan");
  parset.add("nchan", std::to_string(nchanout));
  // This removes the first baseline.
  parset.add("baseline", "[[rs01.s01,rs*]]");
  auto filter = std::make_shared<Filter>(parset, "");
  auto out = std::make_shared<TestOutput>(ntime, nbl, nchan, ncorr, 2,
                                          startchan, nchanout, flag);
  dp3::steps::test::Execute({in, filter, out});
}

BOOST_AUTO_TEST_CASE(filter_channels_1) {
  TestChannelsOnly(10, 3, 32, 4, 2, 24, false);
}

BOOST_AUTO_TEST_CASE(filter_channels_2) {
  TestChannelsOnly(10, 10, 30, 1, 3, 3, true);
}

BOOST_AUTO_TEST_CASE(filter_channels_3) {
  TestChannelsOnly(10, 10, 1, 4, 0, 1, true);
}

BOOST_AUTO_TEST_CASE(channels_and_baselines) {
  TestChannelsAndBaselines(10, 4, 32, 4, 2, 24, false);
}

BOOST_AUTO_TEST_SUITE_END()
