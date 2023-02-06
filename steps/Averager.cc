// Averager.cc: DP3 step class to average in time and/or freq
// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later
//
// @author Ger van Diepen

#include "Averager.h"

#include <iomanip>

#include <aocommon/parallelfor.h>
#include <casacore/casa/Arrays/ArrayMath.h>
#include <casacore/casa/Utilities/Regex.h>
#include <boost/algorithm/string/trim.hpp>

#include <dp3/base/DPBuffer.h>
#include <dp3/base/DPInfo.h>
#include "../base/FlagCounter.h"
#include "../common/ParameterSet.h"
#include "../common/StringTools.h"

using dp3::base::DPBuffer;
using dp3::base::DPInfo;

using casacore::Cube;
using casacore::IPosition;

namespace dp3 {
namespace steps {

const common::Fields Averager::kRequiredFields =
    kDataField | kFlagsField | kWeightsField | kFullResFlagsField | kUvwField;
const common::Fields Averager::kProvidedFields = kRequiredFields;

Averager::Averager(const common::ParameterSet& parset, const string& prefix)
    : itsName(prefix),
      itsMinNPoint(parset.getUint(prefix + "minpoints", 1)),
      itsMinPerc(parset.getFloat(prefix + "minperc", 0.) / 100.),
      itsNTimes(0),
      itsOriginalTimeInterval(0),
      itsNoAvg(true) {
  string freqResolutionStr = parset.getString(prefix + "freqresolution", "0");
  itsFreqResolution = getFreqHz(freqResolutionStr);

  if (itsFreqResolution > 0) {
    itsNChanAvg = 0;  // Will be set later in updateinfo
  } else {
    itsNChanAvg = parset.getUint(prefix + "freqstep", 1);
  }

  itsTimeResolution = parset.getFloat(prefix + "timeresolution", 0.);
  if (itsTimeResolution > 0) {
    itsNTimeAvg = 0;  // Will be set later in updateInfo
  } else {
    itsNTimeAvg = parset.getUint(prefix + "timestep", 1);
  }
}

Averager::Averager(const string& stepName, unsigned int nchanAvg,
                   unsigned int ntimeAvg)
    : itsName(stepName),
      itsFreqResolution(0),
      itsTimeResolution(0),
      itsNChanAvg(nchanAvg == 0 ? 1 : nchanAvg),
      itsNTimeAvg(ntimeAvg == 0 ? 1 : ntimeAvg),
      itsMinNPoint(1),
      itsMinPerc(0),
      itsNTimes(0),
      itsOriginalTimeInterval(0),
      itsNoAvg(itsNChanAvg == 1 && itsNTimeAvg == 1) {}

Averager::Averager(const string& stepName, double freq_resolution,
                   double time_resolution)
    : itsName(stepName),
      itsFreqResolution(freq_resolution),
      itsTimeResolution(time_resolution),
      itsNChanAvg(0),
      itsNTimeAvg(0),
      itsMinNPoint(1),
      itsMinPerc(0),
      itsNTimes(0),
      itsOriginalTimeInterval(0),
      itsNoAvg(itsNChanAvg == 1 && itsNTimeAvg == 1) {}

Averager::~Averager() {}

void Averager::updateInfo(const base::DPInfo& infoIn) {
  Step::updateInfo(infoIn);
  info().setMetaChanged();

  if (itsNChanAvg <= 0) {
    if (itsFreqResolution > 0) {
      double chanwidth = infoIn.chanWidths()[0];
      itsNChanAvg = std::max(1, int(itsFreqResolution / chanwidth + 0.5));
    } else {
      itsNChanAvg = 1;
    }
  }

  itsOriginalTimeInterval = infoIn.timeInterval();
  if (itsNTimeAvg <= 0) {
    if (itsTimeResolution > 0) {
      itsNTimeAvg =
          std::max(1, int(itsTimeResolution / itsOriginalTimeInterval + 0.5));
    } else {
      itsNTimeAvg = 1;
    }
  }

  itsNoAvg = (itsNChanAvg == 1 && itsNTimeAvg == 1);

  // Adapt averaging to available nr of channels and times.
  itsNTimeAvg = std::min(itsNTimeAvg, infoIn.ntime());
  itsNChanAvg = info().update(itsNChanAvg, itsNTimeAvg);

  if (!itsNoAvg) loop_.SetNThreads(getInfo().nThreads());
}

void Averager::show(std::ostream& os) const {
  os << "Averager " << itsName << '\n';
  os << "  freqstep:       " << itsNChanAvg;
  if (itsFreqResolution > 0) {
    os << " (set by freqresolution: " << itsFreqResolution << " Hz)" << '\n';
  }
  os << "  timestep:       " << itsNTimeAvg;
  if (itsTimeResolution > 0) {
    os << " (set by timeresolution: " << itsTimeResolution << ")";
  }
  os << '\n';
  os << "  minpoints:      " << itsMinNPoint << '\n';
  os << "  minperc:        " << 100 * itsMinPerc << '\n';
}

void Averager::showTimings(std::ostream& os, double duration) const {
  os << "  ";
  base::FlagCounter::showPerc1(os, itsTimer.getElapsed(), duration);
  os << " Averager " << itsName << '\n';
}

bool Averager::process(std::unique_ptr<base::DPBuffer> buffer) {
  // Nothing needs to be done if no averaging.
  if (itsNoAvg) {
    getNextStep()->process(std::move(buffer));
    return true;
  }
  itsTimer.start();
  // Sum the data in time applying the weights.
  // The summing in channel and the averaging is done in function average.
  if (itsNTimes == 0) {
    // The first time we move because that is faster than first clearing
    // and adding thereafter.
    assert(!itsBuf);
    itsBuf = std::move(buffer);
    // Ensure `itsBuf` does not refer to data in other DPBuffers.
    // Full res flags are not needed here, since they are resized below.
    itsBuf->MakeIndependent(kDataField | kWeightsField | kFlagsField |
                            kUvwField);
    IPosition shapeIn = itsBuf->GetCasacoreData().shape();
    itsNPoints.resize(shapeIn);
    itsAvgAll.reference(itsBuf->GetCasacoreData() *
                        itsBuf->GetCasacoreWeights());
    itsWeightAll.resize(shapeIn);
    itsWeightAll = itsBuf->GetCasacoreWeights();
    // Take care of the fullRes flags.
    // We have to shape the output array and copy to a part of it.

    // Make sure the current fullResFlags are up to date with the flags
    DPBuffer::mergeFullResFlags(itsBuf->GetCasacoreFullResFlags(),
                                itsBuf->GetCasacoreFlags());

    // Extract fullResFlags before resizing the field in itsBuffer.
    casacore::Cube<bool> full_res_flags =
        std::move(itsBuf->GetCasacoreFullResFlags());

    const IPosition shape = full_res_flags.shape();
    // More time entries, same chan and bl
    itsBuf->ResizeFullResFlags(shape[2], shape[1] * itsNTimeAvg, shape[0]);
    itsBuf->GetFullResFlags().fill(
        true);  // initialize for times missing at end
    copyFullResFlags(full_res_flags, itsBuf->GetCasacoreFlags(), 0);

    // Set middle of new interval.
    const double time = itsBuf->getTime() + 0.5 * (getInfo().timeInterval() -
                                                   itsOriginalTimeInterval);
    itsBuf->setTime(time);
    itsBuf->setExposure(getInfo().timeInterval());
    // Only set.
    itsNPoints = 1;
    // Set flagged points to zero.
    casacore::Array<bool>::const_contiter infIter =
        itsBuf->GetCasacoreFlags().cbegin();
    casacore::Array<casacore::Complex>::contiter dataIter =
        itsBuf->GetCasacoreData().cbegin();
    casacore::Array<float>::contiter wghtIter =
        itsBuf->GetCasacoreWeights().cbegin();
    casacore::Array<int>::contiter outnIter = itsNPoints.cbegin();
    casacore::Array<int>::contiter outnIterEnd = itsNPoints.cend();
    while (outnIter != outnIterEnd) {
      if (*infIter) {
        // Flagged data point
        *outnIter = 0;
        *dataIter = casacore::Complex();
        *wghtIter = 0;
      } else {
        // Weigh the data point
        *dataIter *= *wghtIter;
      }
      ++infIter;
      ++dataIter;
      ++wghtIter;
      ++outnIter;
    }
  } else {
    // Not the first time.
    // For now we assume that all timeslots have the same nr of baselines,
    // so check if the buffer sizes are the same.
    assert(itsBuf);
    if (itsBuf->GetCasacoreData().shape() != buffer->GetCasacoreData().shape())
      throw std::runtime_error(
          "Inconsistent buffer sizes in Averager, possibly because of "
          "inconsistent nr of baselines in timeslots");
    itsBuf->GetCasacoreUvw() += buffer->GetCasacoreUvw();

    // Make sure the current fullResFlags are up to date with the flags
    DPBuffer::mergeFullResFlags(buffer->GetCasacoreFullResFlags(),
                                buffer->GetCasacoreFlags());
    copyFullResFlags(buffer->GetCasacoreFullResFlags(),
                     buffer->GetCasacoreFlags(), itsNTimes);
    const Cube<float>& weights = buffer->GetCasacoreWeights();
    // Ignore flagged points.
    casacore::Array<casacore::Complex>::const_contiter indIter =
        buffer->GetCasacoreData().cbegin();
    casacore::Array<float>::const_contiter inwIter = weights.cbegin();
    casacore::Array<bool>::const_contiter infIter =
        buffer->GetCasacoreFlags().cbegin();
    casacore::Array<casacore::Complex>::contiter outdIter =
        itsBuf->GetCasacoreData().cbegin();
    casacore::Array<casacore::Complex>::contiter alldIter = itsAvgAll.cbegin();
    casacore::Array<float>::contiter outwIter =
        itsBuf->GetCasacoreWeights().cbegin();
    casacore::Array<float>::contiter allwIter = itsWeightAll.cbegin();
    casacore::Array<int>::contiter outnIter = itsNPoints.cbegin();
    casacore::Array<int>::contiter outnIterEnd = itsNPoints.cend();
    while (outnIter != outnIterEnd) {
      *alldIter += *indIter * *inwIter;
      *allwIter += *inwIter;
      if (!*infIter) {
        *outdIter += *indIter * *inwIter;
        *outwIter += *inwIter;
        (*outnIter)++;
      }
      ++indIter;
      ++inwIter;
      ++infIter;
      ++outdIter;
      ++alldIter;
      ++outwIter;
      ++allwIter;
      ++outnIter;
    }
  }
  // Do the averaging if enough time steps have been processed.
  itsNTimes += 1;
  if (itsNTimes >= itsNTimeAvg) {
    average();
    itsTimer.stop();
    getNextStep()->process(std::move(itsBuf));
    itsBuf.reset();
    itsNTimes = 0;
  } else {
    itsTimer.stop();
  }
  return true;
}

void Averager::finish() {
  // Average remaining entries.
  if (itsNTimes > 0) {
    itsTimer.start();
    average();
    itsTimer.stop();
    getNextStep()->process(std::move(itsBuf));
    itsNTimes = 0;
  }
  // Let the next steps finish.
  getNextStep()->finish();
}

void Averager::average() {
  IPosition shape = itsBuf->GetCasacoreData().shape();
  const unsigned int nchanin = shape[1];
  const unsigned int npin = shape[0] * nchanin;
  shape[1] = (shape[1] + itsNChanAvg - 1) / itsNChanAvg;
  const unsigned int ncorr = shape[0];
  const unsigned int nchan = shape[1];
  const unsigned int nbl = shape[2];
  casacore::Cube<casacore::Complex> data_out(ncorr, nchan, nbl);
  casacore::Cube<float> weights_out(ncorr, nchan, nbl);
  casacore::Cube<bool> flags_out(ncorr, nchan, nbl);
  unsigned int npout = ncorr * nchan;
  loop_.Run(0, nbl, [&](size_t begin, size_t end) {
    for (unsigned int k = begin; k != end; ++k) {
      const casacore::Complex* indata = itsBuf->GetData().data() + k * npin;
      const casacore::Complex* inalld = itsAvgAll.data() + k * npin;
      const float* inwght = itsBuf->GetWeights().data() + k * npin;
      const float* inallw = itsWeightAll.data() + k * npin;
      const int* innp = itsNPoints.data() + k * npin;
      casacore::Complex* outdata = data_out.data() + k * npout;
      float* outwght = weights_out.data() + k * npout;
      bool* outflags = flags_out.data() + k * npout;
      for (unsigned int i = 0; i < ncorr; ++i) {
        unsigned int inxi = i;
        unsigned int inxo = i;
        for (unsigned int ch = 0; ch < nchan; ++ch) {
          unsigned int nch = std::min(itsNChanAvg, nchanin - ch * itsNChanAvg);
          unsigned int navgAll = nch * itsNTimes;
          casacore::Complex sumd;
          casacore::Complex sumad;
          float sumw = 0;
          float sumaw = 0;
          unsigned int np = 0;
          for (unsigned int j = 0; j < nch; ++j) {
            sumd += indata[inxi];  // Note: weight is accounted for in process
            sumad += inalld[inxi];
            sumw += inwght[inxi];
            sumaw += inallw[inxi];
            np += innp[inxi];
            inxi += ncorr;
          }
          // Flag the point if insufficient unflagged data.
          if (sumw == 0 || np < itsMinNPoint || np < navgAll * itsMinPerc) {
            outdata[inxo] = (sumaw == 0 ? casacore::Complex() : sumad / sumaw);
            outflags[inxo] = true;
            outwght[inxo] = sumaw;
          } else {
            outdata[inxo] = sumd / sumw;
            outflags[inxo] = false;
            outwght[inxo] = sumw;
          }
          inxo += ncorr;
        }
      }
    }
  });
  // Put the averaged values in the buffer.
  itsBuf->setData(data_out);
  itsBuf->setWeights(weights_out);
  itsBuf->setFlags(flags_out);
  // The result UVWs are the average of the input.
  // If ever needed, UVWCalculator can be used to calculate the UVWs.
  itsBuf->GetUvw() /= double(itsNTimes);
}

void Averager::copyFullResFlags(const Cube<bool>& fullResFlags,
                                const Cube<bool>& flags, int timeIndex) {
  // Copy the fullRes flags to the given index.
  // Furthermore the appropriate FullRes flags are set for a
  // flagged data point. It can be the case that an input data point
  // has been averaged before, thus has fewer channels than FullResFlags.
  // nchan and nbl are the same for in and out.
  // ntimout is a multiple of ntimavg.
  IPosition shapeIn = fullResFlags.shape();
  IPosition shapeOut = itsBuf->GetCasacoreFullResFlags().shape();
  IPosition shapeFlg = flags.shape();
  unsigned int nchan = shapeIn[0];    // original nr of channels
  unsigned int ntimavg = shapeIn[1];  // nr of averaged times in input data
  unsigned int nchanavg = nchan / shapeFlg[1];  // nr of avg chan in input data
  unsigned int ntimout = shapeOut[1];  // nr of averaged times in output data
  unsigned int nbl = shapeIn[2];       // nr of baselines
  unsigned int ncorr = shapeFlg[0];    // nr of correlations (in FLAG)
  // in has to be copied to the correct time index in out.
  bool* outBase =
      itsBuf->GetFullResFlags().data() + nchan * ntimavg * timeIndex;
  for (unsigned int k = 0; k < nbl; ++k) {
    const bool* inPtr = fullResFlags.data() + k * nchan * ntimavg;
    const bool* flagPtr = flags.data() + k * ncorr * shapeFlg[1];
    bool* outPtr = outBase + k * nchan * ntimout;
    memcpy(outPtr, inPtr, nchan * ntimavg * sizeof(bool));
    // Applying the flags only needs to be done if the input data
    // was already averaged before.
    if (ntimavg > 1 || nchanavg > 1) {
      for (int j = 0; j < shapeFlg[1]; ++j) {
        // If a data point is flagged, the flags in the corresponding
        // FullRes window have to be set.
        // Only look at the flags of the first correlation.
        if (*flagPtr) {
          bool* avgPtr = outPtr + j * nchanavg;
          for (unsigned int i = 0; i < ntimavg; ++i) {
            std::fill(avgPtr, avgPtr + nchanavg, true);
            avgPtr += nchan;
          }
        }
        flagPtr += ncorr;
      }
    }
  }
}

double Averager::getFreqHz(const string& freqstr) {
  casacore::String unit;
  // See if a unit is given at the end.
  casacore::String v(freqstr);
  // Remove possible trailing blanks.
  boost::algorithm::trim_right(v);
  casacore::Regex regex("[a-zA-Z]+$");
  casacore::String::size_type pos = v.index(regex);
  if (pos != casacore::String::npos) {
    unit = v.from(pos);
    v = v.before(pos);
  }
  // Set value and unit.

  double value = common::strToDouble(v);
  if (unit.empty()) {
    return value;
  } else {
    casacore::Quantity q(value, unit);
    return q.getValue("Hz", true);
  }
}

}  // namespace steps
}  // namespace dp3
