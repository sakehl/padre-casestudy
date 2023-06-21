// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef INTERPOLATE_H
#define INTERPOLATE_H

#include <deque>
#include <string>

#include "InputStep.h"

#include <dp3/base/DPBuffer.h>

#include "../common/ParameterSet.h"

#include <aocommon/lane.h>

namespace dp3 {
namespace steps {

class Interpolate : public Step {
 public:
  /// Construct the object.
  /// Parameters are obtained from the parset using the given prefix.
  Interpolate(const common::ParameterSet&, const string& prefix);

  ~Interpolate() override = default;

  common::Fields getRequiredFields() const override {
    return kDataField | kFlagsField;
  }

  common::Fields getProvidedFields() const override {
    return kDataField | kFlagsField;
  }

  /// Process the data.
  /// It keeps the data.
  /// When processed, it invokes the process function of the next step.
  bool process(std::unique_ptr<base::DPBuffer> input_buffer) override;

  /// Finish the processing of this step and subsequent steps.
  void finish() override;

  /// Show the step parameters.
  void show(std::ostream&) const override;

  /// Show the timings.
  void showTimings(std::ostream&, double duration) const override;

 private:
  void interpolateTimestep(size_t index);
  void interpolateSample(size_t timestep, size_t baseline, size_t channel,
                         size_t pol);
  void sendFrontBufferToNextStep();
  void interpolationThread();

  struct Sample {
    Sample() = default;
    Sample(size_t timestep_, size_t baseline_, size_t channel_, size_t pol_)
        : timestep(timestep_),
          baseline(baseline_),
          channel(channel_),
          pol(pol_) {}
    size_t timestep;
    size_t baseline;
    size_t channel;
    size_t pol;
  };

  std::string _name;
  size_t _interpolatedPos;
  std::deque<std::unique_ptr<base::DPBuffer>> _buffers;
  size_t _windowSize;
  common::NSTimer _timer;
  aocommon::Lane<Sample> _lane;
  std::vector<float> _kernelLookup;
};

}  // namespace steps
}  // namespace dp3

#endif
