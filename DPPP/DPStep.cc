// DPStep.cc: Abstract base class for a DPPP step
// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later
//
// @author Ger van Diepen

#include "DPStep.h"

#include <assert.h>

namespace DP3 {
namespace DPPP {

DPStep::~DPStep() {}

const DPInfo& DPStep::setInfo(const DPInfo& info) {
  // Update the info of this step using the given info.
  updateInfo(info);
  // If there is a next step, set its info using the info of this step.
  if (getNextStep()) {
    return getNextStep()->setInfo(getInfo());
  }
  return getInfo();
}

void DPStep::updateInfo(const DPInfo& infoIn) { info() = infoIn; }

void DPStep::addToMS(const string& msName) {
  if (itsPrevStep) itsPrevStep->addToMS(msName);
}

void DPStep::showCounts(std::ostream&) const {}

void DPStep::showTimings(std::ostream&, double) const {}

NullStep::~NullStep() {}

bool NullStep::process(const DPBuffer&) { return true; }

void NullStep::finish() {}

void NullStep::show(std::ostream&) const {}

ResultStep::ResultStep() { setNextStep(std::make_shared<NullStep>()); }

ResultStep::~ResultStep() {}

bool ResultStep::process(const DPBuffer& buf) {
  itsBuffer = buf;
  getNextStep()->process(buf);
  return true;
}

void ResultStep::finish() { getNextStep()->finish(); }

void ResultStep::show(std::ostream&) const {}

MultiResultStep::MultiResultStep(unsigned int size) : itsSize(0) {
  setNextStep(std::make_shared<NullStep>());
  itsBuffers.resize(size);
}

MultiResultStep::~MultiResultStep() {}

bool MultiResultStep::process(const DPBuffer& buf) {
  assert(itsSize < itsBuffers.size());
  itsBuffers[itsSize].copy(buf);
  itsSize++;
  getNextStep()->process(buf);
  return true;
}

void MultiResultStep::finish() { getNextStep()->finish(); }

void MultiResultStep::show(std::ostream&) const {}

}  // namespace DPPP
}  // namespace DP3
