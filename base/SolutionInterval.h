// SolutionInterval.h
//
// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

/// @file
/// @brief Solution Interval that can buffer multiple DPBuffers and data that is
/// relevant for using it in DDECal.
/// @author Lars Krombeen

#ifndef COMMON_SOLUTION_INTERVAL
#define COMMON_SOLUTION_INTERVAL

#include "../base/DPBuffer.h"
#include "../steps/InputStep.h"

#include <casacore/casa/Arrays/Cube.h>

namespace dp3 {
namespace base {

class DPBuffer;

class SolutionInterval {
 public:
  SolutionInterval(steps::InputStep* input, const std::size_t n_solution,
                   const std::size_t buffer_size, const std::size_t n_dir,
                   common::NSTimer timer);
  ~SolutionInterval();

  // Copy a buffer to the Solution Interval.
  void CopyBuffer(const DPBuffer&);

  /// Restore the flags and weights of added buffers to the original values
  void RestoreFlagsAndWeights();

  /// Resizes the pointer arrays so they only contain filled data.
  void Fit();

  /// Return the number of added buffers.
  const std::size_t Size() const { return buffer_index_; };

  /**
   * \defgroup Getters
   */
  /**@{*/
  const std::size_t NSolution() const { return n_solution_; }
  std::vector<DPBuffer>& DataBuffers() { return buffers_; }
  std::vector<std::vector<DPBuffer*>>& ModelBuffers() {
    return model_buffers_;
  };
  /**@}*/

  /**
   * \defgroup Operators
   */
  /**@{*/
  /// Get a DPBuffer at index @param i
  const DPBuffer& operator[](int i) const { return buffers_[i]; }
  /**@}*/

 private:
  const std::size_t buffer_size_;
  const std::size_t n_solution_;

  common::NSTimer timer_;  ///< Timer from the step that is using it for metrics
  steps::InputStep* input_;   ///< Input of DP3
  std::size_t buffer_index_;  ///< Current index where to insert the next buffer
  std::vector<DPBuffer> buffers_;  ///< Vector of DPBuffer copies

  std::vector<casacore::Cube<bool>> original_flags_;
  std::vector<casacore::Cube<float>> original_weights_;

  /// For each timeslot, a vector of nDir buffer pointers.
  /// These buffers contain the model data.
  std::vector<std::vector<DPBuffer*>> model_buffers_;
};

}  // namespace base
}  // namespace dp3

#endif
