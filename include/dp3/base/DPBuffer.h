// DPBuffer.h: Buffer holding the data of a timeslot/band
// Copyright (C) 2023 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

/// @file
/// @brief Buffer holding the data of a timeslot/band
/// @author Ger van Diepen

#ifndef DP3_BASE_DPBUFFER_H_
#define DP3_BASE_DPBUFFER_H_

#include <array>
#include <map>
#include <vector>

#include <xtensor/xtensor.hpp>

#include <casacore/casa/Arrays/Vector.h>
#include <aocommon/xt/utensor.h>

#include <dp3/common/Fields.h>
#include <dp3/common/Types.h>

namespace dp3 {
namespace base {

/// @brief Buffer holding the data of a timeslot/band

/// This class holds the data for one time slot in Array variables.
/// It makes heavy use of reference semantics to avoid data copying
/// when data are pushed from one step to another.
/// This means that a data array can be shared between Step objects.
/// So, if a Step object changes data in a buffer, it has to be sure
/// it can do it. If needed, Array::unique should be called to ensure
/// the array is not shared.
///
/// The following data can be kept in a DPBuffer object.
/// <table>
///  <tr>
///   <td>TIME</td>
///   <td>The time slot center of the current data (in MJD seconds).</td>
///  </tr>
///  <tr>
///   <td>ROWNRS</td>
///   <td>The row numbers of the current time slot. It can be empty
///       when e.g. a time slot is inserted or if data are averaged.</td>
///  </tr>
///  <tr>
///   <td>DATA</td>
///   <td>The visibility data as [ncorr,nchan,nbaseline].</td>
///  </tr>
///  <tr>
///   <td>FLAG</td>
///   <td>The data flags as [ncorr,nchan,nbaseline] (True is bad).
///       Note that the ncorr axis is redundant because NDPPP will always
///       have the same flag for all correlations. The reason all
///       correlations are there is because the MS expects them.</td>
///  </tr>
///  <tr>
///   <td>WEIGHT</td>
///   <td>The data weights as [ncorr,nchan,nbaseline].
///       Similarly to FLAG, the ncorr axis is redundant because the
///       same weight is used for all correlations.</td>
///  </tr>
///  <tr>
///   <td>UVW</td>
///   <td>The UVW coordinates in meters as [3,nbaseline].</td>
///  </tr>
/// </table>
/// Each data member (DATA, FLAG, UVW, WEIGHTS) is filled in if
/// any Step needs it (the information about the required fields per each step
/// can be read with the getRequiredFields() function). The first Step
/// (MSReader) will read the requested fields from the MS into the DPBuffer. In
/// that way as little memory as needed is used. Note that e.g. the AOFlagger
/// can use a lot of memory if a large time window is used.
///
/// Until early 2015 DP3 used the strategy of shallow data copies.
/// I.e., a step increased the data reference counter and did not make
/// an actual copy. Only when data were changed, a new data array was made.
/// Thus, MSReader allocated a new array when it read the data.
/// However, it appeared this strategy lead to memory fragmentation and
/// to sudden jumps in memory usage on Linux systems.
/// <br>Therefore the strategy was changed to having each step preallocate
/// its buffers and making deep copies when moving data from one step to
/// the next one. It appeared that it not only improved memory usage,
/// but also improved performance, possible due to far less mallocs.
///
/// The buffer/step guidelines are as follows:
/// 1. If a step keeps a buffer of DPBuffers for later processing (e.g.
///    AOFlagger), it must make a copy of the DPBuffers because the input data
///    arrays might have changed before that step processes the data.
/// 2. A shallow copy of a data member can be used if a step processes
///    the data immediately (e.g. Averager).
class DPBuffer {
 public:
  using Complex = std::complex<float>;

  // Return types for Get* functions.
  using DataType = aocommon::xt::UTensor<std::complex<float>, 3>;
  using WeightsType = xt::xtensor<float, 3>;
  using FlagsType = xt::xtensor<bool, 3>;
  using UvwType = xt::xtensor<double, 2>;
  /// For every channel, contains n_antennas x n_polarizations solutions.
  using SolutionType = std::vector<std::vector<std::complex<double>>>;

  /// Construct object with empty arrays.
  explicit DPBuffer(double time = 0.0, double exposure = 0.0);

  /// The copy constructor copies all data from the source buffer.
  /// It copies row numbers using reference semantics.
  DPBuffer(const DPBuffer&) = default;

  /// The move constructor moves all data without using reference semantics.
  DPBuffer(DPBuffer&&);

  /// This constructor copies the given fields only, without using reference
  /// semantics. It copies row numbers using reference semantics.
  /// It does not copy extra data fields yet (TODO in AST-1241).
  DPBuffer(const DPBuffer& that, const common::Fields& fields);

  /// Copy assignment copies all data from the source buffer.
  /// It copies row numbers using reference semantics.
  DPBuffer& operator=(const DPBuffer&);

  /// Move assignment moves all data without using reference semantics.
  DPBuffer& operator=(DPBuffer&&);

  /// Copy that to this.
  /// Copy row numbers using reference semantics. Copy all other members using
  /// value semantics.
  /// Do not copy extra data fields yet (TODO in AST-1241).
  /// @param fields Copy the given fields from that.
  void Copy(const DPBuffer& that, const common::Fields& fields);

  /// Adds an extra visibility buffer.
  /// The new visibility buffer gets the same shape as the default data buffer.
  /// @param name Name for the new buffer. The name may not be empty. It may
  ///        also not equal the name of an existing extra buffer.
  void AddData(const std::string& name);

  /// Removes extra visibility buffers
  /// @param name Name of the buffer to remove. If empty, removes all extra data
  /// buffers.
  void RemoveData(const std::string& name = "");

  /// Check if the DPBuffer has a data buffer for the given name.
  /// @param name Name. An empty string indicates the main data buffer (always
  ///        exists). A non-empty string indicates an extra data buffer.
  /// @return If the requested data buffer exists. It can be empty, though.
  [[nodiscard]] bool HasData(const std::string& name = "") const {
    return name.empty() || (extra_data_.find(name) != extra_data_.end());
  }

  /// Accesses data (visibilities) in the DPBuffer.
  ///
  /// @param name Data buffer name. An empty string indicates the main data
  ///        buffer. A non-empty string indicates an extra data buffer.
  /// @return An XTensor view to the data in the DPBuffer for the given name.
  ///         The data has shape (n_baselines, n_channels, n_correlations).
  ///         Note: In the future, this function will return a reference to the
  ///         XTensor object that holds the data.
  [[nodiscard]] const DataType& GetData(const std::string& name = "") const {
    if (name.empty()) {
      return data_;
    } else {
      auto found = extra_data_.find(name);
      if (found == extra_data_.end()) {
        throw std::runtime_error("No data named '" + name +
                                 "' is found in the current DPBuffer");
      }

      return found->second;
    }
  }
  [[nodiscard]] DataType& GetData(const std::string& name = "") {
    if (name.empty()) {
      return data_;
    } else {
      auto found = extra_data_.find(name);
      if (found == extra_data_.end()) {
        throw std::runtime_error("No data named '" + name +
                                 "' is found in the current DPBuffer");
      }
      return found->second;
    }
  }

  /// Returns the data and clears the storage in this object.
  ///
  /// Some Steps need to add elements and need to resize the shape of the
  /// storage. Resizing is "destructive". This function allows callers to
  /// "steal" the storage before resizing.
  [[nodiscard]] DataType TakeData() {
    DataType result;
    std::swap(result, data_);
    return result;
  }

  /// Copy a data buffer of another DPBuffer into an extra data buffer
  /// in the current DPBuffer. Overwrites the extra data buffer if it exists.
  /// @param source_name Name of a data buffer in 'source'. If empty, uses
  ///        the main data buffer in 'source'.
  /// @param target_name Name of the target data buffer.
  ///        May not be empty: Copying to the main data buffer is not supported.
  void CopyData(const DPBuffer& source, const std::string& source_name,
                const std::string& target_name);

  /// Move a data buffer from 'source' into an extra data buffer of the
  /// current DPBuffer. If the target buffer already exists, it is overwritten.
  /// @param source_name Name of a data buffer in 'source'. If empty, uses
  ///        the main data buffer in 'source'.
  /// @param target_name Name of the target data buffer.
  ///        May not be empty: Moving to the main data buffer is not supported.
  void MoveData(DPBuffer& source, const std::string& source_name,
                const std::string& target_name);

  /// Accesses the flags for the data (visibilities) in the DPBuffer.
  ///
  /// @return An XTensor view to the flags in the DPBuffer.
  ///         The view has shape (n_baselines, n_channels, n_correlations).
  ///         Note: In the future, this function will return a reference to the
  ///         XTensor object that holds the flags.
  [[nodiscard]] const FlagsType& GetFlags() const { return flags_; }
  [[nodiscard]] FlagsType& GetFlags() { return flags_; }

  /// Accesses weights for the data (visibilities) in the DPBuffer.
  ///
  /// @return An XTensor view to the weights in the DPBuffer.
  ///         The view has shape (n_baselines, n_channels, n_correlations).
  ///         Note: In the future, this function will return a reference to the
  ///         XTensor object that holds the weights.
  [[nodiscard]] const WeightsType& GetWeights() const { return weights_; }
  [[nodiscard]] WeightsType& GetWeights() { return weights_; }

  /// Returns the weights and clears the storage in this object.
  ///
  /// Some Steps need to add elements and need to resize the shape of the
  /// storage. Resizing is "destructive". This function allows callers to
  /// "steal" the storage before resizing.
  [[nodiscard]] WeightsType TakeWeights() {
    WeightsType result;
    std::swap(result, weights_);
    return result;
  }

  /// Get or set the time.
  void SetTime(double time) { time_ = time; }
  [[nodiscard]] double GetTime() const { return time_; }

  /// Get or set the exposure.
  void SetExposure(double exposure) { exposure_ = exposure; }
  [[nodiscard]] double GetExposure() const { return exposure_; }

  /// Get or set the row numbers used by the InputStep class.
  /// It can be empty (e.g. when MSReader inserted a dummy time slot).
  void SetRowNumbers(const casacore::Vector<common::rownr_t>& rownrs) {
    row_numbers_.reference(rownrs);
  }
  [[nodiscard]] const casacore::Vector<common::rownr_t>& GetRowNumbers() const {
    return row_numbers_;
  }

  /// Accesses the UVW coordinates in the DPBuffer.
  ///
  /// @return An XTensor object with the UVW coordinates in the DPBuffer.
  ///         The object has shape (n_baselines, 3).
  [[nodiscard]] const UvwType& GetUvw() const { return uvw_; }
  [[nodiscard]] UvwType& GetUvw() { return uvw_; }

  void SetSolution(const SolutionType& solution) { solution_ = solution; }
  [[nodiscard]] const SolutionType& GetSolution() const { return solution_; }

 private:
  double time_;
  double exposure_;
  casacore::Vector<common::rownr_t> row_numbers_;

  /// Visibilities (n_baselines x n_channels x n_correlations)
  DataType data_;
  /// Extra visibilities, e.g., containing predictions for different directions.
  std::map<std::string, DataType> extra_data_;

  /// Flags (n_baselines x n_channels x n_correlations)
  FlagsType flags_;
  /// Weights (n_baselines x n_channels x n_correlations)
  WeightsType weights_;
  /// UVW coordinates (n_baselines x 3)
  UvwType uvw_;

  SolutionType solution_;  ///< nchan,nant*npol
};

}  // namespace base
}  // namespace dp3

#endif  // DP3_BASE_DPBUFFER_H_
