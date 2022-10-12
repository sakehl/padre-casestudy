// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef DP3_BASE_DP3_H_
#define DP3_BASE_DP3_H_

#include <dp3/steps/Step.h>

#include <map>

namespace dp3 {
namespace common {
class ParameterSet;
}  // namespace common
namespace steps {
class InputStep;
}  // namespace steps
namespace base {

/// @brief Class to run steps like averaging and flagging on an MS

/// This class contains a single static function that creates and executes
/// the steps defined in the parset file.
/// The parset file is documented on the LOFAR wiki.

class DP3 {
 public:
  /// Define the function to create a step from the given parameterset.
  typedef steps::Step::ShPtr StepCtor(steps::InputStep*,
                                      const common::ParameterSet&,
                                      const std::string& prefix);

  /// Add a function creating a Step to the map.
  static void registerStepCtor(const std::string&, StepCtor*);

  /// Create a step object from the given parameters.
  /// It looks up the step type in theirStepMap. If not found, it will
  /// try to load a shared library with that name and execute the
  /// register function in it.
  static StepCtor* findStepCtor(const std::string& type);

  /// Execute the steps defined in the parset file.
  /// Possible parameters given at the command line are taken into account.
  static void execute(const std::string& parsetName, int argc = 0,
                      char* argv[] = 0);

  /// Create a chain of step objects that are connected together.
  /// A writer will be added to the steps if it is not defined,
  /// and a terminating NullStep is added.
  static std::shared_ptr<steps::InputStep> makeMainSteps(
      const common::ParameterSet& parset);

  /// Create a chain of step objects that are connected together.
  /// Unlike makeMainSteps(), this does neither add a writer nor
  /// terminate the chain with a NullStep.
  /// @param step_names_key Parset key for getting the step names.
  /// @return The first step of the step chain or a null pointer if the chain is
  /// empty.
  static std::shared_ptr<steps::Step> makeStepsFromParset(
      const common::ParameterSet& parset, const std::string& prefix,
      const std::string& step_names_key, steps::InputStep& inputStep,
      bool terminateChain, steps::Step::MsType initial_step_output);

  /// Go through the steps to define which fields of the measurement set should
  /// be read by the input step.
  /// @param first_step The first step of a chain of steps.
  /// @return The combined required fields of the step chain.
  static dp3::common::Fields GetChainRequiredFields(
      std::shared_ptr<steps::Step> first_step);

  /// Go through a step chain, combine provided fields of non-output steps and
  /// call SetFieldsToWrite for all output steps.
  /// @param first_step The first step of a chain of steps.
  /// @param provided_fields The provided fields before the step chain.
  /// Steps that have sub-steps should use this argument.
  /// @return The remaining provided fields after the last step in the chain.
  static dp3::common::Fields SetChainProvidedFields(
      std::shared_ptr<steps::Step> first_step,
      dp3::common::Fields provided_fields = dp3::common::Fields());

 private:
  /// The map to create a step object from its type name.
  static std::map<std::string, StepCtor*> theirStepMap;
};

}  // namespace base
}  // namespace dp3

#endif  // DP3_BASE_DP3_H_