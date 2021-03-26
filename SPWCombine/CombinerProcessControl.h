// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

/// @author Adriaan Renting

#ifndef LOFARCOMBINERPROCESSCONTROL_H
#define LOFARCOMBINERPROCESSCONTROL_H

#include <PLC/ProcessControl.h>
#include <Common/ParameterSet.h>
#include <vector>

namespace casacore {
class MeasurementSet;  // foreward declaration
}

namespace dp3 {
namespace spwcombine {

class SPWCombine;  // foreward declaration

class CombinerProcessControl : public DP3ACC::PLC::ProcessControl {
 private:
  vector<std::string> itsInMS;
  std::string itsOutMS;
  vector<casa::MeasurementSet*> inMS;
  SPWCombine* itsCombiner;

 public:
  CombinerProcessControl(void);

  ~CombinerProcessControl(void);
  /// \name Command to control the processes.
  /// There are a dozen commands that can be sent to a application process
  /// to control its flow. The return values for these command are:<br>
  /// - True   - Command executed succesfully.
  /// - False  - Command could not be executed.
  //
  /// @{

  /// During the \c define state the process check the contents of the
  /// ParameterSet it received during start-up. When everthing seems ok the
  /// process constructs the communication channels for exchanging data
  /// with the other processes. The connection are NOT made in the stage.
  tribool define(void);

  /// When a process receives an \c init command it allocates the buffers it
  /// needs an makes the connections with the other processes. When the
  /// process succeeds in this it is ready for dataprocessing (or whatever
  /// task the process has).
  tribool init(void);

  /// During the \c run phase the process does the work it is designed for.
  /// The run phase stays active until another command is send.
  tribool run(void);

  tribool pause(const std::string&);
  tribool release(void);
  tribool quit(void);
  tribool recover(const std::string&);
  tribool reinit(const std::string&);
  tribool snapshot(const std::string&);
  std::string askInfo(const std::string&);

};  // class CombinerProcessControl
}  // namespace spwcombine
};  // namespace dp3

#endif
