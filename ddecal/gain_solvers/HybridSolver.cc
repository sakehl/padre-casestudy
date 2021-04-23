// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include "HybridSolver.h"

#include <cassert>

namespace dp3 {
namespace base {

SolverBase::SolveResult HybridSolver::Solve(
    const SolverBuffer& solver_buffer,
    std::vector<std::vector<DComplex>>& solutions, double time,
    std::ostream* stat_stream) {
  assert(!solvers_.empty());
  size_t available_iterations = GetMaxIterations();
  SolveResult result;
  bool is_converged = false;
  for (const std::pair<std::unique_ptr<SolverBase>, size_t>& solver_info :
       solvers_) {
    solver_info.first->SetMaxIterations(solver_info.second);
    is_converged = RunSolver(*solver_info.first, available_iterations, result,
                             solver_buffer, solutions, time, stat_stream);
    if (is_converged && stop_on_convergence_) return result;
  }
  if (!is_converged) result.iterations = GetMaxIterations() + 1;
  return result;
}

void HybridSolver::Initialize(size_t nAntennas, size_t nDirections,
                              size_t nChannels, size_t nChannelBlocks,
                              const std::vector<int>& ant1,
                              const std::vector<int>& ant2) {
  SolverBase::Initialize(nAntennas, nDirections, nChannels, nChannelBlocks,
                         ant1, ant2);
  for (const std::pair<std::unique_ptr<SolverBase>, size_t>& solver_info :
       solvers_) {
    solver_info.first->Initialize(nAntennas, nDirections, nChannels,
                                  nChannelBlocks, ant1, ant2);
  }
}

void HybridSolver::AddSolver(std::unique_ptr<SolverBase> solver) {
  if (!solvers_.empty()) {
    if (solver->NSolutionPolarizations() !=
        solvers_.front().first->NSolutionPolarizations())
      throw std::runtime_error(
          "Solvers with different nr of polarizations can't be combined in "
          "the hybrid solver");
  }
  size_t iter = solver->GetMaxIterations();
  solvers_.emplace_back(std::move(solver), iter);
}

bool HybridSolver::RunSolver(SolverBase& solver, size_t& available_iterations,
                             SolveResult& result,
                             const SolverBuffer& solver_buffer,
                             std::vector<std::vector<DComplex>>& solutions,
                             double time, std::ostream* stat_stream) {
  if (solver.GetMaxIterations() > available_iterations)
    solver.SetMaxIterations(available_iterations);
  SolveResult nextResult =
      solver.Solve(solver_buffer, solutions, time, stat_stream);
  result.iterations += nextResult.iterations;
  result.constraint_iterations += nextResult.constraint_iterations;
  result.results = std::move(nextResult.results);
  const bool is_converged = nextResult.iterations <= solver.GetMaxIterations();
  if (is_converged)
    available_iterations -= result.iterations;
  else  // If not converged, the solver has taken the maximum nr of
        // iterations.
    available_iterations -= solver.GetMaxIterations();
  return is_converged;
}

}  // namespace base
}  // namespace dp3
