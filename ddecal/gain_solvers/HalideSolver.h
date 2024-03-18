#ifndef DDECAL_GAIN_SOLVERS_HALIDE_SOLVER_H
#define DDECAL_GAIN_SOLVERS_HALIDE_SOLVER_H

#include "SolverBase.h"
#include "SolveData.h"
#include "IterativeDiagonalSolver.h"

#include "HalideBuffer.h"

namespace dp3 {
namespace ddecal {

class IterativeDiagonalSolverHalide final : public SolverBase {
 public:
  friend class HalideTester;

  SolveResult Solve(const SolveData& data,
                    std::vector<std::vector<DComplex>>& solutions, double time,
                    std::ostream* stat_stream) override;

  size_t NSolutionPolarizations() const override { return 2; }

  bool SupportsDdSolutionIntervals() const override { return true; }

  int PerformIteration(
      size_t ch_block, const SolveData::ChannelBlockData& cb_data,
      Halide::Runtime::Buffer<float, 4>& v_res_result_b,
      Halide::Runtime::Buffer<double, 4>& solution_b,
      Halide::Runtime::Buffer<double, 5>& next_solutions_b,
      bool skip=false
      );

  template <bool Add>
  void AddOrSubtractDirection(size_t channel_block, size_t direction);

  // size_t channel_blocks() { return NChannelBlocks(); };
  // size_t directions() { return NDirections(); };
  // size_t solutions() { return NSolutions(); };
  // size_t antennas() { return NAntennas(); };

 private:
  void SetBuffers(const SolveData& data);

  struct Buffers {
    std::vector<std::vector<Halide::Runtime::Buffer<float, 4>>>
        model;      // <cb>[ndir][nvis], MC2x2F
    std::vector<Halide::Runtime::Buffer<uint32_t, 1>>
        antenna1;      // <cb>[nvis], uin32_t
    std::vector<Halide::Runtime::Buffer<uint32_t, 1>>
        antenna2;      // <cb>[nvis], uin32_t
    std::vector<std::vector<Halide::Runtime::Buffer<uint32_t, 1>>>
        solution_map;  // <cb>[n_dir][n_vis] uint32_t
  } buffers_;
};

class HalideTester{
  typedef std::complex<double> DComplex;
  typedef std::complex<float> Complex;
  
  IterativeDiagonalSolver& solver_check;
  IterativeDiagonalSolverHalide& solver;
  const SolveData& data;
  std::vector<std::vector<DComplex>>& solutions;
public:

  HalideTester(IterativeDiagonalSolver &solver_check, IterativeDiagonalSolverHalide &solver, const SolveData& data,
    std::vector<std::vector<DComplex>>& solutions) :
    solver_check(solver_check), solver(solver), data(data), solutions(solutions){
    
    solver.SetBuffers(data);
  }

  int IdTest();
  int AddTest();
  int NumeratorTest();
  int SubDirectionTest();
  int SolveDirectionTest();
  int PerformIterationTest();
  int PerformIterationAllBlocksTest();
  int MultipleIterationsTest();

  std::tuple<
    Halide::Runtime::Buffer<float, 4>,
    Halide::Runtime::Buffer<double, 4>
    > get_inp(int cb, std::vector<aocommon::MC2x2F> &v_residual, std::vector<aocommon::MC2x2F> &v_residual_check);
};

}  // namespace ddecal
}  // namespace dp3

#endif