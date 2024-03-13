#include "HalideSolver.h"
#include "../IdHalide.h"
#include "../SolveDirectionHalide.h"
#include "../SubDirectionHalide.h"
#include "../TestNumerator.h"
#include "IterativeDiagonalSolver.h"
#include <aocommon/matrix2x2.h>
#include <aocommon/matrix2x2diag.h>
#include <aocommon/recursivefor.h>
#include <ctime>

using aocommon::MC2x2F;
using aocommon::MC2x2FDiag;

namespace dp3 {
namespace ddecal {

IterativeDiagonalSolverHalide::SolveResult IterativeDiagonalSolverHalide::Solve(
    const SolveData& data, std::vector<std::vector<DComplex>>& solutions,
    double time, std::ostream* stat_stream) {
  PrepareConstraints();

  SolutionTensor next_solutions(
      {NChannelBlocks(), NAntennas(), NSolutions(), NSolutionPolarizations()});

  SolveResult result;

  // Visibility vector v_residual[cb][vis] of size NChannelBlocks() x
  // n_visibilities
  std::vector<std::vector<MC2x2F>> v_residual(NChannelBlocks());
  // The following loop allocates all structures
  for (size_t ch_block = 0; ch_block != NChannelBlocks(); ++ch_block) {
    v_residual[ch_block].resize(data.ChannelBlock(ch_block).NVisibilities());
  }

  ///
  /// Start iterating
  ///
  size_t iteration = 0;
  bool has_converged = false;
  bool has_previously_converged = false;
  bool constraints_satisfied = false;

  std::vector<double> step_magnitudes;
  step_magnitudes.reserve(GetMaxIterations());
  SetBuffers(data);
  aocommon::StaticFor<size_t> loop;
  do
  {
    MakeSolutionsFinite2Pol(solutions);

    loop.Run(0, NChannelBlocks(), [&](size_t start_block, size_t end_block) {
        for (size_t ch_block = start_block; ch_block != end_block;
              ++ch_block) {
          const SolveData::ChannelBlockData& channel_block =
              data.ChannelBlock(ch_block);
          PerformIteration(ch_block, channel_block,
                                v_residual[ch_block], solutions[ch_block],
                                next_solutions);
        }
      });

    Step(solutions, next_solutions);

    constraints_satisfied =
        ApplyConstraints(iteration, time, has_previously_converged, result,
                         next_solutions, stat_stream);

    double avg_squared_diff;
    has_converged =
        AssignSolutions(solutions, next_solutions, !constraints_satisfied,
                        avg_squared_diff, step_magnitudes);
    iteration++;

    has_previously_converged = has_converged || has_previously_converged;
  } while (!ReachedStoppingCriterion(iteration, has_converged,
                                     constraints_satisfied, step_magnitudes));

  if (has_converged && constraints_satisfied)
    result.iterations = iteration;
  else
    result.iterations = iteration + 1;
  return result;
}

int IterativeDiagonalSolverHalide::PerformIteration(
    size_t ch_block, const SolveData::ChannelBlockData& cb_data,
    std::vector<MC2x2F>& v_residual, const std::vector<DComplex>& solutions,
    SolutionTensor & next_solutions) {
  int result = 0;
  int nvis = cb_data.NVisibilities();
  int nsol = NSolutions();
  int n_ant = NAntennas();

  // Subtract all directions with their current solutions
  const DComplex* solution_data = solutions.data();
      Halide::Runtime::Buffer<double, 4> solution_b =
          Halide::Runtime::Buffer<double, 4>(
              (double*)solution_data, 2, 2,
              NSolutions(),
              NAntennas());
  solution_b.set_host_dirty();

  std::vector<MC2x2F> v_res_in(nvis);
  std::copy(cb_data.DataBegin(), cb_data.DataEnd(), v_res_in.begin());
  aocommon::MC2x2F* v_res_in_host = v_res_in.data();
  Halide::Runtime::Buffer<float, 4> v_res_in_b =
      Halide::Runtime::Buffer<float, 4>((float*)v_res_in_host, 2, 2, 2, nvis);
  v_res_in_b.set_host_dirty();

  aocommon::MC2x2F* v_res_result_host = v_residual.data();
  Halide::Runtime::Buffer<float, 4> v_res_result_b =
      Halide::Runtime::Buffer<float, 4>((float*)v_res_result_host, 2, 2, 2, nvis);

  int solution_index0 = 0;
  for (size_t direction = 0; direction != NDirections(); ++direction) {
    int n_sol_for_dir = cb_data.NSolutionsForDirection(direction);

    if(direction>0){
      // Copy the result of the previous iteration as input for the next
      v_res_in_b.copy_from(v_res_result_b);
    }

    result = SubDirection(
        solution_b,
        buffers_.solution_map[ch_block][direction], buffers_.antenna1[ch_block],
        buffers_.antenna2[ch_block], buffers_.model[ch_block][direction],
        v_res_in_b, solution_index0, n_sol_for_dir, nvis, nsol, n_ant, v_res_result_b);
    solution_index0 += n_sol_for_dir;
    if(result != 0){
      printf("SubDirection: ch_block: %lu dir: %lu\n", ch_block, direction);
      assert(result == 0);
    }
  }

  solution_index0 = 0;
  for (size_t direction = 0; direction != NDirections(); ++direction) {
    int n_sol_for_dir = cb_data.NSolutionsForDirection(direction);

    double* n_s_raw = ((double *) next_solutions.data()) + ch_block * n_ant * nsol * 2 * 2 + solution_index0*2*2;
    Halide::Runtime::Buffer<double, 4> next_solutions_b;
    next_solutions_b = Halide::Runtime::Buffer<double, 4>(n_s_raw, 
      {halide_dimension_t(0, 2, 1),
       halide_dimension_t(0, 2, 2),
       halide_dimension_t(solution_index0, n_sol_for_dir, 2*2),
       halide_dimension_t(0, n_ant, 2*2*nsol)});
    next_solutions_b.set_host_dirty();

    result = SolveDirection(
        solution_b,
        buffers_.solution_map[ch_block][direction], buffers_.antenna1[ch_block],
        buffers_.antenna2[ch_block], buffers_.model[ch_block][direction],
        v_res_result_b, solution_index0, n_sol_for_dir, nvis, nsol, n_ant, next_solutions_b);
    solution_index0 += n_sol_for_dir;

    next_solutions_b.copy_to_host();
    if(result != 0){
      printf("SolveDirection: ch_block: %lu dir: %lu\n", ch_block, direction);
      assert(result == 0);
    }
  }

  return result;
}


template<typename T>
bool nearly_equal(
  T a, T b,
  T epsilon = 1e-5, T abs_th = 1E-37)
  // those defaults are arbitrary and could be removed
{
  assert(std::numeric_limits<T>::epsilon() <= epsilon);
  assert(epsilon < 1.f);

  if (a == b) return true;
  auto diff = std::abs(a-b);
  auto norm = std::min((std::abs(a) + std::abs(b)), std::numeric_limits<T>::max());
  
  if (a == 0 || b == 0 || diff < abs_th) {
        // a or b is zero or both are extremely close to it
        // relative error is less meaningful here
        return diff < (epsilon * abs_th);
  } else {
    // use relative error
    return diff / norm < epsilon;
  }
}

template<typename T>
T relative(
  T a, T b)
  // those defaults are arbitrary and could be removed
{
  if (a == b) return 0.f;
  auto diff = std::abs(a-b);
  auto norm = std::min((std::abs(a) + std::abs(b)), std::numeric_limits<T>::max());
 
  return diff / norm;
}

template<typename T>
bool check_complex(std::complex<T> a, std::complex<T> b) {
  T eps = 1.0E-2;
  if (!nearly_equal<T>(a.real(), b.real(), eps) || !nearly_equal(a.imag(), b.imag(), eps)) {
    T eps = 1.0E-1;
    if (nearly_equal<T>(a.real(), b.real(), eps) && nearly_equal(a.imag(), b.imag(), eps)) {
      std::cout << "Warning: small diff (" << relative(a.real(), b.real()) << ", " << relative(a.imag(), b.imag()) 
        << ") in check: (" << a.real() <<", " << a.imag() << ") " << b
        << std::endl;
    } else {
      std::cout << "Warning: big diff in check: (" << a.real() <<", " << a.imag() << ") " << b
                << std::endl;
      return false;
    }
  }
  return true;
}

template<typename T>
bool check_complex(T a_real, T a_imag, std::complex<T> b) {
  return check_complex(std::complex(a_real, a_imag), b);
}

bool check_float(float a, float b, float epsilon = 1e-5) {
  if (!nearly_equal(a,b, epsilon)) {
    std::cout << "Difference between halide and check: " << a << " " << b
              << std::endl;
    return false;
  }
  return true;
}

bool check_matrix(int nvis, int vis, Halide::Runtime::Buffer<float, 4> v_res_halide, aocommon::MC2x2F mat){
  for(int j = 0; j < 2; j++){
      for(int k = 0; k < 2; k++){
        if(!check_complex(v_res_halide(0, k, j, vis), v_res_halide(1, k, j, vis), mat[j*2+k])){
          std::cout << "Halide: " << v_res_halide(0, 0, 0, vis) << "+" << v_res_halide(1, 0, 0, vis) << "i,  ";
          std::cout << v_res_halide(0, 1, 0, vis) << "+" << v_res_halide(1, 1, 0, vis) << "i, ";
          std::cout << v_res_halide(0, 0, 1, vis) << "+" << v_res_halide(1, 0, 1, vis) << "i, ";
          std::cout << v_res_halide(0, 1, 1, vis) << "+" << v_res_halide(1, 1, 1, vis) << "i " << std::endl;
          std::cout << "Check: " << mat[0] << ",  " << mat[1] << ",  " << mat[2] << ",  " << mat[3] << ",  " << std::endl;

          std::cout << "For visibility " << vis << " (" << k << "x" << j << ")" << std::endl;
          return false;
        }
      }
    }
  return true;
}



bool check_solution_next(SolutionTensor &solutions,
  SolutionTensor &solutions_check){
    assert(solutions.size() == solutions_check.size());
    for(size_t i=0; i<solutions.size(); i++){
      if(!check_complex(solutions[i], solutions_check[i])){
          std::cout << "For i solutions: " << i << std::endl;
        return false;
      }
    }
    return true;
}

bool check_solution_old(std::vector<std::vector<dp3::ddecal::SolverBase::DComplex>> &solutions,
  std::vector<std::vector<dp3::ddecal::SolverBase::DComplex>> &solutions_check){
    assert(solutions.size() == solutions_check.size());
    for(size_t i=0; i<solutions.size(); i++){
      assert(solutions[i].size() == solutions_check[i].size());
      for(size_t j=0; j<solutions[i].size(); j++){
        if(!check_complex(solutions[i][j], solutions_check[i][j])){
           std::cout << "For i, j of solutions: " << i << ", " << j << std::endl;
          return false;
        }
      }
    }
    return true;
}


bool check_solution(int n_ant, int solution_offset, int nr_solutions_dir, int cb,
    SolutionTensor next_solutions, SolutionTensor next_solutions_check){
  for (int i = 0; i < n_ant; i++){
    for(int si = 0; si < nr_solutions_dir; si++){
      for(int pol = 0; pol < 2; pol++){
        if(!check_complex(next_solutions(cb, i, solution_offset+si, pol), next_solutions_check(cb, i, solution_offset+si, pol))) {
          std::cout << "For cb,antenna,sol,pol: " << cb << ", " << i << ", " << si << ", " << pol << std::endl;
          return false;
        }
      }
    }
  }
  return true;
}

void IterativeDiagonalSolverHalide::SetBuffers(const SolveData& data) {
  buffers_.solution_map =
      std::vector<std::vector<Halide::Runtime::Buffer<uint32_t, 1>>>();

  for (size_t ch_block = 0; ch_block < NChannelBlocks(); ch_block++) {
    const SolveData::ChannelBlockData& channel_block_data =
        data.ChannelBlock(ch_block);
    const size_t n_directions = channel_block_data.NDirections();
    const size_t n_visibilities = channel_block_data.NVisibilities();
    std::vector<Halide::Runtime::Buffer<float, 4>> models;
    std::vector<Halide::Runtime::Buffer<uint32_t, 1>> solution_maps;

    // Antennas
    uint32_t* antenna_1 = new uint32_t[n_visibilities];
    uint32_t* antenna_2 = new uint32_t[n_visibilities];
    Halide::Runtime::Buffer<uint32_t, 1> antenna_1_b =
        Halide::Runtime::Buffer<uint32_t, 1>(antenna_1, n_visibilities);
    Halide::Runtime::Buffer<uint32_t, 1> antenna_2_b =
        Halide::Runtime::Buffer<uint32_t, 1>(antenna_2, n_visibilities);
    buffers_.antenna1.emplace_back(antenna_1_b);
    buffers_.antenna2.emplace_back(antenna_2_b);
    for (size_t visibility_index = 0; visibility_index < n_visibilities;
         visibility_index++) {
      antenna_1[visibility_index] =
          channel_block_data.Antenna1Index(visibility_index);
      antenna_2[visibility_index] =
          channel_block_data.Antenna2Index(visibility_index);
    }

    for (size_t direction = 0; direction < n_directions; direction++) {
      // Model
      const aocommon::MC2x2F* model =
          &channel_block_data.ModelVisibility(direction, 0);
      Halide::Runtime::Buffer<float, 4> model_b =
          Halide::Runtime::Buffer<float, 4>((float*)model, 2, 2, 2,
                                            n_visibilities);
      model_b.set_host_dirty();
      models.emplace_back(model_b);

      // Solution Map
      const uint32_t* solution_map =
          // &channel_block_data.SolutionIndex(direction, 0);
          channel_block_data.SolutionMapData() + direction * n_visibilities;
      Halide::Runtime::Buffer<uint32_t, 1> solution_map_b =
          Halide::Runtime::Buffer<uint32_t, 1>((uint32_t*)solution_map,
                                               n_visibilities);
      solution_map_b.set_host_dirty();
      solution_maps.emplace_back(solution_map_b);
    }
    buffers_.model.emplace_back(models);
    buffers_.solution_map.emplace_back(solution_maps);
  }
}

// Testing

bool check_results(int nvis, Halide::Runtime::Buffer<float, 4> v_res_halide, 
  std::vector<aocommon::MC2x2F>& v_residual_check){
  for (int i = 0; i < nvis; i++) {
    if(!check_matrix(nvis, i, v_res_halide, v_residual_check[i])) return false;
  }
  return true;
}

int HalideTester::IdTest(){
  std::cout << "Testing Id" << std::endl;
  int cb = 0;
  int dir = 0;
  
  auto cb_data = data.ChannelBlock(cb);
  int nvis = cb_data.NVisibilities();
  int result = 0;
  int solution_index0 = 0;
  int nsol = solver.NSolutions();
  int n_ant = solver.NAntennas();
  int n_sol_for_dir = cb_data.NSolutionsForDirection(dir);

  // Get inp
  std::vector<aocommon::MC2x2F> v_residual, v_residual_check;
  Halide::Runtime::Buffer<float, 4> v_res_in;
  Halide::Runtime::Buffer<double, 4> solution_b;
  std::tie(v_res_in, solution_b) = get_inp(cb, v_residual, v_residual_check);
  Halide::Runtime::Buffer<float, 4> v_res_in_result = Halide::Runtime::Buffer<float, 4>(2, 2, 2, nvis);

  const clock_t begin_time_1 = clock();
  // Checking if the input, after putting everything through Halide is still the same
  v_res_in_result.set_host_dirty();

  result = IdHalide(solution_b,
                        solver.buffers_.solution_map[cb][dir], solver.buffers_.antenna1[cb],
                        solver.buffers_.antenna2[cb], solver.buffers_.model[cb][dir],
                        v_res_in, solution_index0, n_sol_for_dir, nvis, nsol, n_ant, v_res_in_result);
  if(result != 0){
    std::cout << "Halide execution error" << std::endl;
    return result;
  }
  v_res_in_result.copy_to_host();
  std::cout << "Halide impl: " << float( clock () - begin_time_1 ) /  CLOCKS_PER_SEC << std::endl;

  if(!check_results(nvis, v_res_in_result, v_residual_check)){
    return 1;
  }
  return result;
}


int HalideTester::AddTest(){ return 0; }

int HalideTester::NumeratorTest(){
  std::cout << "Testing Numerator" << std::endl;
  int cb = 0;
  int dir = 0;
  
  auto cb_data = data.ChannelBlock(cb);
  int nvis = cb_data.NVisibilities();
  int result = 0;
  const uint32_t solution_index0 = cb_data.SolutionIndex(dir, 0);
  int nsol = solver.NSolutions();
  int n_ant = solver.NAntennas();
  const uint32_t n_dir_solutions = cb_data.NSolutionsForDirection(dir);

  // Get inp
  std::vector<aocommon::MC2x2F> v_residual, v_residual_check;
  Halide::Runtime::Buffer<float, 4> v_res_in;
  Halide::Runtime::Buffer<double, 4> solution_b;
  std::tie(v_res_in, solution_b) = get_inp(cb, v_residual, v_residual_check);

  Halide::Runtime::Buffer<float, 4> numerator_buf = Halide::Runtime::Buffer<float, 4>(2, 2, n_dir_solutions, n_ant);
  Halide::Runtime::Buffer<float, 3> denominator_buf = Halide::Runtime::Buffer<float, 3>(2, n_dir_solutions, n_ant);

  const clock_t begin_time_1 = clock();
    
  result = TestNumerator(solution_b,
                        solver.buffers_.solution_map[cb][dir], solver.buffers_.antenna1[cb],
                        solver.buffers_.antenna2[cb], solver.buffers_.model[cb][dir],
                        v_res_in, solution_index0, n_dir_solutions, nvis, nsol, n_ant
                        , numerator_buf);
  if(result != 0){
    std::cout << "Halide execution error" << std::endl;
    return result;
  }

  std::cout << "Halide impl: " << float( clock () - begin_time_1 ) /  CLOCKS_PER_SEC << std::endl;

  std::vector<MC2x2FDiag> numerator(n_ant * n_dir_solutions,
                                    MC2x2FDiag::Zero());
  std::vector<float> denominator(n_ant * n_dir_solutions * 2, 0.0);

  const clock_t begin_time_2 = clock();
  // Iterate over all data
  const size_t n_visibilities = cb_data.NVisibilities();
  for (size_t vis_index = 0; vis_index != n_visibilities; ++vis_index) {
    const uint32_t antenna_1 = cb_data.Antenna1Index(vis_index);
    const uint32_t antenna_2 = cb_data.Antenna2Index(vis_index);
    const uint32_t solution_index = cb_data.SolutionIndex(dir, vis_index);
    const DComplex* solution_ant_1 =
        &solutions[cb][(antenna_1 * nsol + solution_index) * 2];
    const DComplex* solution_ant_2 =
        &solutions[cb][(antenna_2 * nsol + solution_index) * 2];
    const MC2x2F& data = v_residual[vis_index];
    const MC2x2F& model = cb_data.ModelVisibility(dir, vis_index);

    const uint32_t rel_solution_index = solution_index - solution_index0;
    // Calculate the contribution of this baseline for antenna_1
    const MC2x2FDiag solution_1{Complex(solution_ant_2[0]),
                                Complex(solution_ant_2[1])};
    const MC2x2F cor_model_transp_1(solution_1 * HermTranspose(model));
    const uint32_t full_solution_1_index =
        antenna_1 * n_dir_solutions + rel_solution_index;
    // numerator[full_solution_1_index] += Diagonal(data * cor_model_transp_1);
    numerator[full_solution_1_index] += Diagonal(data * cor_model_transp_1);
    // The indices (0, 2 / 1, 3) are following from the fact that we want
    // the contribution of antenna2's "X" polarization, and the matrix is
    // ordered [ XX XY / YX YY ].
    denominator[full_solution_1_index * 2] +=
        std::norm(cor_model_transp_1[0]) + std::norm(cor_model_transp_1[2]);
    denominator[full_solution_1_index * 2 + 1] +=
        std::norm(cor_model_transp_1[1]) + std::norm(cor_model_transp_1[3]);

    // Calculate the contribution of this baseline for antenna_2
    // data_ba = data_ab^H, etc., therefore, numerator and denominator
    // become:
    // - num = data_ab^H * solutions_a * model_ab
    // - den = norm(model_ab^H * solutions_a)
    const MC2x2FDiag solution_2{Complex(solution_ant_1[0]),
                                Complex(solution_ant_1[1])};
    const MC2x2F cor_model_2(solution_2 * model);

    const uint32_t full_solution_2_index =
        antenna_2 * n_dir_solutions + rel_solution_index;
    numerator[full_solution_2_index] +=
        Diagonal(HermTranspose(data) * cor_model_2);
    denominator[full_solution_2_index * 2] +=
        std::norm(cor_model_2[0]) + std::norm(cor_model_2[2]);
    denominator[full_solution_2_index * 2 + 1] +=
        std::norm(cor_model_2[1]) + std::norm(cor_model_2[3]);

  }

  std::cout << "Std impl: " << float( clock () - begin_time_2 ) /  CLOCKS_PER_SEC << std::endl;

  for(int a=0; a<(int)n_ant; a++){
    for(int s=0; s<(int)n_dir_solutions; s++){
      for(int d=0; d<2; d++){
        if(!check_complex(numerator_buf(0,d,s,a), numerator_buf(1,d,s,a), numerator[a*n_dir_solutions+s][d])){
          std::cout << "Ant " << a << " Dir: " << dir << " Sol: " << s << " Diagonal: " << d << std::endl;
          return 1;
        }
      }
    }
  }

  return result; 
}

int HalideTester::SubDirectionTest(){
  std::cout << "Testing SubDirection" << std::endl;

  int cb = 0;
  int dir = 0;
  auto cb_data = data.ChannelBlock(cb);
  int nvis = cb_data.NVisibilities();
  int result = 0;
  const uint32_t solution_index0 = cb_data.SolutionIndex(dir, 0);
  const uint32_t n_dir_solutions = cb_data.NSolutionsForDirection(dir);

  // Get inp
  std::vector<aocommon::MC2x2F> v_residual, v_residual_check;
  Halide::Runtime::Buffer<float, 4> v_res_in;
  Halide::Runtime::Buffer<float, 4> v_res_in_result = Halide::Runtime::Buffer<float, 4>(2, 2, 2, nvis);
  Halide::Runtime::Buffer<double, 4> solution_b;
  std::tie(v_res_in, solution_b) = get_inp(cb, v_residual, v_residual_check);

  const clock_t begin_time_1 = clock();
  result = SubDirection(
      solution_b, solver.buffers_.solution_map[cb][dir],
      solver.buffers_.antenna1[cb], solver.buffers_.antenna2[cb], solver.buffers_.model[cb][dir],
      v_res_in, solution_index0, n_dir_solutions, nvis, solver.NSolutions(), solver.NAntennas(), v_res_in_result);
  if(result != 0){
    std::cout << "Halide execution error" << std::endl;
    return result;
  }
  v_res_in_result.copy_to_host();
  std::cout << "Halide impl: " << float( clock () - begin_time_1 ) /  CLOCKS_PER_SEC << std::endl;

  const clock_t begin_time_2 = clock();
  solver_check.AddOrSubtractDirection<false>(cb_data, v_residual_check,
                                              dir, solutions[cb]);
  std::cout << "Std impl: " << float( clock () - begin_time_2 ) /  CLOCKS_PER_SEC << std::endl;

  check_results(nvis, v_res_in_result, v_residual_check);

  return result;
}
int HalideTester::SolveDirectionTest(){
  std::cout << "Testing SolveDirection" << std::endl;

  int cb = 0;
  int dir = 1;
  auto cb_data = data.ChannelBlock(cb);
  int nvis = cb_data.NVisibilities();
  int result = 0;
  const uint32_t solution_index0 = cb_data.SolutionIndex(dir, 0);
  const uint32_t n_dir_solutions = cb_data.NSolutionsForDirection(dir);

  // Get inp
  std::vector<aocommon::MC2x2F> v_residual, v_residual_check;
  Halide::Runtime::Buffer<float, 4> v_res_in;
  Halide::Runtime::Buffer<double, 4> solution_b;
  std::tie(v_res_in, solution_b) = get_inp(cb, v_residual, v_residual_check);

  SolutionTensor next_solutions(
    {data.NChannelBlocks(), solver.NAntennas(), solver.NSolutions(), solver.NSolutionPolarizations()});
  SolutionTensor next_solutions_check(
    {data.NChannelBlocks(), solver.NAntennas(), solver.NSolutions(), solver.NSolutionPolarizations()});

  // double* n_s_raw = (double *) next_solutions.data();
  double* n_s_raw = ((double *) next_solutions.data()) + cb * solver.NAntennas() * solver.NSolutions() * 2 * 2
    + solution_index0*2*2;
  // Halide::Runtime::Buffer<double, 4> next_solutions_b =
  //   Halide::Runtime::Buffer<double, 4>(n_s_raw, 2, 2, solver.NSolutions(), solver.NAntennas());
  Halide::Runtime::Buffer<double, 4> next_solutions_b = Halide::Runtime::Buffer<double, 4>(n_s_raw, 
      {halide_dimension_t(0, 2, 1),
       halide_dimension_t(0, 2, 2),
      //  halide_dimension_t(solution_index0, n_dir_solutions, 2*2),
       halide_dimension_t(solution_index0, n_dir_solutions, 2*2),
       halide_dimension_t(0, solver.NAntennas(), 2*2*solver.NSolutions())});
  
  std::vector<std::vector<DComplex>> solutions_check = solutions;
  
  
  const clock_t begin_time_1 = clock();
  result = SolveDirection(
      solution_b, solver.buffers_.solution_map[cb][dir],
      solver.buffers_.antenna1[cb], solver.buffers_.antenna2[cb], solver.buffers_.model[cb][dir],
      v_res_in, solution_index0, n_dir_solutions, nvis, solver.NSolutions(), solver.NAntennas(), next_solutions_b);
  if(result != 0){
    std::cout << "Halide execution error" << std::endl;
    return result;
  }
  std::cout << "Halide impl: " << float( clock () - begin_time_1 ) /  CLOCKS_PER_SEC << std::endl;

  const clock_t begin_time_2 = clock();
  solver_check.AddOrSubtractDirection<true>(cb_data, v_residual_check,
                                              dir, solutions_check[cb]);
  solver_check.SolveDirection(cb, cb_data, v_residual_check, dir,
                                solutions_check[cb], next_solutions_check);
  std::cout << "Std impl: " << float( clock () - begin_time_2 ) /  CLOCKS_PER_SEC << std::endl;

  if (!check_solution(solver.NAntennas(), solution_index0, n_dir_solutions, cb, next_solutions, next_solutions_check)) {
    return 1;
  }
  return result;
}

int HalideTester::PerformIterationTest(){
  std::cout << "Testing PerformIteration" << std::endl;

  int cb = 0;
  auto cb_data = data.ChannelBlock(cb);
  int result = 0;

  // Get inp
  std::vector<aocommon::MC2x2F> v_residual, v_residual_check;
  Halide::Runtime::Buffer<float, 4> v_res_in;
  Halide::Runtime::Buffer<double, 4> solution_b;
  std::tie(v_res_in, solution_b) = get_inp(cb, v_residual, v_residual_check);

  SolutionTensor next_solutions(
    {data.NChannelBlocks(), solver.NAntennas(), solver.NSolutions(), solver.NSolutionPolarizations()});
  SolutionTensor next_solutions_check(
    {data.NChannelBlocks(), solver.NAntennas(), solver.NSolutions(), solver.NSolutionPolarizations()});
  std::vector<std::vector<DComplex>> solutions_check = solutions;

  const clock_t begin_time_1 = clock();
  result = solver.PerformIteration(cb, cb_data, v_residual, solutions[cb], next_solutions);
  if(result != 0){
    std::cout << "Halide execution error" << std::endl;
    return result;
  }
  std::cout << "Halide impl: " << float( clock () - begin_time_1 ) /  CLOCKS_PER_SEC << std::endl;

  const clock_t begin_time_2 = clock();
  solver_check.PerformIteration(cb, cb_data, v_residual_check,
                                  solutions_check[cb], next_solutions_check);
  std::cout << "Std impl: " << float( clock () - begin_time_2 ) /  CLOCKS_PER_SEC << std::endl;

  int solution_offset = 0;
  for (size_t direction = 0; direction != solver.NDirections(); ++direction) {
    int n_dir_solutions = cb_data.NSolutionsForDirection(direction);
    std::cout << "Testing dir " << direction << std::endl;
    if (!check_solution(solver.NAntennas(), solution_offset, n_dir_solutions, cb, next_solutions, next_solutions_check)) {
      std::cout << "(test PerformIteration, dir "<< direction << ")" << std::endl;
      return 1;
    }
    solution_offset += n_dir_solutions;
  }
  return result;
}
int HalideTester::PerformIterationAllBlocksTest(){
  std::cout << "Testing PerformIterationAllBlocks" << std::endl;

  int result = 0;

  // Get inp
  std::vector<std::vector<aocommon::MC2x2F>> v_residual, v_residual_check;
  std::vector<Halide::Runtime::Buffer<float, 4>> v_res_in;
  std::vector<Halide::Runtime::Buffer<double, 4>> solution_b;

  for(size_t cb=0; cb<solver.NChannelBlocks(); cb++){
    std::vector<aocommon::MC2x2F> v_residual_cb, v_residual_cb_check;
    auto res = get_inp(cb, v_residual_cb, v_residual_cb_check);
    v_residual.emplace_back(v_residual_cb);
    v_residual_check.emplace_back(v_residual_cb_check);
    v_res_in.emplace_back(std::get<0>(res));
    solution_b.emplace_back(std::get<1>(res));
  }
  std::vector<std::vector<DComplex>> solutions_check = solutions;
  

  SolutionTensor next_solutions(
    {solver.NChannelBlocks(), solver.NAntennas(), solver.NSolutions(), solver.NSolutionPolarizations()});
  SolutionTensor next_solutions_check(
    {solver.NChannelBlocks(), solver.NAntennas(), solver.NSolutions(), solver.NSolutionPolarizations()});


  const clock_t begin_time_1 = clock();
  for (size_t cb = 0; cb < data.NChannelBlocks(); cb++) {
    auto cb_data = data.ChannelBlock(cb);
    result = solver.PerformIteration(cb, cb_data, v_residual[cb], solutions[cb], next_solutions);
    if(result != 0){
      std::cout << "Halide execution error" << std::endl;
      return result;
    }
  }
  std::cout << "Halide impl: " << float( clock () - begin_time_1 ) /  CLOCKS_PER_SEC << std::endl;

  const clock_t begin_time_2 = clock();
  for (size_t cb = 0; cb < data.NChannelBlocks(); cb++) {
    auto cb_data = data.ChannelBlock(cb);
    solver_check.PerformIteration(cb, cb_data, v_residual_check[cb],
                                    solutions_check[cb], next_solutions_check);
  }
  std::cout << "Std impl: " << float( clock () - begin_time_2 ) /  CLOCKS_PER_SEC << std::endl;
  
  for (size_t cb = 0; cb < data.NChannelBlocks(); cb++) {
    int solution_offset = 0;
    for (size_t direction = 0; direction != solver.NDirections(); ++direction) {
      auto dir = direction;
      auto cb_data = data.ChannelBlock(cb);
      auto n_sol_for_dir = cb_data.NSolutionsForDirection(direction);
      if (!check_solution(solver.NAntennas(), solution_offset, n_sol_for_dir, cb, next_solutions, next_solutions_check)) {
        std::cout << "(test PerformIterationAllBlocksTest, dir "<< dir << ")" << std::endl;
        return 1;
      }
      solution_offset += n_sol_for_dir;
    }
  }

  return result;
}
int HalideTester::MultipleIterationsTest(){
  std::cout << "Testing MultipleIterationsTest" << std::endl;

  int check = 0;

  // Get inp
  std::vector<std::vector<aocommon::MC2x2F>> v_residual, v_residual_check;
  std::vector<Halide::Runtime::Buffer<float, 4>> v_res_in;
  std::vector<Halide::Runtime::Buffer<double, 4>> solution_b;

  for(size_t cb=0; cb<solver.NChannelBlocks(); cb++){
    std::vector<aocommon::MC2x2F> v_residual_cb, v_residual_cb_check;
    auto res = get_inp(cb, v_residual_cb, v_residual_cb_check);
    v_residual.emplace_back(v_residual_cb);
    v_residual_check.emplace_back(v_residual_cb_check);
    v_res_in.emplace_back(std::get<0>(res));
    solution_b.emplace_back(std::get<1>(res));
  }
  std::vector<std::vector<DComplex>> solutions_check = solutions;
  
  solver.PrepareConstraints();
  SolutionTensor next_solutions(
    {solver.NChannelBlocks(), solver.NAntennas(), solver.NSolutions(), solver.NSolutionPolarizations()});
  SolutionTensor next_solutions_check(
    {solver.NChannelBlocks(), solver.NAntennas(), solver.NSolutions(), solver.NSolutionPolarizations()});

  SolverBase::SolveResult result;
  SolverBase::SolveResult result_check;
  size_t iteration = 0;
  double time = 0.0f;
  bool has_converged = false;
  bool has_previously_converged = false;
  bool constraints_satisfied = false;
  std::ostream* stat_stream = nullptr;

  double time_check = 0.0f;
  bool has_converged_check = false;
  bool has_previously_converged_check = false;
  bool constraints_satisfied_check = false;
  std::vector<double> step_magnitudes;
  std::vector<double> step_magnitudes_check;
  step_magnitudes.reserve(solver.GetMaxIterations());
  step_magnitudes_check.reserve(solver.GetMaxIterations());
    
  for(int iter = 0; iter < 10; iter++){
    solver.MakeSolutionsFinite2Pol(solutions);
    solver_check.MakeSolutionsFinite2Pol(solutions_check);

    for (size_t cb = 0; cb < data.NChannelBlocks(); cb++) {
      auto cb_data = data.ChannelBlock(cb);
      check = solver.PerformIteration(cb, cb_data, v_residual[cb], solutions[cb], next_solutions);
      assert(check == 0);
      solver_check.PerformIteration(cb, cb_data, v_residual_check[cb],
                                    solutions_check[cb], next_solutions_check);
    }

    for (size_t cb = 0; cb < data.NChannelBlocks(); cb++) {
      int solution_offset = 0;
      for (size_t direction = 0; direction != solver.NDirections(); ++direction) {
        auto n_sol_for_dir = data.ChannelBlock(cb).NSolutionsForDirection(direction);
        if (!check_solution(solver.NAntennas(), solution_offset, n_sol_for_dir, cb, next_solutions, next_solutions_check)) {
          return 1;
        }
        solution_offset += n_sol_for_dir;
      }
    }
    solver.Step(solutions, next_solutions);
    solver.Step(solutions_check, next_solutions_check);

    constraints_satisfied =
      solver.ApplyConstraints(iteration, time, has_previously_converged, result,
                        next_solutions, stat_stream);
    constraints_satisfied_check =
      solver.ApplyConstraints(iteration, time_check, has_previously_converged_check, result_check,
                        next_solutions_check, stat_stream);
    assert(constraints_satisfied == constraints_satisfied_check);
    double avg_squared_diff;
    double avg_squared_diff_check;
    has_converged =
      solver.AssignSolutions(solutions, next_solutions, !constraints_satisfied,
                      avg_squared_diff, step_magnitudes);
    has_converged_check =
      solver.AssignSolutions(solutions_check, next_solutions_check, !constraints_satisfied_check,
                      avg_squared_diff_check, step_magnitudes_check);
    std::cout << "Iteration " << iteration << " Avg squared diff: " << avg_squared_diff << " Check: " << avg_squared_diff_check << std::endl;
    
    assert(has_converged == has_converged_check);
    iteration++;
    has_previously_converged = has_converged || has_previously_converged;
    has_previously_converged_check = has_converged_check || has_previously_converged_check;
    assert(check_solution_old(solutions, solutions_check));
  }

  return check;
}

std::tuple<
    Halide::Runtime::Buffer<float, 4>,
    Halide::Runtime::Buffer<double, 4>
    > HalideTester::get_inp(int cb, std::vector<aocommon::MC2x2F> &v_residual, std::vector<aocommon::MC2x2F> &v_residual_check){
  auto cb_data = data.ChannelBlock(cb);
  int nvis = cb_data.NVisibilities();

  v_residual.resize(nvis);
  v_residual_check.resize(nvis);
  std::copy(cb_data.DataBegin(), cb_data.DataEnd(), v_residual.begin());
  std::copy(cb_data.DataBegin(), cb_data.DataEnd(), v_residual_check.begin());
  aocommon::MC2x2F* v_res_in_host = v_residual.data();
  Halide::Runtime::Buffer<float, 4> v_res_in =
      Halide::Runtime::Buffer<float, 4>((float*)v_res_in_host, 2, 2, 2, nvis);

  v_res_in.set_host_dirty();

  // std::vector<std::vector<DComplex>> solutions_copy = solutions;
  const DComplex* solution_data = solutions[cb].data();
    Halide::Runtime::Buffer<double, 4> solution_b =
        Halide::Runtime::Buffer<double, 4>(
            (double*)solution_data, 2, 2,
            solver.NSolutions(),
            solver.NAntennas());

  solution_b.set_host_dirty();

  return std::tie(v_res_in, solution_b);
}


}  // namespace ddecal
}  // namespace dp3