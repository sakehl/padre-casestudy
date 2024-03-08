#include "Halide.h"
#include "HalideComplex.h"

// using dp3::ddecal;
using namespace Halide;


class HalideDiagionalSolver{
public:
    // Inputs
    ImageParam ant1;
    ImageParam ant2;
    ImageParam solution_map;
    ImageParam v_res_in;
    ImageParam model_;
    ImageParam sol_;
    Param<int> solution_index0;
    Param<int> n_dir_sol;
    Param<int> n_solutions;
    Param<int> n_vis;
    Param<int> n_antennas;
    
    Func model, sol;
    Var x, y, i, j, vis, si, a, pol, c;

    Func antenna_1, antenna_2, solution_index, v_res0;
    Func sol_ann;
    std::vector<Argument> args;

    HalideDiagionalSolver() :
        ant1(type_of<uint32_t>(), 1, "ant1"), // <1>[n_ant] uint32_t
        ant2(type_of<uint32_t>(), 1, "ant2"), // <1>[n_ant] uint32_t
        solution_map(type_of<uint32_t>(), 1, "solution_map"), // <1>[n_vis] uint32_t
        v_res_in(type_of<float>(), 4, "v_res_in"), // <4>[n_vis], Complex 2x2 Float (+3)
        model_(type_of<float>(), 4, "model_"), // <4>[n_vis], Complex 2x2 Float (+3)
        sol_(type_of<double>(), 4, "sol_"), // <4> [n_ant][n_dir_sol][2] Complex Double (+1)
        solution_index0("solution_index0"),
        n_dir_sol("n_dir_sol"), 
        n_solutions("n_solutions"),
        n_vis("n_vis"),
        n_antennas("n_antennas"),
        model("model"), sol("sol"),
        x("x"), y("y"), i("i"), j("j"), vis("vis"), si("si"), a("a"), pol("pol"), c("c"),
        antenna_1("antenna_1"), antenna_2("antenna_2"), solution_index("solution_index"),
        v_res0("v_res0"), sol_ann("sol_ann"){
        
        model(vis) = toComplexMatrix(model_, {vis});
        sol(i, si, a) = Tuple(sol_(0, i, si, a), sol_(1, i, si, a));

        antenna_1(vis) = unsafe_promise_clamped(cast<int>(ant1(vis)), 0, n_antennas-1);
        antenna_2(vis) = unsafe_promise_clamped(cast<int>(ant2(vis)), 0, n_antennas-1);
        solution_index(vis) = unsafe_promise_clamped(cast<int>(solution_map(vis)), 0, n_dir_sol-1);

        sol_ann(i,vis,a) = castc<float>(sol(i,solution_index(vis),a));

        v_res0(vis) = toComplexMatrix(v_res_in, {vis});

        args = {sol_, solution_map, ant1, ant2, model_, v_res_in, solution_index0, n_dir_sol, n_vis, n_solutions, n_antennas};
    }

    Func toComplex(Func f){
        std::vector<Var> args(f.args().begin()+1, f.args().end());
        std::vector<Expr> args_real(f.args().begin()+1, f.args().end());
        args_real.insert(args_real.begin(), 0);

        std::vector<Expr> args_imag(f.args().begin()+1, f.args().end());
        args_imag.insert(args_imag.begin(), 1);

        Func out("toComplex");
        out(args) = Complex(f(args_real), f(args_imag));
        return out;
    }

    Func fromComplex(Func f){
        std::vector<Var> args = f.args();
        args.insert(args.begin(), i);
        
        Complex c = Complex(f(f.args()));

        Func out("fromComplex");
        out(args) = mux(i, c);
        return out;
    }
    
    template<typename T>
    std::vector<T> concat(std::vector<T> a, std::vector<T> b){
        std::vector<T> out;
        out.reserve(a.size() + b.size());
        out.insert(out.end(), a.begin(), a.end());
        out.insert(out.end(), b.begin(), b.end());
        return out;
    }
    
    // Makes a function for which 2 dimensions are matrix dimensions
    // towards a function of Matrix type.
    // E.g. f(i,j, ..args) 
    // -> f(..args) = Matrix(f(0,0, ..args), f(1,0, ..args),
    //                       f(1,0, ..args), f(1,1, ..args))
    Matrix toMatrix(Func f, std::vector<Expr> args){
        Complex c1, c2, c3, c4;
        c1 = f(concat({0,0}, args));
        c2 = f(concat({1,0}, args));
        c3 = f(concat({0,1}, args));
        c4 = f(concat({1,1}, args));

        return Matrix(c1, c2, c3, c4);
    }

    Matrix toComplexMatrix(Func f, std::vector<Expr> args){
        Complex c1, c2, c3, c4;
        c1 = Complex(f(concat({0,0,0}, args)), f(concat({1,0,0}, args)));
        c2 = Complex(f(concat({0,1,0}, args)), f(concat({1,1,0}, args)));
        c3 = Complex(f(concat({0,0,1}, args)), f(concat({1,0,1}, args)));
        c4 = Complex(f(concat({0,1,1}, args)), f(concat({1,1,1}, args)));

        return Matrix(c1, c2, c3, c4);
    }

    Func matrixId(Func in){
        Func v_res0("v_res0");

        v_res0(vis) = Matrix(
            Complex(in(0,0,0,vis), in(1,0,0,vis)),
            Complex(in(0,1,0,vis), in(1,1,0,vis)),
            Complex(in(0,0,1,vis), in(1,0,1,vis)),
            Complex(in(0,1,1,vis), in(1,1,1,vis))
        );

        return matrixToDimensions(v_res0, {vis});
    }

    Func matrixToDimensions(Func in, std::vector<Var> args){
        Matrix m = Matrix(in(args));
        Func v_res_out("v_res_out");

        v_res_out(concat({c, i, j}, args)) = select(
            c == 0 && i == 0 && j == 0, m.m00.real,
            c == 1 && i == 0 && j == 0, m.m00.imag,
            c == 0 && i == 1 && j == 0, m.m01.real,
            c == 1 && i == 1 && j == 0, m.m01.imag,
            c == 0 && i == 0 && j == 1, m.m10.real,
            c == 1 && i == 0 && j == 1, m.m10.imag,
            c == 0 && i == 1 && j == 1, m.m11.real,
            m.m11.imag
        );
        v_res_out.bound(c, 0, 2).bound(i, 0, 2).bound(j, 0, 2);

        return v_res_out;
    }

    Func diagMatrixToDimensions(Func in, std::vector<Var> args){
        MatrixDiag m = MatrixDiag(in(args));
        Func v_res_out("v_res_out");

        v_res_out(concat({c,i}, args)) = select(
            c == 0 && i == 0, m.m00.real,
            c == 1 && i == 0, m.m00.imag,
            c == 0 && i == 1, m.m11.real,
                              m.m11.imag
        );
        v_res_out.bound(c, 0, 2).bound(i, 0, 2);

        return v_res_out;
    }

    Func AddOrSubtractDirection(bool add, Func v_res_in_local){
        Complex sol_ann_1_0 = sol_ann(0,vis,antenna_1(vis));
        Complex sol_ann_1_1 = sol_ann(1,vis,antenna_1(vis));
        Complex sol_ann_2_0 = sol_ann(0,vis,antenna_2(vis));
        Complex sol_ann_2_1 = sol_ann(1,vis,antenna_2(vis));
        Matrix modelM = Matrix(model(vis));

        Func contribution("contribution");
        contribution(vis) = 
        Matrix(
            sol_ann_1_0 * modelM.m00 * conj(sol_ann_2_0),
            sol_ann_1_0 * modelM.m01 * conj(sol_ann_2_1),
            sol_ann_1_1 * modelM.m10 * conj(sol_ann_2_0),
            sol_ann_1_1 * modelM.m11 * conj(sol_ann_2_1)
        );

        Func v_res0("v_res0");
        v_res0(vis) = toComplexMatrix(v_res_in_local, {vis});

        if(add){
            v_res0(vis) = Matrix(v_res0(vis)) + Matrix(contribution(vis));
        } else {
            v_res0(vis) = Matrix(v_res0(vis)) - Matrix(contribution(vis));
        }

        return v_res0;
    }

    Func TestNumerator(Func v_res_in_local){
        Func numerator("numerator"), denominator("denominator");
        numerator(si,a) = MatrixDiag({0.0f, 0.0f, 0.0f, 0.0f});
        denominator(i,si,a) = 0.0f;

        v_res_in_local.compute_root();

        RDom rv(0, n_vis, "rv");
        Expr rel_solution_index = (unsafe_promise_clamped(
            cast<int>(solution_map(rv) - solution_map(0))
            , 0, n_dir_sol-1));

        Complex sol_ann_1_0_ = sol_ann(0,rv,antenna_1(rv));
        Complex sol_ann_1_1_ = sol_ann(1,rv,antenna_1(rv));
        Complex sol_ann_2_0_ = sol_ann(0,rv,antenna_2(rv));
        Complex sol_ann_2_1_ = sol_ann(1,rv,antenna_2(rv));
        Matrix modelM = Matrix(model(rv));

        MatrixDiag solution_1 = {sol_ann_2_0_, sol_ann_2_1_};
        Matrix cor_model_transp_1 = solution_1 * HermTranspose(modelM);

        numerator(rel_solution_index, antenna_1(rv)) += Diagonal(
            Matrix(v_res_in_local(rv)) * cor_model_transp_1
        );        

        denominator(0, rel_solution_index, antenna_1(rv)) += 
            cor_model_transp_1.m00.norm() + cor_model_transp_1.m10.norm();
        denominator(1, rel_solution_index, antenna_1(rv)) += 
            cor_model_transp_1.m01.norm() + cor_model_transp_1.m11.norm();

        MatrixDiag solution_2 = {sol_ann_1_0_, sol_ann_1_1_};
        Matrix cor_model_2 = solution_2 * modelM;

        numerator(rel_solution_index, antenna_2(rv)) += Diagonal(
            HermTranspose(Matrix(v_res_in_local(rv))) * cor_model_2
        );

        denominator(0, rel_solution_index, antenna_2(rv)) += 
            cor_model_2.m00.norm() + cor_model_2.m10.norm();
        denominator(1, rel_solution_index, antenna_2(rv)) += 
            cor_model_2.m01.norm() + cor_model_2.m11.norm();

        Expr nan = Expr(std::numeric_limits<double>::quiet_NaN());
        Complex cnan = Complex(nan, nan);
        numerator.compute_root();
        denominator.compute_root();

        return diagMatrixToDimensions(numerator, {si,a});
    }

    Func SolveDirection(Func v_res_in_local){
        Func numerator("numerator"), denominator("denominator");
        numerator(si,a) = MatrixDiag({0.0f, 0.0f, 0.0f, 0.0f});
        denominator(i,si,a) = 0.0f;

        v_res_in_local.compute_root();

        RDom rv(0, n_vis, "rv");
        Expr rel_solution_index = (unsafe_promise_clamped(
            cast<int>(solution_map(rv) - solution_map(0))
            , 0, n_dir_sol-1));

        Complex sol_ann_1_0_ = sol_ann(0,rv,antenna_1(rv));
        Complex sol_ann_1_1_ = sol_ann(1,rv,antenna_1(rv));
        Complex sol_ann_2_0_ = sol_ann(0,rv,antenna_2(rv));
        Complex sol_ann_2_1_ = sol_ann(1,rv,antenna_2(rv));
        Matrix modelM = Matrix(model(rv));

        MatrixDiag solution_1 = {sol_ann_2_0_, sol_ann_2_1_};
        Matrix cor_model_transp_1 = solution_1 * HermTranspose(modelM);

        numerator(rel_solution_index, antenna_1(rv)) += Diagonal(
            Matrix(v_res_in_local(rv)) * cor_model_transp_1
        );        

        denominator(0, rel_solution_index, antenna_1(rv)) += 
            cor_model_transp_1.m00.norm() + cor_model_transp_1.m10.norm();
        denominator(1, rel_solution_index, antenna_1(rv)) += 
            cor_model_transp_1.m01.norm() + cor_model_transp_1.m11.norm();

        MatrixDiag solution_2 = {sol_ann_1_0_, sol_ann_1_1_};
        Matrix cor_model_2 = solution_2 * modelM;

        numerator(rel_solution_index, antenna_2(rv)) += Diagonal(
            HermTranspose(Matrix(v_res_in_local(rv))) * cor_model_2
        );

        denominator(0, rel_solution_index, antenna_2(rv)) += 
            cor_model_2.m00.norm() + cor_model_2.m10.norm();
        denominator(1, rel_solution_index, antenna_2(rv)) += 
            cor_model_2.m01.norm() + cor_model_2.m11.norm();

        Expr nan = Expr(std::numeric_limits<double>::quiet_NaN());
        Complex czero = Complex(Expr(0.0), Expr(0.0));
        Complex cnan = Complex(nan, nan);
        numerator.compute_root();
        denominator.compute_root();

        Func next_solutions("next_solutions");
        next_solutions(pol,si,a) = Complex(Expr(0.0),Expr(0.0));// {undef<double>(), undef<double>()};

        next_solutions(0,si,a) = tuple_select(
            denominator(0,si,a) == 0.0f,
            cnan,
            castc<double>(MatrixDiag(numerator(si,a)).m00) / cast<double>(denominator(0,si,a))
        );

        next_solutions(1,si,a) = tuple_select(
            denominator(1,si,a) == 0.0f,
            cnan,
            castc<double>(MatrixDiag(numerator(si,a)).m11) / cast<double>(denominator(1,si,a))
        );

        next_solutions.bound(pol, 0, 2).bound(si, 0, n_dir_sol).bound(a, 0, n_antennas);

        Func next_solutions_complex("next_solutions_complex");
        Expr rel_sol_index = clamp(si - solution_index0, 0, n_dir_sol-1);
        Expr op1 = mux(c, {Complex(next_solutions(pol, rel_sol_index, a)).real, Complex(next_solutions(pol, rel_sol_index, a)).imag});

        next_solutions_complex(c, pol, si, a) = 
            select(0 <= si - solution_index0 && si - solution_index0 < n_dir_sol, op1, undef(type_of<double>()));
            
        return next_solutions_complex;
    }

    void compile(){
        Func v_sub_out = AddOrSubtractDirection(false, v_res_in);
        Func v_sub_out_matrix = matrixToDimensions(v_sub_out, {vis});
        Func idFunc = matrixId(v_res_in);

        Func v_add_out = AddOrSubtractDirection(true, v_res_in);
        Func solve_out = SolveDirection(v_add_out);

        Func testNumerator = TestNumerator(v_res0);

        idFunc.compile_to_c("IdHalide.cc", args, "IdHalide");
        idFunc.compile_to_static_library("IdHalide", args, "IdHalide");

        testNumerator.compile_to_c("TestNumerator.cc", args, "TestNumerator");
        testNumerator.compile_to_static_library("TestNumerator", args, "TestNumerator");

        v_sub_out_matrix.compile_to_c("SubDirectionHalide.cc", args, "SubDirection");
        v_sub_out_matrix.compile_to_static_library("SubDirectionHalide", args, "SubDirection");

        solve_out.compile_to_c("SolveDirectionHalide.cc", args, "SolveDirection");
        solve_out.compile_to_static_library("SolveDirectionHalide", args, "SolveDirection");
        solve_out.print_loop_nest();

        compile_standalone_runtime("HalideRuntime.o", get_target_from_environment());
    }
};


int main(int argc, char **argv){
    bool complete = false;
    // if(complete){
    //     return HalideDiagonalComplete();
    // } else {
    //     return HalideDiagonalPartial();
    // }

    HalideDiagionalSolver solver;
    solver.compile();

    // solver.AddOrSubtractDirection(true);
    
    // next_solutions(ch_block, ant, solution_index, pol)
}