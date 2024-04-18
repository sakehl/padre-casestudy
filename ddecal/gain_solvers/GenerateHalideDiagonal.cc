#include "Halide.h"
#include "HalideComplex.h"

// using dp3::ddecal;
using namespace Halide;

void set_bounds(std::vector<std::tuple<Expr, Expr>> dims, Halide::OutputImageParam p){
    Expr stride = 1;
    for(size_t i = 0; i < dims.size(); i++){
        p.dim(i).set_bounds(std::get<0>(dims[i]), std::get<1>(dims[i]));
        p.dim(i).set_stride(stride);
        stride *= std::get<1>(dims[i]);
    }
}

class HalideDiagionalSolver{
public:
    // Inputs
    ImageParam ant1;
    ImageParam ant2;
    ImageParam solution_map;
    ImageParam v_res_in;
    ImageParam model_;
    ImageParam sol_;
    Param<int> _solution_index0;
    Param<int> _n_dir_sol;
    Param<int> _n_solutions;
    Param<int> _n_vis;
    Param<int> _n_antennas;

    Expr solution_index0;
    Expr n_dir_sol;
    Expr n_solutions;
    Expr n_vis;
    Expr n_antennas;
    
    Func model, sol;
    Var x, y, i, j, vis, si, a, pol, c;

    Func antenna_1, antenna_2, solution_index, v_res0;
    Func sol_ann;
    std::vector<Argument> args;

    int schedule;
    bool gpu;

    HalideDiagionalSolver() :
        ant1(type_of<int32_t>(), 1, "ant1"), // <1>[n_ant] uint32_t
        ant2(type_of<int32_t>(), 1, "ant2"), // <1>[n_ant] uint32_t
        solution_map(type_of<int32_t>(), 1, "solution_map"), // <1>[n_vis] uint32_t
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

        schedule = 0;
        gpu = false;

        solution_index0 = _solution_index0;
        n_dir_sol = _n_dir_sol;
        n_solutions = _n_solutions;
        n_vis = _n_vis;
        n_antennas = _n_antennas;

        // solution_index0 = 0;
        // n_dir_sol = 3;
        // n_solutions = 8;
        // n_vis = 230930;
        // n_antennas = 50;

        // ImageParam ant1;
        // ImageParam ant2;
        // ImageParam solution_map;
        // ImageParam v_res_in;
        // ImageParam model_;
        // ImageParam sol_;
        
        set_bounds({{0, n_vis}}, ant1);
        set_bounds({{0, n_vis}}, ant2);
        set_bounds({{0, n_vis}}, solution_map);
        set_bounds({{0, 2}, {0, 2}, {0, 2}, {0, n_vis}}, v_res_in);
        set_bounds({{0, 2}, {0, 2}, {0, 2}, {0, n_vis}}, model_);
        set_bounds({{0, 2}, {0, 2}, {0, n_solutions}, {0, n_antennas}}, sol_);
        
        
        model(vis) = toComplexMatrix(model_, {vis});
        sol(i, si, a) = Tuple(sol_(0, i, si, a), sol_(1, i, si, a));

        antenna_1(vis) = unsafe_promise_clamped(cast<int>(ant1(vis)), 0, n_antennas-1);
        antenna_2(vis) = unsafe_promise_clamped(cast<int>(ant2(vis)), 0, n_antennas-1);
        solution_index(vis) = unsafe_promise_clamped(cast<int>(solution_map(vis)), solution_index0, solution_index0+n_dir_sol-1);

        sol_ann(i,vis,a) = castc<float>(sol(i,solution_index(vis),a));

        v_res0(vis) = toComplexMatrix(v_res_in, {vis});

        args = {sol_, solution_map, ant1, ant2, model_, v_res_in, _solution_index0, _n_dir_sol, _n_vis, _n_solutions, _n_antennas};
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

    MatrixDiag toDiagMatrix(Func f, std::vector<Expr> args){
        Complex c00, c11;
        c00 = f(concat({0}, args));
        c11 = f(concat({1}, args));

        return MatrixDiag(c00, c11);
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
        
        Var vis_i, vis_o;
        Func out = matrixToDimensions(v_res0, {vis});
        return out;
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

        Func v_res0("v_res0"), v_res1("v_res1");
        v_res0(vis) = toComplexMatrix(v_res_in_local, {vis});

        if(add){
            v_res1(vis) = Matrix(v_res0(vis)) + Matrix(contribution(vis));
        } else {
            v_res1(vis) = Matrix(v_res0(vis)) - Matrix(contribution(vis));
        }

        return v_res1;
    }

    Func TestNumerator(Func v_res_in_local){
        Func numerator("numerator"), denominator("denominator");
        numerator(si,a) = MatrixDiag({0.0f, 0.0f, 0.0f, 0.0f});
        denominator(i,si,a) = 0.0f;

        v_res_in_local.compute_root();
        RDom rv2(0, 2, 0, n_vis, "rv2");

        Complex sol_ann_1_0_ = sol_ann(0,vis,antenna_1(vis));
        Complex sol_ann_1_1_ = sol_ann(1,vis,antenna_1(vis));
        Complex sol_ann_2_0_ = sol_ann(0,vis,antenna_2(vis));
        Complex sol_ann_2_1_ = sol_ann(1,vis,antenna_2(vis));
        Matrix modelM = Matrix(model(vis));

        MatrixDiag solution_1 = {sol_ann_2_0_, sol_ann_2_1_};
        Matrix cor_model_transp_1 = solution_1 * HermTranspose(modelM);
        
        MatrixDiag solution_2 = {sol_ann_1_0_, sol_ann_1_1_};
        Matrix cor_model_2 = solution_2 * modelM;

        Func denominator_inter("denominator_inter");
        Func ant_i("ant_i");
        ant_i(a, vis) = select(a == 0, antenna_1(vis), antenna_2(vis));
        
        denominator_inter(a, i, vis)
            = select( a==0 && i==0, cor_model_transp_1.m00.norm() + cor_model_transp_1.m10.norm()
                    , a==0 && i==1, cor_model_transp_1.m01.norm() + cor_model_transp_1.m11.norm()
                    , a==1 && i==0, cor_model_2.m00.norm() + cor_model_2.m10.norm()
                    , cor_model_2.m01.norm() + cor_model_2.m11.norm()
                );
        
        Expr sol_index = solution_index(rv2.y);
        denominator(i, sol_index, ant_i(rv2.x, rv2.y)) += denominator_inter(rv2.x, i, rv2.y);

        Func numerator_inter("numerator_inter");
        numerator_inter(a, vis)
            = tuple_select(a==0, Diagonal(Matrix(v_res_in_local(vis)) * cor_model_transp_1),
                     Diagonal(HermTranspose(Matrix(v_res_in_local(vis))) * cor_model_2));
    
        numerator(sol_index, ant_i(rv2.x, rv2.y)) += numerator_inter(rv2.x, rv2.y);

        denominator.update().reorder(i, rv2.x, rv2.y).unroll(i).unroll(rv2.x);
        numerator.update().reorder(rv2.x, rv2.y).unroll(rv2.x);
        RVar rv2_y_i("rv2_y_i"), rv2_y_o("rv2_y_o");
        
        numerator.compute_root();
        denominator.compute_root();

        return diagMatrixToDimensions(numerator, {si,a});
    }

    Func SolveDirection(Func v_res_in){
        Func v_res_in_local = AddOrSubtractDirection(true, v_res_in);

        Func numerator("numerator"), denominator("denominator");
        numerator(si,a) = MatrixDiag({0.0f, 0.0f, 0.0f, 0.0f});
        denominator(i,si,a) = 0.0f;

        RDom rv2(0, 2, 0, n_vis, "rv2");

        Complex sol_ann_1_0_ = sol_ann(0,vis,antenna_1(vis));
        Complex sol_ann_1_1_ = sol_ann(1,vis,antenna_1(vis));
        Complex sol_ann_2_0_ = sol_ann(0,vis,antenna_2(vis));
        Complex sol_ann_2_1_ = sol_ann(1,vis,antenna_2(vis));

        MatrixDiag solution_1 = {sol_ann_2_0_, sol_ann_2_1_};
        Func cor_model_transp_1("cor_model_transp_1");
        cor_model_transp_1(vis) = solution_1 * HermTranspose(Matrix(model(vis)));
        
        MatrixDiag solution_2 = {sol_ann_1_0_, sol_ann_1_1_};
        Func cor_model_2("cor_model_2");
        cor_model_2(vis) = solution_2 * Matrix(model(vis));
        Matrix cor_model_2M = Matrix(cor_model_2(vis));

        Func ant_i("ant_i");
        ant_i(a, vis) = select(a == 0, antenna_1(vis), antenna_2(vis));

        Func denominator_inter("denominator_inter");
        denominator_inter(a, i, vis) = undef<float>();
        denominator_inter(0, 0, vis) = Matrix(cor_model_transp_1(vis)).m00.norm() + Matrix(cor_model_transp_1(vis)).m10.norm();
        denominator_inter(0, 1, vis) = Matrix(cor_model_transp_1(vis)).m01.norm() + Matrix(cor_model_transp_1(vis)).m11.norm();
        denominator_inter(1, 0, vis) = Matrix(cor_model_2(vis)).m00.norm() + Matrix(cor_model_2(vis)).m10.norm();
        denominator_inter(1, 1, vis) = Matrix(cor_model_2(vis)).m01.norm() + Matrix(cor_model_2(vis)).m11.norm();
        denominator(i, solution_index(rv2.y), ant_i(rv2.x, rv2.y)) += denominator_inter(rv2.x, i, rv2.y);

        Func numerator_inter("numerator_inter");
        numerator_inter(a, vis) = {undef<float>(), undef<float>(), undef<float>(), undef<float>()};
        numerator_inter(0, vis) = Diagonal(Matrix(v_res_in_local(vis)) * Matrix(cor_model_transp_1(vis)));
        numerator_inter(1, vis) = Diagonal(HermTranspose(Matrix(v_res_in_local(vis))) * Matrix(cor_model_2(vis)));
        numerator(solution_index(rv2.y), ant_i(rv2.x, rv2.y)) += numerator_inter(rv2.x, rv2.y);

        Expr nan = Expr(std::numeric_limits<double>::quiet_NaN());
        Complex czero = Complex(Expr(0.0), Expr(0.0));
        Complex cnan = Complex(nan, nan);

        Func next_solutions("next_solutions");
        Func next_solutions_inter("next_solutions_inter");
        next_solutions_inter(pol ,si,a) = {undef<double>(), undef<double>()};
        next_solutions_inter(0,si,a) = tuple_select(
            denominator(0,si,a) == 0.0f,
            cnan,
            castc<double>(MatrixDiag(numerator(si,a)).m00) / cast<double>(denominator(0,si,a))
        );
        next_solutions_inter(1,si,a) = tuple_select(
            denominator(1,si,a) == 0.0f,
            cnan,
            castc<double>(MatrixDiag(numerator(si,a)).m11) / cast<double>(denominator(1,si,a))
        );
 
        next_solutions(pol,si,a) = tuple_select(pol==0, next_solutions_inter(0,si,a), next_solutions_inter(1,si,a));

        Func next_solutions_complex("next_solutions_complex");
        next_solutions_complex(c, pol, si, a) = mux(c, {Complex(next_solutions(pol, si, a)).real, Complex(next_solutions(pol, si, a)).imag});

        next_solutions_complex.bound(c,0,2).bound(pol, 0, 2).bound(a, 0, n_antennas).bound(si, solution_index0, n_dir_sol);
        

        set_bounds({{0, 2}, {0, 2}}, next_solutions_complex.output_buffer());
        next_solutions_complex.output_buffer().dim(2).set_stride(2*2);
        next_solutions_complex.output_buffer().dim(2).set_bounds(solution_index0, n_dir_sol);
        next_solutions_complex.output_buffer().dim(3).set_stride(2*2*n_solutions);
        next_solutions_complex.output_buffer().dim(3).set_bounds(0, n_antennas);
        if(schedule == 0) {
            numerator.compute_root();
            denominator.compute_root();
            cor_model_transp_1.compute_root();
            cor_model_2.compute_root();
        }
        if(schedule == 1) {
            numerator.compute_root();
            denominator.compute_root();
            denominator.update().reorder(i, rv2.x, rv2.y).unroll(i).unroll(rv2.x);
            numerator.update().reorder(rv2.x, rv2.y).unroll(rv2.x);
            numerator.update().compute_with(denominator.update(), rv2.y);
            cor_model_transp_1.compute_at(denominator, rv2.y);
            cor_model_2.compute_at(denominator, rv2.y);
            cor_model_2.compute_with(cor_model_transp_1, vis);

            next_solutions_complex.unroll(c).unroll(pol);
        }

            
        return next_solutions_complex;
    }

    void compile(){
        try{
            Target target = get_target_from_environment();
            if(gpu){
                target.set_feature(Target::OpenCL);
                target.set_feature(Target::CLDoubles);
            }
            
            target.set_feature(Target::AVX512);
            target.set_features({Target::NoAsserts, Target::NoBoundsQuery});
            // target.set_feature(Target::Debug);
            if(!host_supports_target_device(target)){ 
                std::cout << "The target " << target.to_string() << " is not supported on this host." << std::endl;
            } else {
                std::cout << "The target " << target.to_string() << " is supported on this host." << std::endl;
            }
            
            Func v_sub_out = AddOrSubtractDirection(false, v_res_in);
            Func v_sub_out_matrix = matrixToDimensions(v_sub_out, {vis});
            v_sub_out_matrix.bound(vis, 0, n_vis);
            set_bounds({{0, 2}, {0, 2}, {0, 2}, {0, n_vis}}, v_sub_out_matrix.output_buffer());

            Func idFunc = matrixId(v_res_in);
            Func testNumerator = TestNumerator(v_res0);
            Func solve_out = SolveDirection(v_res_in);
            
            // Bounds on input
            ant1.requires((ant1(_0) >= 0 && ant1(_0) < n_antennas));
            ant2.requires((ant2(_0) >= 0 && ant2(_0) < n_antennas));
            solution_map.requires((solution_map(_0) >= solution_index0 
                && solution_map(_0) < solution_index0+n_dir_sol));
            Annotation bounds = context(solution_index0 >= 0 && n_dir_sol > 0 
                && solution_index0 + n_dir_sol <= n_solutions
                && n_antennas>0 && n_vis>0 && n_solutions>0);
        
            idFunc.compile_to_c("IdHalide.c", args, {bounds}, "IdHalide", target);
            idFunc.compile_to_static_library("IdHalide", args, "IdHalide", target);

            testNumerator.compile_to_c("TestNumerator.c", args, {bounds}, "TestNumerator", target);
            testNumerator.compile_to_static_library("TestNumerator", args, "TestNumerator", target);

            v_sub_out_matrix.compile_to_c("SubDirectionHalide.c", args, {bounds}, "SubDirection", target);
            v_sub_out_matrix.compile_to_static_library("SubDirectionHalide", args, "SubDirection", target);
            v_sub_out_matrix.compile_to_lowered_stmt("SubDirectionHalide.html", args, StmtOutputFormat::HTML, target);
            v_sub_out_matrix.print_loop_nest();

            solve_out.compile_to_c("SolveDirectionHalide.c", args, {bounds}, "SolveDirection", target);
            solve_out.compile_to_static_library("SolveDirectionHalide", args, "SolveDirection", target);
            solve_out.compile_to_lowered_stmt("SolveDirectionHalide.html", args, StmtOutputFormat::HTML, target);
            solve_out.print_loop_nest();

            compile_standalone_runtime("HalideRuntime.o", target);
        } catch (Halide::Error &e){
            std::cerr << "Halide Error: " << e.what() << std::endl;
            __throw_exception_again;
        }
    }
};


int main(int argc, char **argv){

    HalideDiagionalSolver solver;
    solver.compile();
}