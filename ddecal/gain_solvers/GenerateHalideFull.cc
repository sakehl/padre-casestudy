#include "Halide.h"
#include "HalideComplex.h"
#include <math.h>

using namespace Halide;

template<typename T>
std::vector<T> concat(std::vector<T> a, std::vector<T> b){
    std::vector<T> out;
    out.reserve(a.size() + b.size());
    out.insert(out.end(), a.begin(), a.end());
    out.insert(out.end(), b.begin(), b.end());
    return out;
}

void set_bounds(std::vector<std::tuple<Expr, Expr>> dims, Halide::OutputImageParam p){
    Expr stride = 1;
    for(size_t i = 0; i < dims.size(); i++){
        p.dim(i).set_bounds(std::get<0>(dims[i]), std::get<1>(dims[i]));
        p.dim(i).set_stride(stride);
        stride *= std::get<1>(dims[i]);
    }
}

class HalideFullSolver{
public:
    // Inputs
    ImageParam ant;
    ImageParam solution_map;
    ImageParam v_res_;
    ImageParam model_;
    ImageParam sol_;
    ImageParam next_sol_;
    ImageParam n_sol0_direction;
    ImageParam n_sol_direction;
    ImageParam n_dir;
    ImageParam n_vis;
    
    Param<int> n_cb;
    Param<int> n_sol;
    Param<int> n_antennas;
    Param<int> max_n_visibilities;
    Param<int> max_n_direction_solutions;
    Param<int> max_n_directions;
    Param<double> step_size;
    Param<bool> phase_only;

    Func v_res, model, sol, next_sol, antenna_1, antenna_2, solution_index;
    Var x, y, i, j, v, si, a, pol, c, cb, d;

    int schedule;
    bool gpu;

    HalideFullSolver() :
        ant(type_of<int32_t>(), 3, "ant"), // <3>[n_cb][n_vis][2] uint32_t
        solution_map(type_of<int32_t>(), 3, "solution_map"), // <3>[n_cb][dir][n_vis] uint32_t
        v_res_(type_of<float>(), 5, "v_res_"), // <5>[n_cb][n_vis], Complex 2x2 Float (+3)
        model_(type_of<float>(), 6, "model_"), // <6>[n_cb][dir][n_vis], Complex 2x2 Float (+3)

        sol_(type_of<double>(), 5, "sol_"), // <5> [n_cb][n_ant][n_dir_sol][2] Complex Double (+1)
        next_sol_(type_of<double>(), 5, "next_sol_"), // <5> [n_cb][n_ant][n_dir_sol][2] Complex Double (+1)

        n_sol0_direction(type_of<int32_t>(), 2, "n_sol0_direction"), // <2> [n_cb][dir] uint32_t
        n_sol_direction(type_of<int32_t>(), 2, "n_sol_direction"), // <2> [n_cb][dir] uint32_t
        n_dir(type_of<int32_t>(), 1, "n_dir"), // <1> [n_cb] uint32_t
        n_vis(type_of<int32_t>(), 1, "n_vis"), // <1> [n_cb] uint32_t
        
        n_cb("n_cb"),
        n_sol("n_sol"),
        n_antennas("n_antennas"),
        max_n_visibilities("max_n_visibilities"),
        max_n_direction_solutions("max_n_direction_solutions"),
        max_n_directions("max_n_directions"),
        step_size("step_size"),
        phase_only("phase_only"),
        
        v_res("v_res"), model("model"), sol("sol"), next_sol("next_sol"),
        antenna_1("antenna_1"), antenna_2("antenna_2"), solution_index("solution_index"),
        x("x"), y("y"), i("i"), j("j"), v("v"), si("si"), a("a"), pol("pol"), c("c"), cb("cb"), d("d")
        {

        schedule = 0;

        set_bounds({{0,2}, {0, max_n_visibilities}, {0,n_cb}}, ant);
        set_bounds({{0, max_n_visibilities}, {0,max_n_directions}, {0,n_cb}}, solution_map);

        set_bounds({{0, 2}, {0, 2}, {0, 2}, {0, max_n_visibilities}, {0,n_cb}}, v_res_);
        set_bounds({{0, 2}, {0, 2}, {0, 2}, {0, max_n_visibilities}, {0,max_n_directions}, {0,n_cb}}, model_);
        set_bounds({{0, 2}, {0, 2}, {0, n_sol}, {0, n_antennas}, {0,n_cb}}, sol_);
        set_bounds({{0, 2}, {0, 2}, {0, n_sol}, {0, n_antennas}, {0,n_cb}}, next_sol_);

        set_bounds({{0, max_n_directions}, {0, n_cb}}, n_sol0_direction);
        set_bounds({{0, max_n_directions}, {0, n_cb}}, n_sol_direction);
        set_bounds({{0, n_cb}}, n_dir);
        set_bounds({{0, n_cb}}, n_vis);
        
        model(v, d, cb) = toComplexMatrix(model_, {v, d, cb});
        sol(si, a, cb) = toComplexDiagMatrix(sol_, {si, a, cb});
        next_sol(si, a, cb) = toComplexDiagMatrix(next_sol_, {si, a, cb});
        
        antenna_1(v, cb) = unsafe_promise_clamped(cast<int>(ant(0, v, cb)), 0, n_antennas-1);
        antenna_2(v, cb) = unsafe_promise_clamped(cast<int>(ant(1, v, cb)), 0, n_antennas-1);
        solution_index(v, d, cb) = 
          unsafe_promise_clamped(cast<int>(solution_map(v, d, cb)), 0, n_sol-1);

        v_res(v, cb) = toComplexMatrix(v_res_, {v, cb});
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

    MatrixDiag toComplexDiagMatrix(Func f, std::vector<Expr> args){
        Complex c00, c11;
        c00 = Complex(f(concat({0,0}, args)), f(concat({1,0}, args)));
        c11 = Complex(f(concat({0,1}, args)), f(concat({1,1}, args)));

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

    Func out(int debug_stop, bool gpu){
        Func result("out");
        Func v_res_sub("v_res_sub");
        Func denominator_inter("denominator_inter"), numerator_inter("numerator_inter");
        Func numerator("numerator"), denominator("denominator");
        Func ant_i("ant_i");
        Func contribution("contribution");
        Func cor_model_transp_1("cor_model_transp_1"), cor_model_2("cor_model_2");

        // First, substract all contributions from the current model
        Expr vb = clamp(v, 0, n_vis(cb)-1);
        v_res_sub(v, cb) = v_res(vb, cb);
        if(debug_stop == 0){
            Func res = matrixToDimensions(v_res_sub, {v, cb});
            set_bounds({{0, 2}, {0, 2}, {0, 2}, {0, max_n_visibilities}, {0, n_cb}}, res.output_buffer());
            return res;
        }
        RDom dir(0, max_n_directions, "dir");
        dir.where(dir < n_dir(cb));

        MatrixDiag solution_1 = castT<float>(sol(solution_index(v, d, cb), antenna_1(v, cb), cb));
        MatrixDiag solution_2 = castT<float>(sol(solution_index(v, d, cb), antenna_2(v, cb), cb));
        contribution(v, d, cb) = solution_1 * Matrix(model(v, d, cb)) * HermTranspose(solution_2);
        v_res_sub(v, cb) = Matrix(v_res_sub(v, cb)) - Matrix(contribution(vb, dir, cb));
        if(debug_stop == 1){
            Func res = matrixToDimensions(v_res_sub, {v, cb});
            set_bounds({{0, 2}, {0, 2}, {0, 2}, {0, max_n_visibilities}, {0, n_cb}}, res.output_buffer());
            return res;
        }

        ///////////////////////////////////////////
        // Add this direction back before solving
        Func vis_in_add("vis_in_add");
        vis_in_add(v, d, cb) = Matrix(v_res_sub(v, cb)) + Matrix(contribution(v, d, cb));

        cor_model_transp_1(v, d, cb) = solution_2 * HermTranspose(Matrix(model(v, d, cb)));
        cor_model_2(v, d, cb) = solution_1 * Matrix(model(v, d, cb));

        ////////////////////////////////
        // Inter
        numerator_inter(a, v, d, cb) = {undef<float>(), undef<float>(), undef<float>(), undef<float>()};
        numerator_inter(0, v, d, cb) = Diagonal(Matrix(vis_in_add(v,d,cb)) * Matrix(cor_model_transp_1(v,d,cb)));
        numerator_inter(1, v, d, cb) = Diagonal(HermTranspose(Matrix(vis_in_add(v,d,cb))) * Matrix(cor_model_2(v,d,cb)));

        denominator_inter(a, i, v, d, cb) = undef<float>();
        denominator_inter(0, 0, v, d, cb) = Matrix(cor_model_transp_1(v,d,cb)).m00.norm() + Matrix(cor_model_transp_1(v,d,cb)).m10.norm();
        denominator_inter(0, 1, v, d, cb) = Matrix(cor_model_transp_1(v,d,cb)).m01.norm() + Matrix(cor_model_transp_1(v,d,cb)).m11.norm();
        denominator_inter(1, 0, v, d, cb) = Matrix(cor_model_2(v,d,cb)).m00.norm() + Matrix(cor_model_2(v,d,cb)).m10.norm();
        denominator_inter(1, 1, v, d, cb) = Matrix(cor_model_2(v,d,cb)).m01.norm() + Matrix(cor_model_2(v,d,cb)).m11.norm();

        ///////////////////////////////
        ant_i(a, v, cb) = select(a == 0, antenna_1(v, cb), antenna_2(v, cb));
        RDom rv2(0, 2, 0, max_n_visibilities, "rv2");
        rv2.where(rv2.y < n_vis(cb));
        Expr rel_si = unsafe_promise_clamped(solution_index(rv2.y, d, cb) - n_sol0_direction(d, cb), 0, max_n_direction_solutions-1);
        numerator(si,a,d,cb) = MatrixDiag({0.0f, 0.0f, 0.0f, 0.0f});
        numerator(rel_si, ant_i(rv2.x, rv2.y, cb), d, cb) += numerator_inter(rv2.x, rv2.y, d, cb);
        denominator(i,si,a,d,cb) = 0.0f;
        denominator(i,rel_si, ant_i(rv2.x, rv2.y, cb), d, cb) += denominator_inter(rv2.x, i, rv2.y, d, cb);

        ///////////////////

        Expr nan = Expr(std::numeric_limits<double>::quiet_NaN());
        Complex cnan = Complex(nan, nan);

        // RDom dir2(0, max_n_direction_solutions, 0, max_n_directions, "dir2");
        
        // RVar si_r = dir2.x;
        // RVar d_r = dir2.y;
        
        // d_r < n_dir(cb) && 
        // Expr si_r_total = unsafe_promise_clamped(n_sol0_direction(d_r, cb) + si_r, 0, n_sol-1);
        
        RDom si_r(0, max_n_direction_solutions, "si_r");
        si_r.where(si_r < n_sol_direction(d, cb));
        Func next_solutions_complex0("next_solutions_complex0");
        next_solutions_complex0(pol,si,d,a,cb) = {undef<double>(), undef<double>()};
        next_solutions_complex0(pol,si_r,d,a,cb) = tuple_select(
            denominator(pol,si_r,a,d,cb) == 0.0f, cnan,
            pol==0, castC<double>(MatrixDiag(numerator(si_r,a,d,cb)).m00) / cast<double>(denominator(0,si_r,a,d,cb)),
            castC<double>(MatrixDiag(numerator(si_r,a,d,cb)).m11) / cast<double>(denominator(1,si_r,a,d,cb))
        );

        RDom d_r2(0, max_n_direction_solutions, 0, max_n_directions, "d_r");
        RVar d_r = d_r2.y;
        RVar si_r2 = d_r2.x;
        d_r2.where( d_r < n_dir(cb) && si_r2 < n_sol_direction(d_r, cb));
        Func next_solutions_complex1("next_solutions_complex1");
        Expr si_total = unsafe_promise_clamped(n_sol0_direction(d_r, cb) + si_r2, 0, n_sol-1);
        next_solutions_complex1(pol,si,a,cb) = {undef<double>(), undef<double>()};
        next_solutions_complex1(pol,si_total,a,cb) = next_solutions_complex0(pol,si_r2,d_r,a,cb);

        Func next_solutions("next_solutions");

        // Step
        Expr pi = Expr(M_PI);
        Func distance0("distance0");
        Func distance1("distance1");
        Func next_solutions_step("next_solutions_step");
        Complex sol_c = Complex(sol_(0, i, si, a, cb), sol_(1, i, si, a, cb));
        Expr phase_from = arg(sol_c);
        distance0(i, si, a, cb) = phase_from - arg(next_solutions_complex1(i, si, a, cb));
        distance1(i, si, a, cb) = select(distance0(i, si, a, cb) > pi, distance0(i, si, a, cb) - 2*pi, distance0(i, si, a, cb) + 2*pi);
        Complex phase_only_true = polar(Expr(1.0), phase_from + step_size * distance1(i, si, a, cb));
        Complex phase_only_false = Complex(sol_c) * (Expr(1.0) - step_size) + Complex(next_solutions_complex1(i, si, a, cb)) * Complex(step_size);
        
        next_solutions_step(i, si, a, cb) = tuple_select(phase_only, 
           phase_only_true,
           phase_only_false);
           
        next_solutions(c, i, si, a, cb) = mux(c, 
            {Complex(next_solutions_step(i, si, a, cb)).real, 
             Complex(next_solutions_step(i, si, a, cb)).imag});
        next_solutions.bound(c,0,2).bound(i, 0, 2).bound(a, 0, n_antennas).bound(si, 0, n_sol).bound(cb, 0, n_cb);
        set_bounds({{0, 2}, {0, 2}, {0, n_sol}, {0, n_antennas}, {0, n_cb}}, next_solutions.output_buffer());

        if(gpu){
            Var block("block"), thread("thread"), fuse("fuse");
            RVar rblock("rblock"), rthread("rthread");  
            int block_size = 32;

            next_solutions.compute_root()
                .gpu_tile(a, block, thread, block_size, TailStrategy::GuardWithIf)
                .unroll(i)
                .unroll(c)
                .specialize(phase_only);
                ;

            next_solutions_complex1.compute_root()
                .update()
                .reorder(pol, si_r2, a, d_r)
                .gpu_tile(a, block, thread, block_size, TailStrategy::GuardWithIf)
                .unroll(pol);
                ;
            next_solutions_complex0.compute_at(next_solutions_complex1, d_r)
                .update()
                .reorder(pol, si_r, d, a)
                .gpu_tile(a, block, thread, block_size, TailStrategy::GuardWithIf)
                .unroll(pol);
                ;

            denominator.compute_at(next_solutions_complex1, d_r)
                .fuse(si, a, fuse)
                .gpu_tile(fuse, block, thread, block_size, TailStrategy::GuardWithIf)
                .update()
                .atomic()
                .reorder(i, rv2.x, rv2.y)
                .unroll(i)
                .unroll(rv2.x)
                .gpu_tile(rv2.y, rblock, rthread, block_size, TailStrategy::GuardWithIf)
                ;
            numerator.compute_at(next_solutions_complex1, d_r)
                .fuse(si, a, fuse)
                .gpu_tile(fuse, block, thread, block_size, TailStrategy::GuardWithIf)
                .update()
                .atomic()
                .unroll(rv2.x)
                .gpu_tile(rv2.y, rblock, rthread, block_size, TailStrategy::GuardWithIf)
#ifndef HAVE_HALIVER
                .compute_with(denominator.update(), rthread);
#endif
                ;
            // cor_model_transp_1.compute_at(denominator, rthread);

            // v_res_sub.compute_at(next_solutions_complex1, cb)
            //     .gpu_tile(v, block, thread, block_size, TailStrategy::GuardWithIf)
            //     .update()
            //     .reorder(dir, v, cb)
            //     .gpu_tile(v, block, thread, block_size, TailStrategy::GuardWithIf)
            //     ;
        } else if(schedule == 0){
            next_solutions.compute_root()
                .unroll(i)
                .unroll(c)
                .specialize(phase_only)
                .parallel(cb)
                ;

            next_solutions_complex1.compute_root()
                .update()
                .reorder(pol, si_r2, a, d_r)
                .unroll(pol)
                .parallel(cb)
                ;
            next_solutions_complex0.compute_at(next_solutions_complex1, d_r)
                .update()
                .reorder(pol, si_r, d, a)
                .unroll(pol);
                ;

            denominator.compute_at(next_solutions_complex1, d_r)
                .update()
                .reorder(i, rv2.x, rv2.y)
                .unroll(i)
                .unroll(rv2.x)
                ;
            numerator.compute_at(next_solutions_complex1, d_r)
                .update()
                .unroll(rv2.x)
// #ifndef HAVE_HALIVER                
                .compute_with(denominator.update(), rv2.y);
// #endif
                ;
#ifndef HAVE_HALIVER   
            cor_model_2.compute_at(denominator, rv2.y);
            cor_model_transp_1.compute_at(denominator, rv2.y)       
                .compute_with(cor_model_2, d)
                ;
#endif
            Var v_out("v_out"), v_in("v_in");
            v_res_sub.compute_at(next_solutions_complex1, cb)
                .update()
                .reorder(dir, v, cb)
                ;
        } else {
            Var fuse("fuse"), a_out("a_out"), a_in("a_in");
            next_solutions.compute_root()
                .split(a, a_out, a_in, n_antennas/8, TailStrategy::GuardWithIf)
                .parallel(a_out)
                .unroll(i)
                .unroll(c)
                .specialize(phase_only);
                ;

            next_solutions_complex1.compute_root()
                .update()
                .reorder(pol, si_r2, a, d_r)
                .split(a, a_out, a_in, n_antennas/8, TailStrategy::GuardWithIf)
                .parallel(a_out)
                .unroll(pol)
                // .parallel(cb)
                ;
            next_solutions_complex0.compute_at(next_solutions_complex1, d_r)
                .update()
                .reorder(pol, si_r, d, a)
                .split(a, a_out, a_in, n_antennas/8, TailStrategy::GuardWithIf)
                .parallel(a_out)
                .unroll(pol);
                ;

            RVar r_out("r_out"), r_in("r_in");
            denominator.compute_at(next_solutions_complex1, d_r)
                // .fuse(si, a, fuse)
                // .gpu_tile(fuse, block, thread, block_size, TailStrategy::GuardWithIf)
                .split(a, a_out, a_in, n_antennas/8, TailStrategy::GuardWithIf)
                .parallel(a_out)
                .unroll(i)
                .update()
                .reorder(i, rv2.x, rv2.y)
                .unroll(i)
                .unroll(rv2.x)
                .split(rv2.y, r_out, r_in, max_n_visibilities/8, TailStrategy::GuardWithIf)
                ;

            Var par("par");
            Func denom_inter("denom_inter");
            denom_inter = denominator.update().rfactor(r_out, par);
            denom_inter.compute_at(next_solutions_complex1, d_r)
                .parallel(par)
                .unroll(i)
                .update()
                .parallel(par)
                ;
            

            numerator.compute_at(next_solutions_complex1, d_r)
                .split(a, a_out, a_in, n_antennas/8, TailStrategy::GuardWithIf)
                .parallel(a_out)
                .update()
                .unroll(rv2.x)
                .split(rv2.y, r_out, r_in, max_n_visibilities/8, TailStrategy::GuardWithIf)
                ;

            Func num_inter("denom_inter");
            num_inter = numerator.update().rfactor(r_out, par);
            num_inter.compute_at(next_solutions_complex1, d_r)
                .parallel(par)
                .compute_with(denom_inter, si)
                .update()
                .parallel(par)
                .compute_with(denom_inter.update(), r_in);
                ;

            numerator.compute_with(denominator, si)
                .update()
                .compute_with(denominator.update(), r_out)
                ;

            Var v_out("v_out"), v_in("v_in");
            v_res_sub.compute_at(next_solutions_complex1, cb)
                .update()
                .split(v, v_out, v_in, max_n_visibilities/8, TailStrategy::GuardWithIf)
                .parallel(v_out)
                .reorder(dir, v_in, v_out, cb)
                ;
        }

        return next_solutions;
    }

    void compile(){
        try{
            std::vector<Argument> args = {ant, solution_map, v_res_, model_, sol_, next_sol_,
              n_sol0_direction, n_sol_direction, n_dir, n_vis,
              n_cb, n_sol, n_antennas, max_n_visibilities, max_n_direction_solutions, max_n_directions,
              step_size, phase_only};
            Target target = get_target_from_environment();
            target.set_feature(Target::AVX512);
            target.set_features({Target::NoAsserts, Target::NoBoundsQuery});
            // Target target_gpu = target;
            #ifndef HAVE_HALIVER
            target.set_features({Target::CUDA, Target::CLDoubles});
            #endif
            Func debug_vres_in("debug_vres_in");
            debug_vres_in = out(0, false);
            Func debug_substract_all("debug_substract_all");
            debug_substract_all = out(1, false);
            Func result("out"), result_gpu("out_gpu");
            result = out(-1, false);
            #ifndef HAVE_HALIVER
            result_gpu = out(-1, true);
            #endif

            #ifdef HAVE_HALIVER
            debug_vres_in.compile_to_c("VResIn.c", args, {}, "VResIn", target);
            debug_substract_all.compile_to_c("SubstractFull.c", args, {}, "SubstractFull", target);
            result.compile_to_c("PerformIterationHalide.c", args, {}, "PerformIterationHalide", target);
            // result_gpu.compile_to_c("PerformIterationHalideGPU.c", args, {}, "PerformIterationHalideGPU", target);
            #else
            debug_vres_in.compile_to_c("VResIn.cc", args, "VResIn", target);
            debug_substract_all.compile_to_c("SubstractFull.cc", args, "SubstractFull", target);
            result.compile_to_c("PerformIterationHalide.cc", args, "PerformIterationHalide", target);
            result_gpu.compile_to_c("PerformIterationHalideGPU.cc", args, "PerformIterationHalideGPU", target);
            #endif
            debug_vres_in.compile_to_static_library("VResIn", args, "VResIn", target);
            debug_substract_all.compile_to_static_library("SubstractFull", args, "SubstractFull", target);
            result.compile_to_static_library("PerformIterationHalide", args, "PerformIterationHalide", target);
            #ifndef HAVE_HALIVER
            result_gpu.compile_to_static_library("PerformIterationHalideGPU", args, "PerformIterationHalideGPU", target);
            #endif
            // result.compile_to_lowered_stmt("PerformIterationHalide.html", args, StmtOutputFormat::HTML, target);
            // result.compile_to_lowered_stmt("PerformIterationHalide.txt", args, StmtOutputFormat::Text, target);
            // result.print_loop_nest();
        } catch (Halide::Error &e){
            std::cerr << "Halide Error: " << e.what() << std::endl;
            __throw_exception_again;
        }
    }
};

int main(int argc, char **argv){

    HalideFullSolver solver;
    solver.compile();
}