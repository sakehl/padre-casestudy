#include "Halide.h"

using namespace Halide;

struct Complex {
    Expr real, imag;

    Complex() : real(Expr()), imag(Expr()) {
    }

    // Construct from a Tuple
    Complex(Tuple t)
        : real(t[0]), imag(t[1]) {
        assert (t.size() == 2);
    }

    // Construct from a pair of Exprs
    Complex(Expr r, Expr i)
        : real(r), imag(i) {
    }

    // Construct from a a single Expr (imaginary part is 0)
    Complex(Expr r)
        : real(r), imag(Expr(0.0)) {
    }

    // Construct from a call to a Func by treating it as a Tuple
    Complex(FuncRef t)
        : Complex(Tuple(t)) {
    }

    // Convert to a Tuple
    operator Tuple() const {
        return {real, imag};
    }

    // Complex addition
    Complex operator+(const Complex &other) const {
        return {real + other.real, imag + other.imag};
    }

    // Complex substraction
    Complex operator-(const Complex &other) const {
        return {real - other.real, imag - other.imag};
    }

    // Complex multiplication
    Complex operator*(const Complex &other) const {
        return {real * other.real - imag * other.imag,
                real * other.imag + imag * other.real};
    }

    Complex operator/(const Expr &other) const {
        return {real / other, imag / other};
    }

    // Complex conjugate
    Complex conj() const {
        return {real, -imag};
    }

    Expr norm() const {
        return real*real + imag*imag;
    }

    Complex cast_float() const {
        return {cast<float>(real), cast<float>(imag)};
    }

    Complex cast_double() const {
        return {cast<double>(real), cast<double>(imag)};
    }
};

struct Matrix {
    Complex m00, m01, m10, m11;

    Matrix(Tuple t)
        : m00(Complex(t[0],t[1])), m01(Complex(t[2],t[3])),
        m10(Complex(t[4],t[5])), m11(Complex(t[6],t[7])) {
            assert(t.size() == 8);
    }

    // Construct from a call to a Func by treating it as a Tuple
    Matrix(FuncRef t)
        : Matrix(Tuple(t)) {
    }

    // Construct from a pair of Exprs
    Matrix(Complex m00, Complex m01, Complex m10, Complex m11)
        : m00(m00), m01(m01), m10(m10), m11(m11) {
    }

    operator Tuple() const {
        return {m00.real, m00.imag, m01.real, m01.imag, 
            m10.real, m10.imag, m11.real, m11.imag};
    }

    Matrix operator+(const Matrix &other) const {
        return {m00 + other.m00, m01 + other.m01, m10 + other.m10, m11 + other.m11};
    }

    Matrix operator-(const Matrix &other) const {
        return {m00 - other.m00, m01 - other.m01, m10 - other.m10, m11 - other.m11};
    }

    Matrix operator*(const Matrix &other) const {
        return {
            m00 * other.m00 + m01 * other.m10,
            m00 * other.m01 + m01 * other.m11,
            m10 * other.m00 + m11 * other.m10,
            m10 * other.m01 + m11 * other.m11};
    }
};

struct MatrixDiag {
    Complex m00, m11;

    MatrixDiag(Tuple t)
        : m00(Complex(t[0],t[1])), m11(Complex(t[2],t[3])) {
            assert(t.size() == 4);
    }

    // Construct from a call to a Func by treating it as a Tuple
    MatrixDiag(FuncRef t)
        : MatrixDiag(Tuple(t)) {
    }

    // Construct from a pair of Exprs
    MatrixDiag(Complex m00, Complex m11)
        : m00(m00), m11(m11) {
    }

    operator Tuple() const {
        return {m00.real, m00.imag, m11.real, m11.imag};
    }

    MatrixDiag operator+(const MatrixDiag &other) const {
        return {m00 + other.m00, m11 + other.m11};
    }

    MatrixDiag operator-(const MatrixDiag &other) const {
        return {m00 - other.m00, m11 - other.m11};
    }

    MatrixDiag operator*(const MatrixDiag &other) const {
        return {m00 * other.m00, m11 * other.m11};
    }
};

Complex conj(Complex t){
    return {t.real, -t.imag};
}

Matrix HermTranspose(const Matrix& matrix){
    return Matrix(conj(matrix.m00), conj(matrix.m10), conj(matrix.m01), conj(matrix.m11));
}

MatrixDiag HermTranspose(const MatrixDiag& matrix){
    return MatrixDiag(conj(matrix.m00), conj(matrix.m11));
}

/**
 * Diagonal - non-diagonal Matrix multiplication operator
 */
Matrix operator*(const MatrixDiag& lhs,
                             const Matrix& rhs) {
  return Matrix(lhs.m00 * rhs.m00, lhs.m00 * rhs.m01, lhs.m11 * rhs.m10,
                            lhs.m11 * rhs.m11);
}

Matrix operator*(const Matrix& lhs,
                             const MatrixDiag& rhs) {
  return Matrix(lhs.m00 * rhs.m00, lhs.m01 * rhs.m11, lhs.m10 * rhs.m00,
                            lhs.m11 * rhs.m11);
}

MatrixDiag Diagonal(const Matrix matrix){
    return MatrixDiag{matrix.m00, matrix.m11};
}

template<typename T>
inline Tuple castT(Tuple t) {
    std::vector<Expr> result;
    std::transform(t.as_vector().begin(), t.as_vector().end(), std::back_inserter(result), [](Expr e) {
        return cast(type_of<T>(), std::move(e));
    });
    return Tuple(result);
}

template<typename T>
inline Complex castC(Complex t) {
    return castT<T>(t);
}

inline Expr arg(Complex t){
    return atan2(t.imag, t.real);
}

inline Complex polar(Expr r, Expr z){
    return Complex(r*cos(z), r*sin(z));
}

// template<typename T>
// inline Complex cast(Complex t) {
//     return {cast(type_of<T>(), std::move(t.real)),  cast(type_of<T>(), std::move(t.imag))};
// }

// template<typename T>
// inline Complex cast(MatrixDiag m) {
//     return {cast(type_of<T>(), std::move(m.m00)),  cast(type_of<T>(), std::move(m.m11))};
// }

//     // Inputs
//     ImageParam ant1(type_of<int>(), 2, "ant1"); // <2>[cb][n_ant] int
//     ImageParam ant2(type_of<int>(), 2, "ant2"); // <2>[cb][n_ant] int
//     ImageParam solution_map(type_of<int>(), 3, "solution_map"); // <3>[cb][n_dir][n_vis] int
//     ImageParam v_res_in(type_of<float>(), 5, "v_res_in"); // <5>[cb][n_vis], Complex 2x2 Float (3)
//     ImageParam model_(type_of<float>(), 6, "model_"); // <6>[cb][dir][n_vis], Complex 2x2 Float (3)
//     ImageParam sol_(type_of<double>(), 5, "sol_");
//     ImageParam n_dir_sol(type_of<int>(), 2, "n_dir_sol"); // <2>[cb][dir] int
//     ImageParam n_vis(type_of<int>(), 2, "n_vis"); // <2>[cb][dir] int

//     Func v_res0("v_res0"), v_res1("v_res1");
//     Func model("model"), sol("sol");
//     Var x("x"), y("y"), i("i"), j("j"), vis("vis"), d("d"), si("si"), a("a"), cb("cb"), pol("pol");

//     Func contribution("contribution");

//     // Func next_solutions("next_solutions");
//     //   {NChannelBlocks(), NAntennas(), NSolutions(), NSolutionPolarizations()});
//     Param<int> NChannelBlocks;
//     Param<int> NDirections;// 3;
//     Param<int>  NSolutions;// 8
//     Param<int> NAntennas;; //50;
//     model(i, j, vis, d, cb) = Tuple(model_(0, i, j, vis, d, cb), model_(1, i, j, vis, d, cb));
//     sol(i, si, a, cb) = Tuple(sol_(0, i, si, a, cb), sol_(1, i, si, a, cb));

//     Func antenna_1("antenna_1"), antenna_2("antenna_2"), solution_index("solution_index"), n_dir_solutions("n_dir_solutions");
//     antenna_1(vis,cb) = unsafe_promise_clamped(ant1(vis, cb), 0, NAntennas-1);
//     antenna_2(vis,cb) = unsafe_promise_clamped(ant2(vis, cb), 0, NAntennas-1);
//     solution_index(vis,d,cb) = unsafe_promise_clamped(solution_map(vis,d,cb), 0, NSolutions-1);
//     n_dir_solutions(d,cb) = unsafe_promise_clamped(n_dir_sol(d, cb), 0, NSolutions-1);

//     // PerformIteration

//     // Contribution (for add/substract direction)
//     Func sol_ann("sol_ann");
//     sol_ann(i,vis,a,d,cb) = castc<float>(sol(i,solution_index(vis,d,cb),a,cb));
//     Complex sol_ann_1_0 = sol_ann(0,vis,antenna_1(vis,cb), d, cb);
//     Complex sol_ann_1_1 = sol_ann(1,vis,antenna_1(vis,cb), d, cb);
//     Complex sol_ann_2_0 = sol_ann(0,vis,antenna_2(vis,cb), d, cb);
//     Complex sol_ann_2_1 = sol_ann(1,vis,antenna_2(vis,cb), d, cb);
//     Matrix modelM = Matrix(model(0,0,vis,d,cb), model(0,1,vis,d,cb),
//         model(1,0,vis,d,cb),model(1,1,vis,d,cb));
//     contribution(vis,d,cb) = 
//       Matrix(
//         sol_ann_1_0 * modelM.m00 * conj(sol_ann_2_0),
//         sol_ann_1_0 * modelM.m01 * conj(sol_ann_2_1),
//         sol_ann_1_1 * modelM.m10 * conj(sol_ann_2_0),
//         sol_ann_1_1 * modelM.m11 * conj(sol_ann_2_1)
//       );
    
//     // SubtractDirection
//     v_res0(vis, cb) = Matrix(
//         Complex(v_res_in(0,0,0,vis,cb), v_res_in(1,0,0,vis,cb)),
//         Complex(v_res_in(0,0,1,vis,cb), v_res_in(1,0,1,vis,cb)),
//         Complex(v_res_in(0,1,0,vis,cb), v_res_in(1,1,0,vis,cb)),
//         Complex(v_res_in(0,1,1,vis,cb), v_res_in(1,1,1,vis,cb))
//     );
//     RDom r(0, NDirections, "r");
//     v_res0(vis, cb) = Matrix(v_res0(vis, cb)) - Matrix(contribution(vis,r,cb));

//     // Add Direction
//     v_res1(vis, d, cb) = v_res0(vis,cb);
//     v_res1(vis, d, cb) = Matrix(v_res1(vis, d, cb)) + Matrix(contribution(vis,d,cb));

//     // NChannelBlocks (cb): 2
//     // Antennas (a): 50
//     // Solutions (sol): 8
//     // SolutionPolarizations (pol): 2
//     // next_solutions(cb, a, sol, pol);
//     // n_solutions_ = {1, 2, 5}

//     Func numerator("numerator"), denominator("denominator");
//     numerator(si,a,d,cb) = MatrixDiag({0.0f, 0.0f, 0.0f, 0.0f});
//     denominator(i,si,a,d,cb) = 0.0f;

//     RDom rv(0, NDirections, "rv");
//     Expr rel_solution_index = unsafe_promise_clamped(
//         solution_map(rv,d,cb) - solution_map(0,d,cb)
//         , 0, NSolutions-1);

//     Complex sol_ann_1_0_ = sol_ann(0,rv,antenna_1(rv,cb), d, cb);
//     Complex sol_ann_1_1_ = sol_ann(1,rv,antenna_1(rv,cb), d, cb);
//     Complex sol_ann_2_0_ = sol_ann(0,rv,antenna_2(rv,cb), d, cb);
//     Complex sol_ann_2_1_ = sol_ann(1,rv,antenna_2(rv,cb), d, cb);
//     Matrix modelM_ = Matrix(model(0,0,rv,d,cb), model(0,1,rv,d,cb),
//         model(1,0,rv,d,cb),model(1,1,rv,d,cb));

//     // TODO: Unsure if this is correct? It mentions "solution 1", but we index antenna 2?
//     MatrixDiag solution_1 = {sol_ann_2_0_, sol_ann_2_1_};
//     Matrix cor_model_transp_1 = solution_1 * HermTranspose(modelM_);
//     // Expr full_solution_1_index =
//     //     antenna_1(vis,cb) * n_dir_solutions + rel_solution_index;
    
//     numerator(rel_solution_index, antenna_1(rv,cb), d, cb) += Diagonal(
//         Matrix(v_res1(rv, d, cb)) * cor_model_transp_1
//     );
//     denominator(0, rel_solution_index, antenna_1(rv,cb), d, cb) += 
//         cor_model_transp_1.m00.norm() + cor_model_transp_1.m10.norm();
//     denominator(1, rel_solution_index, antenna_1(rv,cb), d, cb) += 
//         cor_model_transp_1.m01.norm() + cor_model_transp_1.m11.norm();

//     MatrixDiag solution_2 = {sol_ann_1_0_, sol_ann_1_1_};
//     Matrix cor_model_2 = solution_2 * modelM_;

//     numerator(rel_solution_index, antenna_2(rv,cb), d, cb) += Diagonal(
//         HermTranspose(Matrix(v_res1(rv, d, cb))) * cor_model_2
//     );

//     denominator(0, rel_solution_index, antenna_2(rv,cb), d, cb) += 
//         cor_model_2.m00.norm() + cor_model_2.m10.norm();
//     denominator(1, rel_solution_index, antenna_2(rv,cb), d, cb) += 
//         cor_model_2.m01.norm() + cor_model_2.m11.norm();

//     Expr nan = Expr(std::numeric_limits<double>::quiet_NaN());
//     Complex cnan = Complex(nan, nan);

//     // TODO: Check this, rd should occur somewhere?
//     RDom rd(0, NDirections, "rd");

//     Func next_solutions("next_solutions");
//     next_solutions(pol,si,a,cb) = {undef<double>(), undef<double>()};
//     next_solutions(0,si,a,cb) = tuple_select(
//         denominator(0,si,a,d,cb) == 0.0f,
//         cnan,
//         castc<double>(MatrixDiag(numerator(si,a,d,cb)).m00) / cast<double>(denominator(0,si,a,d,cb))
//     );
//     next_solutions(1,si,a,cb) = tuple_select(
//         denominator(1,si,a,d,cb) == 0.0f,
//         cnan,
//         castc<double>(MatrixDiag(numerator(si,a,d,cb)).m11) / cast<double>(denominator(1,si,a,d,cb))
//     );

//     Func result = next_solutions;
//     result.compile_to_c("iterativeDiagionalHalide.cc", 
//      {ant1,ant2,model_,sol_,solution_map,v_res_in,n_dir_sol,
//         NChannelBlocks, NDirections, NSolutions, NAntennas}, "IterativeDiagonal");

//     result.compile_to_header("iterativeDiagionalHalide.h",
//      {ant1,ant2,model_,sol_,solution_map,v_res_in,n_dir_sol,
//         NChannelBlocks, NDirections, NSolutions, NAntennas}, "IterativeDiagonal");

//     compile_standalone_runtime("iterativeDiagionalHalideRuntime.o", get_target_from_environment());

//     return 0;
// }


// int HalideDiagonalComplete() {
//     // Inputs
//     ImageParam ant1(type_of<int>(), 2, "ant1"); // <2>[cb][n_ant] int
//     ImageParam ant2(type_of<int>(), 2, "ant2"); // <2>[cb][n_ant] int
//     ImageParam solution_map(type_of<int>(), 3, "solution_map"); // <3>[cb][n_dir][n_vis] int
//     ImageParam v_res_in(type_of<float>(), 5, "v_res_in"); // <5>[cb][n_vis], Complex 2x2 Float (3)
//     ImageParam model_(type_of<float>(), 6, "model_"); // <6>[cb][dir][n_vis], Complex 2x2 Float (3)
//     ImageParam sol_(type_of<double>(), 5, "sol_");
//     ImageParam n_dir_sol(type_of<int>(), 2, "n_dir_sol"); // <2>[cb][dir] int
//     ImageParam n_vis(type_of<int>(), 2, "n_vis"); // <2>[cb][dir] int

//     Func v_res0("v_res0"), v_res1("v_res1");
//     Func model("model"), sol("sol");
//     Var x("x"), y("y"), i("i"), j("j"), vis("vis"), d("d"), si("si"), a("a"), cb("cb"), pol("pol");

//     Func contribution("contribution");

//     // Func next_solutions("next_solutions");
//     //   {NChannelBlocks(), NAntennas(), NSolutions(), NSolutionPolarizations()});
//     Param<int> NChannelBlocks;
//     Param<int> NDirections;// 3;
//     Param<int>  NSolutions;// 8
//     Param<int> NAntennas;; //50;
//     model(i, j, vis, d, cb) = Tuple(model_(0, i, j, vis, d, cb), model_(1, i, j, vis, d, cb));
//     sol(i, si, a, cb) = Tuple(sol_(0, i, si, a, cb), sol_(1, i, si, a, cb));

//     Func antenna_1("antenna_1"), antenna_2("antenna_2"), solution_index("solution_index"), n_dir_solutions("n_dir_solutions");
//     antenna_1(vis,cb) = unsafe_promise_clamped(ant1(vis, cb), 0, NAntennas-1);
//     antenna_2(vis,cb) = unsafe_promise_clamped(ant2(vis, cb), 0, NAntennas-1);
//     solution_index(vis,d,cb) = unsafe_promise_clamped(solution_map(vis,d,cb), 0, NSolutions-1);
//     n_dir_solutions(d,cb) = unsafe_promise_clamped(n_dir_sol(d, cb), 0, NSolutions-1);

//     // PerformIteration

//     // Contribution (for add/substract direction)
//     Func sol_ann("sol_ann");
//     sol_ann(i,vis,a,d,cb) = castc<float>(sol(i,solution_index(vis,d,cb),a,cb));
//     Complex sol_ann_1_0 = sol_ann(0,vis,antenna_1(vis,cb), d, cb);
//     Complex sol_ann_1_1 = sol_ann(1,vis,antenna_1(vis,cb), d, cb);
//     Complex sol_ann_2_0 = sol_ann(0,vis,antenna_2(vis,cb), d, cb);
//     Complex sol_ann_2_1 = sol_ann(1,vis,antenna_2(vis,cb), d, cb);
//     Matrix modelM = Matrix(model(0,0,vis,d,cb), model(0,1,vis,d,cb),
//         model(1,0,vis,d,cb),model(1,1,vis,d,cb));
//     contribution(vis,d,cb) = 
//       Matrix(
//         sol_ann_1_0 * modelM.m00 * conj(sol_ann_2_0),
//         sol_ann_1_0 * modelM.m01 * conj(sol_ann_2_1),
//         sol_ann_1_1 * modelM.m10 * conj(sol_ann_2_0),
//         sol_ann_1_1 * modelM.m11 * conj(sol_ann_2_1)
//       );
    
//     // SubtractDirection
//     v_res0(vis, cb) = Matrix(
//         Complex(v_res_in(0,0,0,vis,cb), v_res_in(1,0,0,vis,cb)),
//         Complex(v_res_in(0,0,1,vis,cb), v_res_in(1,0,1,vis,cb)),
//         Complex(v_res_in(0,1,0,vis,cb), v_res_in(1,1,0,vis,cb)),
//         Complex(v_res_in(0,1,1,vis,cb), v_res_in(1,1,1,vis,cb))
//     );
//     RDom r(0, NDirections, "r");
//     v_res0(vis, cb) = Matrix(v_res0(vis, cb)) - Matrix(contribution(vis,r,cb));

//     // Add Direction
//     v_res1(vis, d, cb) = v_res0(vis,cb);
//     v_res1(vis, d, cb) = Matrix(v_res1(vis, d, cb)) + Matrix(contribution(vis,d,cb));

//     // NChannelBlocks (cb): 2
//     // Antennas (a): 50
//     // Solutions (sol): 8
//     // SolutionPolarizations (pol): 2
//     // next_solutions(cb, a, sol, pol);
//     // n_solutions_ = {1, 2, 5}

//     Func numerator("numerator"), denominator("denominator");
//     numerator(si,a,d,cb) = MatrixDiag({0.0f, 0.0f, 0.0f, 0.0f});
//     denominator(i,si,a,d,cb) = 0.0f;

//     RDom rv(0, NDirections, "rv");
//     Expr rel_solution_index = unsafe_promise_clamped(
//         solution_map(rv,d,cb) - solution_map(0,d,cb)
//         , 0, NSolutions-1);

//     Complex sol_ann_1_0_ = sol_ann(0,rv,antenna_1(rv,cb), d, cb);
//     Complex sol_ann_1_1_ = sol_ann(1,rv,antenna_1(rv,cb), d, cb);
//     Complex sol_ann_2_0_ = sol_ann(0,rv,antenna_2(rv,cb), d, cb);
//     Complex sol_ann_2_1_ = sol_ann(1,rv,antenna_2(rv,cb), d, cb);
//     Matrix modelM_ = Matrix(model(0,0,rv,d,cb), model(0,1,rv,d,cb),
//         model(1,0,rv,d,cb),model(1,1,rv,d,cb));

//     // TODO: Unsure if this is correct? It mentions "solution 1", but we index antenna 2?
//     MatrixDiag solution_1 = {sol_ann_2_0_, sol_ann_2_1_};
//     Matrix cor_model_transp_1 = solution_1 * HermTranspose(modelM_);
//     // Expr full_solution_1_index =
//     //     antenna_1(vis,cb) * n_dir_solutions + rel_solution_index;
    
//     numerator(rel_solution_index, antenna_1(rv,cb), d, cb) += Diagonal(
//         Matrix(v_res1(rv, d, cb)) * cor_model_transp_1
//     );
//     denominator(0, rel_solution_index, antenna_1(rv,cb), d, cb) += 
//         cor_model_transp_1.m00.norm() + cor_model_transp_1.m10.norm();
//     denominator(1, rel_solution_index, antenna_1(rv,cb), d, cb) += 
//         cor_model_transp_1.m01.norm() + cor_model_transp_1.m11.norm();

//     MatrixDiag solution_2 = {sol_ann_1_0_, sol_ann_1_1_};
//     Matrix cor_model_2 = solution_2 * modelM_;

//     numerator(rel_solution_index, antenna_2(rv,cb), d, cb) += Diagonal(
//         HermTranspose(Matrix(v_res1(rv, d, cb))) * cor_model_2
//     );

//     denominator(0, rel_solution_index, antenna_2(rv,cb), d, cb) += 
//         cor_model_2.m00.norm() + cor_model_2.m10.norm();
//     denominator(1, rel_solution_index, antenna_2(rv,cb), d, cb) += 
//         cor_model_2.m01.norm() + cor_model_2.m11.norm();

//     Expr nan = Expr(std::numeric_limits<double>::quiet_NaN());
//     Complex cnan = Complex(nan, nan);

//     // TODO: Check this, rd should occur somewhere?
//     RDom rd(0, NDirections, "rd");

//     Func next_solutions("next_solutions");
//     next_solutions(pol,si,a,cb) = {undef<double>(), undef<double>()};
//     next_solutions(0,si,a,cb) = tuple_select(
//         denominator(0,si,a,d,cb) == 0.0f,
//         cnan,
//         castc<double>(MatrixDiag(numerator(si,a,d,cb)).m00) / cast<double>(denominator(0,si,a,d,cb))
//     );
//     next_solutions(1,si,a,cb) = tuple_select(
//         denominator(1,si,a,d,cb) == 0.0f,
//         cnan,
//         castc<double>(MatrixDiag(numerator(si,a,d,cb)).m11) / cast<double>(denominator(1,si,a,d,cb))
//     );

//     Func result = next_solutions;
//     result.compile_to_c("iterativeDiagionalHalide.cc", 
//      {ant1,ant2,model_,sol_,solution_map,v_res_in,n_dir_sol,
//         NChannelBlocks, NDirections, NSolutions, NAntennas}, "IterativeDiagonal");

//     result.compile_to_header("iterativeDiagionalHalide.h",
//      {ant1,ant2,model_,sol_,solution_map,v_res_in,n_dir_sol,
//         NChannelBlocks, NDirections, NSolutions, NAntennas}, "IterativeDiagonal");

//     compile_standalone_runtime("iterativeDiagionalHalideRuntime.o", get_target_from_environment());

//     return 0;
// }


// Func AddOrSubtractDirection(bool add){
//     // Inputs
//     ImageParam ant1(type_of<int>(), 1, "ant1"); // <1>[n_ant] int
//     ImageParam ant2(type_of<int>(), 1, "ant2"); // <1>[n_ant] int
//     ImageParam solution_map(type_of<int>(), 1, "solution_map"); // <1>[n_vis] int
//     ImageParam v_res_in(type_of<float>(), 4, "v_res_in"); // <4>[n_vis], Complex 2x2 Float (+3)
//     ImageParam model_(type_of<float>(), 4, "model_"); // <4>[n_vis], Complex 2x2 Float (+3)
//     ImageParam sol_(type_of<double>(), 4, "sol_"); // <4> [n_ant][n_dir_sol][2] Complex Double (+1)
//     Param<int> n_dir_sol("n_dir_sol");
//     Param<int> n_vis("n_vis");

//     Func v_res0("v_res0"), v_res1("v_res1");
//     Func model("model"), sol("sol");
//     Var x("x"), y("y"), i("i"), j("j"), vis("vis"), si("si"), a("a"), pol("pol");

//     Func contribution("contribution");

//     Param<int> NAntennas;; //50;
//     model(i, j, vis) = Tuple(model_(0, i, j, vis), model_(1, i, j, vis));
//     sol(i, si, a) = Tuple(sol_(0, i, si, a), sol_(1, i, si, a));

//     Func antenna_1("antenna_1"), antenna_2("antenna_2"), solution_index("solution_index");
//     antenna_1(vis) = unsafe_promise_clamped(ant1(vis), 0, NAntennas-1);
//     antenna_2(vis) = unsafe_promise_clamped(ant2(vis), 0, NAntennas-1);
//     solution_index(vis) = unsafe_promise_clamped(solution_map(vis), 0, n_dir_sol-1);

//     // PerformIteration

//     // Contribution (for add/substract direction)
//     Func sol_ann("sol_ann");
//     sol_ann(i,vis,a) = castc<float>(sol(i,solution_index(vis),a));
//     Complex sol_ann_1_0 = sol_ann(0,vis,antenna_1(vis));
//     Complex sol_ann_1_1 = sol_ann(1,vis,antenna_1(vis));
//     Complex sol_ann_2_0 = sol_ann(0,vis,antenna_2(vis));
//     Complex sol_ann_2_1 = sol_ann(1,vis,antenna_2(vis));
//     Matrix modelM = Matrix(model(0,0,vis), model(0,1,vis),
//         model(1,0,vis),model(1,1,vis));
//     contribution(vis) = 
//       Matrix(
//         sol_ann_1_0 * modelM.m00 * conj(sol_ann_2_0),
//         sol_ann_1_0 * modelM.m01 * conj(sol_ann_2_1),
//         sol_ann_1_1 * modelM.m10 * conj(sol_ann_2_0),
//         sol_ann_1_1 * modelM.m11 * conj(sol_ann_2_1)
//       );
    
//     // SubtractDirection
//     v_res0(vis) = Matrix(
//         Complex(v_res_in(0,0,0,vis), v_res_in(1,0,0,vis)),
//         Complex(v_res_in(0,0,1,vis), v_res_in(1,0,1,vis)),
//         Complex(v_res_in(0,1,0,vis), v_res_in(1,1,0,vis)),
//         Complex(v_res_in(0,1,1,vis), v_res_in(1,1,1,vis))
//     );
//     RDom r(0, n_dir_sol, "r");
//     v_res0(vis) = Matrix(v_res0(vis)) - Matrix(contribution(vis,r));

//     // Add Direction
//     v_res1(vis) = v_res0(vis);
//     v_res1(vis) = Matrix(v_res1(vis)) + Matrix(contribution(vis));

// }

// int HalideDiagonalPartial(){
//     // Inputs
//     ImageParam ant1(type_of<int>(), 1, "ant1"); // <1>[n_ant] int
//     ImageParam ant2(type_of<int>(), 1, "ant2"); // <1>[n_ant] int
//     ImageParam solution_map(type_of<int>(), 1, "solution_map"); // <1>[n_vis] int
//     ImageParam v_res_in(type_of<float>(), 4, "v_res_in"); // <4>[n_vis], Complex 2x2 Float (+3)
//     ImageParam model_(type_of<float>(), 4, "model_"); // <4>[n_vis], Complex 2x2 Float (+3)
//     ImageParam sol_(type_of<double>(), 4, "sol_"); // <4> [n_ant][n_dir_sol][2] Complex Double (+1)
//     Param<int> n_dir_sol("n_dir_sol");
//     Param<int> n_vis("n_vis");

//     Param<int> n_visibilities;
//     Param<int> NSolutions;// 8
//     Param<int> NAntennas;; //50;

//     Func v_res0("v_res0"), v_res1("v_res1");
//     Func model("model"), sol("sol");
//     Var x("x"), y("y"), i("i"), j("j"), vis("vis"), si("si"), a("a"), pol("pol");

//     Func contribution("contribution");

//     model(i, j, vis) = Tuple(model_(0, i, j, vis), model_(1, i, j, vis));
//     sol(i, si, a) = Tuple(sol_(0, i, si, a), sol_(1, i, si, a));

//     Func antenna_1("antenna_1"), antenna_2("antenna_2"), solution_index("solution_index");
//     antenna_1(vis) = unsafe_promise_clamped(ant1(vis), 0, NAntennas-1);
//     antenna_2(vis) = unsafe_promise_clamped(ant2(vis), 0, NAntennas-1);
//     solution_index(vis) = unsafe_promise_clamped(solution_map(vis), 0, n_dir_sol-1);

//     // PerformIteration

//     // Contribution (for add/substract direction)
//     Func sol_ann("sol_ann");
//     sol_ann(i,vis,a) = castc<float>(sol(i,solution_index(vis),a));
//     Complex sol_ann_1_0 = sol_ann(0,vis,antenna_1(vis));
//     Complex sol_ann_1_1 = sol_ann(1,vis,antenna_1(vis));
//     Complex sol_ann_2_0 = sol_ann(0,vis,antenna_2(vis));
//     Complex sol_ann_2_1 = sol_ann(1,vis,antenna_2(vis));
//     Matrix modelM = Matrix(model(0,0,vis), model(0,1,vis),
//         model(1,0,vis),model(1,1,vis));
//     contribution(vis) = 
//       Matrix(
//         sol_ann_1_0 * modelM.m00 * conj(sol_ann_2_0),
//         sol_ann_1_0 * modelM.m01 * conj(sol_ann_2_1),
//         sol_ann_1_1 * modelM.m10 * conj(sol_ann_2_0),
//         sol_ann_1_1 * modelM.m11 * conj(sol_ann_2_1)
//       );
    
//     // SubtractDirection
//     v_res0(vis) = Matrix(
//         Complex(v_res_in(0,0,0,vis), v_res_in(1,0,0,vis)),
//         Complex(v_res_in(0,0,1,vis), v_res_in(1,0,1,vis)),
//         Complex(v_res_in(0,1,0,vis), v_res_in(1,1,0,vis)),
//         Complex(v_res_in(0,1,1,vis), v_res_in(1,1,1,vis))
//     );
//     // RDom r(0, n_dir_sol, "r");
//     v_res0(vis) = Matrix(v_res0(vis)) - Matrix(contribution(vis));

//     // Add Direction
//     v_res1(vis) = v_res0(vis);
//     v_res1(vis) = Matrix(v_res1(vis)) + Matrix(contribution(vis));

//     // NChannelBlocks (cb): 2
//     // Antennas (a): 50
//     // Solutions (sol): 8
//     // SolutionPolarizations (pol): 2
//     // next_solutions(cb, a, sol, pol);
//     // n_solutions_ = {1, 2, 5}

//     Func numerator("numerator"), denominator("denominator");
//     numerator(si,a) = MatrixDiag({0.0f, 0.0f, 0.0f, 0.0f});
//     denominator(i,si,a) = 0.0f;

//     RDom rv(0, n_visibilities, "rv");
//     Expr rel_solution_index = unsafe_promise_clamped(
//         solution_map(rv) - solution_map(0)
//         , 0, NSolutions-1);

//     Complex sol_ann_1_0_ = sol_ann(0,rv,antenna_1(rv));
//     Complex sol_ann_1_1_ = sol_ann(1,rv,antenna_1(rv));
//     Complex sol_ann_2_0_ = sol_ann(0,rv,antenna_2(rv));
//     Complex sol_ann_2_1_ = sol_ann(1,rv,antenna_2(rv));
//     Matrix modelM_ = Matrix(model(0,0,rv), model(0,1,rv),
//         model(1,0,rv),model(1,1,rv));

//     // TODO: Unsure if this is correct? It mentions "solution 1", but we index antenna 2?
//     MatrixDiag solution_1 = {sol_ann_2_0_, sol_ann_2_1_};
//     Matrix cor_model_transp_1 = solution_1 * HermTranspose(modelM_);
//     // Expr full_solution_1_index =
//     //     antenna_1(vis,cb) * n_dir_solutions + rel_solution_index;
    
//     numerator(rel_solution_index, antenna_1(rv)) += Diagonal(
//         Matrix(v_res1(rv)) * cor_model_transp_1
//     );
//     denominator(0, rel_solution_index, antenna_1(rv)) += 
//         cor_model_transp_1.m00.norm() + cor_model_transp_1.m10.norm();
//     denominator(1, rel_solution_index, antenna_1(rv)) += 
//         cor_model_transp_1.m01.norm() + cor_model_transp_1.m11.norm();

//     MatrixDiag solution_2 = {sol_ann_1_0_, sol_ann_1_1_};
//     Matrix cor_model_2 = solution_2 * modelM_;

//     numerator(rel_solution_index, antenna_2(rv)) += Diagonal(
//         HermTranspose(Matrix(v_res1(rv))) * cor_model_2
//     );

//     denominator(0, rel_solution_index, antenna_2(rv)) += 
//         cor_model_2.m00.norm() + cor_model_2.m10.norm();
//     denominator(1, rel_solution_index, antenna_2(rv)) += 
//         cor_model_2.m01.norm() + cor_model_2.m11.norm();

//     Expr nan = Expr(std::numeric_limits<double>::quiet_NaN());
//     Complex cnan = Complex(nan, nan);

//     // TODO: Check this, rd should occur somewhere?
//     RDom rd(0, NSolutions, "rd");

//     Func next_solutions("next_solutions");
//     next_solutions(pol,si,a) = {undef<double>(), undef<double>()};
//     next_solutions(0,si,a) = tuple_select(
//         denominator(0,si,a) == 0.0f,
//         cnan,
//         castc<double>(MatrixDiag(numerator(si,a)).m00) / cast<double>(denominator(0,si,a))
//     );
//     next_solutions(1,si,a) = tuple_select(
//         denominator(1,si,a) == 0.0f,
//         cnan,
//         castc<double>(MatrixDiag(numerator(si,a)).m11) / cast<double>(denominator(1,si,a))
//     );

//     Func result = next_solutions;
//     // Func result("result");
//     // ImageParam inp(type_of<int>(), 1, "inp");
//     // result(x) = inp(x)+1;
//     // std::vector<Argument> args = {inp};

//     std::vector<Argument> args = {ant1,ant2,model_,sol_,solution_map,v_res_in,n_dir_sol,
//         n_visibilities, NSolutions, NAntennas};

//     result.compile_to_c("iterativeDiagionalHalide.cc", args, "IterativeDiagonal");

//     result.compile_to_static_library("iterativeDiagionalHalide", args, "IterativeDiagonal");

//     compile_standalone_runtime("iterativeDiagionalHalideRuntime.o", get_target_from_environment());

//     return 0;
// }