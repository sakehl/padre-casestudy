
/* MACHINE GENERATED By Halide. */

#include <assert.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <stdlib.h>

#ifndef HALIVER_GLOBALS
#define HALIVER_GLOBALS

struct halide_dimension_t {
    int32_t min, extent, stride;
};

inline void halide_unused(bool e){};

/*@
pure int max(int x, int y) = x > y ? x : y;

pure int min(int x, int y) = x > y ? y : x;

pure float max(float x, float y) = x > y ? x : y;

pure float min(float x, float y) = x > y ? y : x;

pure int abs(int x) = x >= 0 ? x : -x;

pure float abs(float x) = x >= 0 ? x : -x;

// Euclidean division is defined internally in VerCors
pure int hdiv(int x, int y) = y == 0 ? 0 : \euclidean_div(x, y);
pure int hmod(int x, int y) = y == 0 ? 0 : \euclidean_mod(x, y);
@*/

/*@
  requires y != 0;
  ensures \result == \euclidean_div(x, y);
@*/
inline int /*@ pure @*/ div_eucl(int x, int y)
{
    int q = x/y;
    int r = x%y;
    return r < 0 ? q + (y > 0 ? -1 : 1) : q;
}

/*@
  requires y != 0;
  ensures \result == \euclidean_mod(x, y);
@*/
inline int /*@ pure @*/ mod_eucl(int x, int y)
{
    int r = x%y;
    return (x >= 0 || r == 0) ? r : r + abs(y);
}

static inline int /*@ pure @*/ min(int x, int y) {return x < y ? x : y;}

static inline float /*@ pure @*/ fast_inverse_f32(float x) {return 1.0f/x;}
static inline float /*@ pure @*/ sqrt_f32(double x) {return (float)sqrt((double)x);}
static inline float /*@ pure @*/ pow_f32(float x, float y){ return (float) pow((double) x, (double) y);}
static inline float /*@ pure @*/ floor_f32(float x){ return (float) floor((double) x); }
static inline float /*@ pure @*/ ceil_f32(float x){ return (float) ceil((double) x); }
static inline float /*@ pure @*/ round_f32(float x){ return (float) round((double) x); }

static inline double /*@ pure @*/ sqrt_f64(double x) {return sqrt(x);}
static inline double /*@ pure @*/ pow_f64(double x, double y) {return pow(x, y);}
static inline double /*@ pure @*/ floor_f64(double x) {return floor(x);}
static inline double /*@ pure @*/ ceil_f64(double x) {return ceil(x);}
static inline double /*@ pure @*/ round_f64(double x){ return round(x); }

//inline float nan_f32() {return NAN;}
inline float nan_f32() {return 0.0f;}
/*@
inline resource dim_perm(struct halide_dimension_t *dim, rational p, int i) = 
 Perm(&dim[i], 1\2) **
 Perm(dim[i].min, 1\2) **
 Perm(dim[i].stride, 1\2) **
 Perm(dim[i].extent, 1\2)
 ;

 ghost
 requires a >= 0;
 requires b > 0;
 requires a < max_a;
 ensures a*b <= (max_a-1)*b;
 ensures \result;
 decreases b;
pure bool lemma_nonlinear(int a, int b, int max_a) = 
  b>1 ? lemma_nonlinear(a, b-1, max_a) : true;
// {
//  if(b>1){
//    lemma_nonlinear(a, b-1, max_a);
//    assert a*(b-1) <= max_a*(b-1);
//  }
//}

 ghost
 requires a-min_a >= 0  && a-min_a<extent_a;
 requires b-min_b >= 0 && b-min_b<extent_b;
 requires stride_a > 0;
 requires stride_b > 0;
 requires stride_b >= extent_a*stride_a;

 ensures 0 <= (a-min_a)*stride_a + (b-min_b)*stride_b;
 ensures (a-min_a)*stride_a + (b-min_b)*stride_b < stride_b*extent_b;
 ensures \result;
 decreases;
pure bool lemma_2d_access(
 int a, int min_a, int stride_a, int extent_a,
 int b, int min_b, int stride_b, int extent_b) =
  lemma_nonlinear(a-min_a, stride_a, extent_a) &&
  lemma_nonlinear(b-min_b, stride_b, extent_b);
//{
//  lemma_nonlinear(a-min_a, stride_a, extent_a);
//  lemma_nonlinear(b-min_b, stride_b, extent_b);
//  return;
//}

inline pure bool req_3d(int a, int min_a, int stride_a, int extent_a,
  int b, int min_b, int stride_b, int extent_b,
  int c, int min_c, int stride_c, int extent_c) = 
    a-min_a >= 0 && a-min_a < extent_a &&
    b-min_b >= 0 && b-min_b < extent_b && 
    c-min_c >= 0 && c-min_c < extent_c && 
    stride_a > 0 && 
    stride_b > 0 && 
    stride_c > 0 && 
    stride_b >= extent_a * stride_a && 
    stride_c >= extent_b * stride_b;

 
 requires a-min_a >= 0 && a-min_a < extent_a;
 requires b-min_b >= 0 && b-min_b < extent_b;
 requires c-min_c >= 0 && c-min_c < extent_c;
 requires stride_a > 0;
 requires stride_b > 0;
 requires stride_c > 0;
 requires stride_b >= extent_a * stride_a;
 requires stride_c >= extent_b * stride_b;
 
 ensures 0 <= (a-min_a) * stride_a + (b-min_b) * stride_b + (c-min_c) * stride_c;
 ensures (a-min_a) * stride_a + (b-min_b) * stride_b + (c-min_c) * stride_c < stride_c * extent_c;
 ensures \result;
 decreases;
pure bool lemma_3d_access(
  int a, int min_a, int stride_a, int extent_a,
  int b, int min_b, int stride_b, int extent_b,
  int c, int min_c, int stride_c, int extent_c
)
; // = lemma_2d_access(a, min_a, stride_a, extent_a, b, min_b, stride_b, extent_b) && lemma_nonlinear(c-min_c, stride_c, extent_c);

 ghost
 requires a-min_a >= 0 && a-min_a < extent_a;
 requires b-min_b >= 0 && b-min_b < extent_b;
 requires c-min_c >= 0 && c-min_c < extent_c;
 requires d-min_d >= 0 && d-min_d < extent_d;
 requires stride_a > 0;
 requires stride_b > 0;
 requires stride_c > 0;
 requires stride_d > 0;
 requires stride_b >= extent_a * stride_a;
 requires stride_c >= extent_b * stride_b;
 requires stride_d >= extent_c * stride_c;

ensures 0 <= (a-min_a) * stride_a + (b-min_b) * stride_b + (c-min_c) * stride_c + (d-min_d) * stride_d;
ensures (a-min_a) * stride_a + (b-min_b) * stride_b + (c-min_c) * stride_c + (d-min_d) * stride_d < stride_d * extent_d;
decreases;
pure bool lemma_4d_access(
  int a, int min_a, int stride_a, int extent_a,
  int b, int min_b, int stride_b, int extent_b,
  int c, int min_c, int stride_c, int extent_c,
  int d, int min_d, int stride_d, int extent_d
)
;//{
//  lemma_3d_access(a, min_a, stride_a, extent_a, b, min_b, stride_b, extent_b, c, min_c, stride_c, extent_c);
//  lemma_nonlinear(d-min_d, stride_d, extent_d);
//  return;
//}

 ghost
 requires a-min_a >= 0 && a-min_a < extent_a;
 requires b-min_b >= 0 && b-min_b < extent_b;
 requires c-min_c >= 0 && c-min_c < extent_c;
 requires d-min_d >= 0 && d-min_d < extent_d;
 requires e-min_e >= 0 && e-min_e < extent_e;
 requires stride_a > 0;
 requires stride_b > 0;
 requires stride_c > 0;
 requires stride_d > 0;
 requires stride_e > 0;
 requires stride_b >= extent_a * stride_a;
 requires stride_c >= extent_b * stride_b;
 requires stride_d >= extent_c * stride_c;
 requires stride_e >= extent_d * stride_d;
 
 ensures 0 <= (a-min_a) * stride_a + (b-min_b) * stride_b + (c-min_c) * stride_c + (d-min_d) * stride_d + (e-min_e) * stride_e;
 ensures (a-min_a) * stride_a + (b-min_b) * stride_b + (c-min_c) * stride_c + (d-min_d) * stride_d + (e-min_e) * stride_e < stride_e * extent_e;
 decreases;
pure bool lemma_5d_access(
  int a, int min_a, int stride_a, int extent_a,
  int b, int min_b, int stride_b, int extent_b,
  int c, int min_c, int stride_c, int extent_c,
  int d, int min_d, int stride_d, int extent_d,
  int e, int min_e, int stride_e, int extent_e
)
;//{
//  lemma_4d_access(a, min_a, stride_a, extent_a, b, min_b, stride_b, extent_b, c, min_c, stride_c, extent_c, d, min_d, stride_d, extent_d);
//  lemma_nonlinear(e-min_e, stride_e, extent_e);
//  return;
//}
@*/
#endif // HALIVER_GLOBALS

#ifndef HALIDE_BUFFER_TYPE_DOUBLE
#define HALIDE_BUFFER_TYPE_DOUBLE
struct halide_buffer_double {

    /** The dimensionality of the buffer. */
    int32_t dimensions;

    /** The shape of the buffer. Halide does not own this array - you
     * must manage the memory for it yourself. */
    struct halide_dimension_t *dim;

    /** A pointer to the start of the data in main memory. In terms of
     * the Halide coordinate system, this is the address of the min
     * coordinates (defined below). */

    double *host;
};

/*@ 
 requires buf != NULL ** \pointer_length(buf) == 1 ** Perm(buf, 1\2);
 requires Perm(buf->host, 1\2);
 @*/
/*@ pure @*/ inline double *_halide_buffer_get_host_double(struct halide_buffer_double *buf) {
    return buf->host;
}

/*@ 
    requires buf != NULL ** \pointer_length(buf) == 1 ** Perm(buf, 1\2);
    requires Perm(buf->dim, 1\2) ** buf->dim != NULL;
    requires 0 <= d && d < \pointer_length(buf->dim);
    requires Perm(&buf->dim[d], 1\2);
    requires Perm(buf->dim[d].min, 1\2);
@*/
/*@ pure @*/ inline int _halide_buffer_get_min_double(struct halide_buffer_double *buf, int d) {
    return buf->dim[d].min;
}

/*@ 
    requires buf != NULL ** \pointer_length(buf) == 1 ** Perm(buf, 1\2);
    requires Perm(buf->dim, 1\2) ** buf->dim != NULL;
    requires 0 <= d && d < \pointer_length(buf->dim);
    requires Perm(&buf->dim[d], 1\2);
    requires Perm(buf->dim[d].min, 1\2) ** Perm(buf->dim[d].extent, 1\2);
@*/
/*@ pure @*/ inline int _halide_buffer_get_max_double(struct halide_buffer_double *buf, int d) {
    return buf->dim[d].min + buf->dim[d].extent - 1;
}

/*@ 
    requires buf != NULL ** \pointer_length(buf) == 1 ** Perm(buf, 1\2);
    requires Perm(buf->dim, 1\2) ** buf->dim != NULL;
    requires 0 <= d && d < \pointer_length(buf->dim);
    requires Perm(&buf->dim[d], 1\2);
    requires Perm(buf->dim[d].extent, 1\2);
@*/
/*@ pure @*/ inline int _halide_buffer_get_extent_double(struct halide_buffer_double *buf, int d) {
    return buf->dim[d].extent;
}

/*@ 
    requires buf != NULL ** \pointer_length(buf) == 1 ** Perm(buf, 1\2);
    requires Perm(buf->dim, 1\2) ** buf->dim != NULL;
    requires 0 <= d && d < \pointer_length(buf->dim);
    requires Perm(&buf->dim[d], 1\2);
    requires Perm(buf->dim[d].stride, 1\2);
@*/
/*@ pure @*/ inline int _halide_buffer_get_stride_double(struct halide_buffer_double *buf, int d) {
    return buf->dim[d].stride;
}

/*@
inline resource buffer_double(struct halide_buffer_double *buf, rational p, int n_dims) = 
 buf != NULL **
 \pointer_length(buf) == 1 **
 Perm(buf, p) **
 Perm(buf->dim, p) **
 buf->dim != NULL **
 \pointer_length(buf->dim) == n_dims **
 Perm(buf->host, p) **
 buf->host != NULL;
@*/
#endif //HALIDE_BUFFER_TYPE_DOUBLE
#ifndef HALIDE_BUFFER_TYPE_INT32_T
#define HALIDE_BUFFER_TYPE_INT32_T
struct halide_buffer_int32_t {

    /** The dimensionality of the buffer. */
    int32_t dimensions;

    /** The shape of the buffer. Halide does not own this array - you
     * must manage the memory for it yourself. */
    struct halide_dimension_t *dim;

    /** A pointer to the start of the data in main memory. In terms of
     * the Halide coordinate system, this is the address of the min
     * coordinates (defined below). */

    int32_t *host;
};

/*@ 
 requires buf != NULL ** \pointer_length(buf) == 1 ** Perm(buf, 1\2);
 requires Perm(buf->host, 1\2);
 @*/
/*@ pure @*/ inline int32_t *_halide_buffer_get_host_int32_t(struct halide_buffer_int32_t *buf) {
    return buf->host;
}

/*@ 
    requires buf != NULL ** \pointer_length(buf) == 1 ** Perm(buf, 1\2);
    requires Perm(buf->dim, 1\2) ** buf->dim != NULL;
    requires 0 <= d && d < \pointer_length(buf->dim);
    requires Perm(&buf->dim[d], 1\2);
    requires Perm(buf->dim[d].min, 1\2);
@*/
/*@ pure @*/ inline int _halide_buffer_get_min_int32_t(struct halide_buffer_int32_t *buf, int d) {
    return buf->dim[d].min;
}

/*@ 
    requires buf != NULL ** \pointer_length(buf) == 1 ** Perm(buf, 1\2);
    requires Perm(buf->dim, 1\2) ** buf->dim != NULL;
    requires 0 <= d && d < \pointer_length(buf->dim);
    requires Perm(&buf->dim[d], 1\2);
    requires Perm(buf->dim[d].min, 1\2) ** Perm(buf->dim[d].extent, 1\2);
@*/
/*@ pure @*/ inline int _halide_buffer_get_max_int32_t(struct halide_buffer_int32_t *buf, int d) {
    return buf->dim[d].min + buf->dim[d].extent - 1;
}

/*@ 
    requires buf != NULL ** \pointer_length(buf) == 1 ** Perm(buf, 1\2);
    requires Perm(buf->dim, 1\2) ** buf->dim != NULL;
    requires 0 <= d && d < \pointer_length(buf->dim);
    requires Perm(&buf->dim[d], 1\2);
    requires Perm(buf->dim[d].extent, 1\2);
@*/
/*@ pure @*/ inline int _halide_buffer_get_extent_int32_t(struct halide_buffer_int32_t *buf, int d) {
    return buf->dim[d].extent;
}

/*@ 
    requires buf != NULL ** \pointer_length(buf) == 1 ** Perm(buf, 1\2);
    requires Perm(buf->dim, 1\2) ** buf->dim != NULL;
    requires 0 <= d && d < \pointer_length(buf->dim);
    requires Perm(&buf->dim[d], 1\2);
    requires Perm(buf->dim[d].stride, 1\2);
@*/
/*@ pure @*/ inline int _halide_buffer_get_stride_int32_t(struct halide_buffer_int32_t *buf, int d) {
    return buf->dim[d].stride;
}

/*@
inline resource buffer_int32_t(struct halide_buffer_int32_t *buf, rational p, int n_dims) = 
 buf != NULL **
 \pointer_length(buf) == 1 **
 Perm(buf, p) **
 Perm(buf->dim, p) **
 buf->dim != NULL **
 \pointer_length(buf->dim) == n_dims **
 Perm(buf->host, p) **
 buf->host != NULL;
@*/
#endif //HALIDE_BUFFER_TYPE_INT32_T
#ifndef HALIDE_BUFFER_TYPE_FLOAT
#define HALIDE_BUFFER_TYPE_FLOAT
struct halide_buffer_float {

    /** The dimensionality of the buffer. */
    int32_t dimensions;

    /** The shape of the buffer. Halide does not own this array - you
     * must manage the memory for it yourself. */
    struct halide_dimension_t *dim;

    /** A pointer to the start of the data in main memory. In terms of
     * the Halide coordinate system, this is the address of the min
     * coordinates (defined below). */

    float *host;
};

/*@ 
 requires buf != NULL ** \pointer_length(buf) == 1 ** Perm(buf, 1\2);
 requires Perm(buf->host, 1\2);
 @*/
/*@ pure @*/ inline float *_halide_buffer_get_host_float(struct halide_buffer_float *buf) {
    return buf->host;
}

/*@ 
    requires buf != NULL ** \pointer_length(buf) == 1 ** Perm(buf, 1\2);
    requires Perm(buf->dim, 1\2) ** buf->dim != NULL;
    requires 0 <= d && d < \pointer_length(buf->dim);
    requires Perm(&buf->dim[d], 1\2);
    requires Perm(buf->dim[d].min, 1\2);
@*/
/*@ pure @*/ inline int _halide_buffer_get_min_float(struct halide_buffer_float *buf, int d) {
    return buf->dim[d].min;
}

/*@ 
    requires buf != NULL ** \pointer_length(buf) == 1 ** Perm(buf, 1\2);
    requires Perm(buf->dim, 1\2) ** buf->dim != NULL;
    requires 0 <= d && d < \pointer_length(buf->dim);
    requires Perm(&buf->dim[d], 1\2);
    requires Perm(buf->dim[d].min, 1\2) ** Perm(buf->dim[d].extent, 1\2);
@*/
/*@ pure @*/ inline int _halide_buffer_get_max_float(struct halide_buffer_float *buf, int d) {
    return buf->dim[d].min + buf->dim[d].extent - 1;
}

/*@ 
    requires buf != NULL ** \pointer_length(buf) == 1 ** Perm(buf, 1\2);
    requires Perm(buf->dim, 1\2) ** buf->dim != NULL;
    requires 0 <= d && d < \pointer_length(buf->dim);
    requires Perm(&buf->dim[d], 1\2);
    requires Perm(buf->dim[d].extent, 1\2);
@*/
/*@ pure @*/ inline int _halide_buffer_get_extent_float(struct halide_buffer_float *buf, int d) {
    return buf->dim[d].extent;
}

/*@ 
    requires buf != NULL ** \pointer_length(buf) == 1 ** Perm(buf, 1\2);
    requires Perm(buf->dim, 1\2) ** buf->dim != NULL;
    requires 0 <= d && d < \pointer_length(buf->dim);
    requires Perm(&buf->dim[d], 1\2);
    requires Perm(buf->dim[d].stride, 1\2);
@*/
/*@ pure @*/ inline int _halide_buffer_get_stride_float(struct halide_buffer_float *buf, int d) {
    return buf->dim[d].stride;
}

/*@
inline resource buffer_float(struct halide_buffer_float *buf, rational p, int n_dims) = 
 buf != NULL **
 \pointer_length(buf) == 1 **
 Perm(buf, p) **
 Perm(buf->dim, p) **
 buf->dim != NULL **
 \pointer_length(buf->dim) == n_dims **
 Perm(buf->host, p) **
 buf->host != NULL;
@*/
#endif //HALIDE_BUFFER_TYPE_FLOAT










#ifdef __cplusplus
extern "C" {
#endif

/*@
  resource float_perm(float* xs) = xs != NULL
    ** (\forall* int i; 0 <= i && i < \pointer_length(xs); Perm({:&xs[i]:}, write));

  resource double_perm(double* xs) = xs != NULL 
    ** (\forall* int i; 0 <= i && i < \pointer_length(xs); Perm({:&xs[i]:}, write));

  resource int_perm(int* xs) = xs != NULL 
    ** (\forall* int i; 0 <= i && i < \pointer_length(xs); Perm({:&xs[i]:}, write));

  resource int_perm_bound(int* xs, int lo, int hi) = xs != NULL 
    ** (\forall* int i; 0 <= i && i < \pointer_length(xs); Perm({:&xs[i]:}, write))
    ** (\forall int i; 0 <= i && i <\pointer_length(xs); lo <= {:xs[i]:} && {:xs[i]:} < hi);

  resource denominator_r(int a, int a_min, int a_extent, int a_stride,
                         int si, int si_min, int si_extent, int si_stride,
                         int i, int i_min, int i_extent, int i_stride,
                         float* data) =
    (data != NULL && \pointer_length(data) == a_stride*a_extent &&
    req_3d(i, i_min, i_stride, i_extent,si, si_min, si_stride, si_extent,a, a_min, a_stride, a_extent) &&
    lemma_3d_access(i, i_min, i_stride, i_extent,si, si_min, si_stride, si_extent,a, a_min, a_stride, a_extent)
    ) ** 
    (\let int i = (a-a_min)*a_stride + (si-si_min)*si_stride + (i-i_min)*i_stride;
      (0 <= i && i < \pointer_length(data)) **
      Perm(&data[i], write)
    );

 ghost
  context_everywhere
    a_extent > 0 && 
    si_extent > 0 &&
    i_extent > 0 &&
    a_stride == si_extent * i_extent &&
    // i_extent == 2 && si_extent == 3 &&
    data != NULL && \pointer_length(data) == a_stride*a_extent;
  requires (\forall* int j; 0 <= j && j < a_stride*a_extent; Perm({:&data[j]:}, write));
  ensures (\forall* int a, int si, int i; a >= 0 && a < a_extent &&
    si >= 0 && si < si_extent &&
    i >= 0 && i < i_extent;
    {:denominator_r(a, 0, a_extent, a_stride, si, 0, si_extent, i_extent, i, 0, i_extent, 1, data):});
  void fold_denominator_r(int a_extent,
                          int si_extent,
                          int i_extent,
                          int a_stride,
                          float* data){
      loop_invariant a0 >= 0 && a0 <= a_extent;
      loop_invariant (\forall* int a, int si, int i; 
       a >= 0 && a < a0 &&
       si >= 0 && si < si_extent &&
       i >= 0 && i < i_extent;
       {:denominator_r(a, 0, a_extent, a_stride, si, 0, si_extent, i_extent, i, 0, i_extent, 1, data):});
      loop_invariant (\forall* int a, int si, int i; 
       a >= a0 && a >= 0 && a < a_extent &&
       si >= 0 && si < si_extent &&
       i >= 0 && i < i_extent;
       Perm(&data[(a)*a_stride + (si)*i_extent + (i)], write));
     for(int a0 = 0; a0 < a_extent; a0++){
        loop_invariant si0 >= 0 && si0 <= si_extent;
        loop_invariant (\forall* int si, int i;
        si >= 0 && si < si0 &&
        i >= 0 && i < i_extent;
        {:denominator_r(a0, 0, a_extent, a_stride, si, 0, si_extent, i_extent, i, 0, i_extent, 1, data):});
        loop_invariant (\forall* int si, int i;
          si >= si0 && si >= si && si < si_extent &&
          i >= 0 && i < i_extent 
          //&& lemma_3d_access(i, 0, 1, i_extent,si, 0, i_extent, si_extent,a0, 0, a_stride, a_extent)
          ;
          Perm(&data[(a0)*a_stride + (si)*i_extent + (i)], write));
       for(int si0 = 0; si0 < si_extent; si0++){
          loop_invariant i0 >= 0 && i0 <= i_extent;
          loop_invariant a0 >= 0 && a0 < a_extent;
          loop_invariant (\forall* int i;
            i >= 0 && i < i0 ;
            {:denominator_r(a0, 0, a_extent, a_stride, si0, 0, si_extent, i_extent, i, 0, i_extent, 1, data):});
          loop_invariant (\forall* int i;
            i >= i0 && i < i_extent;
            Perm(&data[(a0)*a_stride + (si0)*i_extent + (i)], write));  
         for(int i0 = 0; i0 < i_extent; i0++){
          assert i0-0>=0 && i0-0<i_extent;
          assert lemma_3d_access(
            i0, 0, 1, i_extent,
            si0, 0, i_extent, si_extent,
            a0, 0, a_stride, a_extent
          );
          fold denominator_r(a0, 0, a_extent, a_stride, si0, 0, si_extent, i_extent, i0, 0, i_extent, 1, data);
         }
       }
     }
    }  
@*/

/*
  ghost
  context_everywhere 
    a_extent > 0 && a_stride > 0 &&
    si_extent > 0 && si_stride > 0 &&
    i_extent > 0 && i_stride > 0 &&
    a_stride == si_extent * si_stride &&
    si_stride == i_extent * i_stride &&
    data != NULL && \pointer_length(data) == a_stride*a_extent;
  requires (\forall* int j; 0 <= j && j < a_stride*a_extent; Perm({:&data[j]:}, write));
  ensures (\forall* int a, int si, int i; a >= a_min && a < a_min + a_extent &&
    si >= si_min && si < si_min + si_extent &&
    i >= i_min && i < i_min + i_extent;
    {:denominator_r(a, a_min, a_extent, a_stride, si, si_min, si_extent, si_stride, i, i_min, i_extent, i_stride, data):});
  void fold_denominator_r(int a_min, int a_extent, int a_stride,
                          int si_min, int si_extent, int si_stride,
                          int i_min, int i_extent, int i_stride,
                          float* data){
      loop_invariant a0 >= a_min && a0 <= a_min + a_extent;
      loop_invariant (\forall* int a, int si, int i; 
       a >= a_min && a < a0 &&
       si >= si_min && si < si_min + si_extent &&
       i >= i_min && i < i_min + i_extent;
       {:denominator_r(a, a_min, a_extent, a_stride, si, si_min, si_extent, si_stride, i, i_min, i_extent, i_stride, data):});
      loop_invariant (\forall* int a, int si, int i; 
       a >= a0 && a >= a_min && a < a_min + a_extent &&
       si >= si_min && si < si_min + si_extent &&
       i >= i_min && i < i_min + i_extent;
       Perm(&data[(a-a_min)*a_stride + (si-si_min)*si_stride + (i-i_min)*i_stride], write));
     for(int a0 = a_min; a0 < a_min + a_extent; a0++){
        loop_invariant si0 >= si_min && si0 <= si_min + si_extent;
        loop_invariant (\forall* int si, int i;
        si >= si_min && si < si0 &&
        i >= i_min && i < i_min + i_extent;
        {:denominator_r(a0, a_min, a_extent, a_stride, si, si_min, si_extent, si_stride, i, i_min, i_extent, i_stride, data):});
        loop_invariant (\forall* int si, int i;
          si >= si0 && si >= si && si < si_min + si_extent &&
          i >= i_min && i < i_min + i_extent;
          Perm(&data[(a0-a_min)*a_stride + (si-si_min)*si_stride + (i-i_min)*i_stride], write));
       for(int si0 = si_min; si0 < si_min + si_extent; si0++){
          loop_invariant i0 >= i_min && i0 <= i_min + i_extent;
          loop_invariant (\forall* int i;
            i >= i_min && i < i0 ;
            {:denominator_r(a0, a_min, a_extent, a_stride, si0, si_min, si_extent, si_stride, i, i_min, i_extent, i_stride, data):});
          loop_invariant (\forall* int i;
            i >= i0 && i < i_min + i_extent;
            Perm(&data[(a0-a_min)*a_stride + (si0-si_min)*si_stride + (i-i_min)*i_stride], write));  
         for(int i0 = i_min; i0 < i_min + i_extent; i0++){
          lemma_3d_access(
            i0, i_min, i_stride, i_extent,
            si0, si_min, si_stride, si_extent,
            a0, a_min, a_stride, a_extent
          );  
          fold denominator_r(a0, a_min, a_extent, a_stride, si0, si_min, si_extent, si_stride, i0, i_min, i_extent, i_stride, data);
         }
       }
     }
    }  
*/


#ifdef __cplusplus
}  // extern "C"
#endif

