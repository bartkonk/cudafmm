#ifndef IOADAPTER_HPP
#define IOADAPTER_HPP

#include "cuda_keywords.hpp"
#include "xyz.hpp"
#include <vector>
#include "cuda_alloc.hpp"
#include "cuda_atomics.hpp"
#include "data_type.hpp"
#include "cudavector.hpp"
#include "managed.hpp"
#include "architecture.hpp"

namespace gmx_gpu_fmm{

template <typename T1Vector, typename T3Vector, typename arch = Host<typename T1Vector::value_type>>
struct outputadapter : public Managed<arch>
{
    typedef T1Vector vector1_type;
    typedef T3Vector vector3_type;
    typedef typename vector1_type::value_type Real;
    typedef typename vector3_type::value_type Real3;

    vector1_type & vPhi;
    vector3_type & vF;

    Real*  const vPhi_ptr;
    Real3* const vF_ptr;

    outputadapter(vector1_type & vPhi, vector3_type & vF);

    ~outputadapter();

    CUDA
    void reduce_pf(size_t i, Real p, const Real3 & f);

    DEVICE
    void atomic_reduce_pf(size_t i, Real p, const Real3 & f);

    DEVICE
    void atomic_reduce_pf(size_t i, Real p, const REAL3 &f);

    DEVICE
    void atomic_reduce_f(size_t i, const Real3 & f);

    DEVICE
    void atomic_reduce_f(size_t i, const REAL3 &f);

    CUDA
    void not_atomic_reduce_pf(size_t i, Real p, const Real3 & f);

    DEVICE
    void not_atomic_reduce_pf(size_t i, Real p, const REAL3 & f);
};

}//namespace end

#endif // IOADAPTER_HPP
