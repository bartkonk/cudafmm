#ifndef CUDA_LATTICE_HPP
#define CUDA_LATTICE_HPP

#include "cuda_keywords.hpp"

namespace gmx_gpu_fmm{

template <typename CoefficientMatrix>
__global__
void __copy_root_expasion(CoefficientMatrix *m_in, CoefficientMatrix **m_from, size_t p1xp2_2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < p1xp2_2)
    {
        (*m_in)(i) = m_from[0]->get_vectorized(i);
    }
}

template <typename CoefficientMatrix, typename CoefficientMatrixSoA, typename Real, typename Real3, typename complex_type, bool SoA = true>
__global__ void
__lattice(
         CoefficientMatrix *Omega,
         CoefficientMatrix *mu,
         CoefficientMatrixSoA *muSoA,
         CoefficientMatrix * B,
         const size_t p,
         const size_t p1,
         const size_t p1xx2,
         const size_t op_p1xx2
         )

{
    extern __shared__ complex_type cache[];
    complex_type* op_jkml = cache;
    complex_type* omega_jk =(&cache[op_p1xx2]);

    const ssize_t l = threadIdx.x;
    const ssize_t m = threadIdx.y;
    const ssize_t tx = l*p1 + m;

    if(tx<p1xx2)
    {
        if(l>0 && m>0)
        {
            if(m <= l)
            {
                //self
                complex_type opp(B->get_vectorized(l,m));
                ssize_t l2 = l*l;
                op_jkml[l2+l+m] = opp;
                //complex_type omm(omegaSoA->getSoA(l,m,0));
                complex_type omm(Omega->get_vectorized(l,m));
                omega_jk[l2+l+m] = omm;

                ssize_t lm = -m;
                op_jkml[l2+l+lm] = cuda_toggle_sign_if_odd(m, conj(opp));
                omega_jk[l2+l+lm] = cuda_toggle_sign_if_odd(m, conj(omm));

                //bottom right
                ssize_t lp = l+p;
                ssize_t rbl2 = lp*lp;
                complex_type opp_br(B->get_vectorized(lp,m));
                op_jkml[rbl2+lp+m] = opp_br;

                //bottom left
                op_jkml[rbl2+lp+lm] = cuda_toggle_sign_if_odd(m, conj(opp_br));

                //corner bottom right
                ssize_t rbm = m+p;
                complex_type opp_rbm(B->get_vectorized(lp,rbm));
                op_jkml[rbl2+lp+rbm] = opp_rbm;

                //corner bottom left
                ssize_t lbm = -rbm;
                op_jkml[rbl2+lp+lbm] = cuda_toggle_sign_if_odd(m, conj(opp_rbm));
            }
            if(m > l)
            {
                //bottom right
                ssize_t lp = l+p;
                ssize_t rbl2 = lp*lp;
                complex_type opp_br(B->get_vectorized(lp,m));
                op_jkml[rbl2+lp+m] = opp_br;

                //bottom left
                ssize_t lm = -m;
                op_jkml[rbl2+lp+lm] = cuda_toggle_sign_if_odd(m, conj(opp_br));
            }
        }
        if(m==0)
        {
            //self
            ssize_t l2 = l*l;
            op_jkml[l2+l] = B->get_vectorized(l,0);
            //omega_jk[l2+l] = omegaSoA->getSoA(l,0,0);
            omega_jk[l2+l] = Omega->get_vectorized(l,0);
            //bottom
            ssize_t lp = l+p;
            ssize_t rbl2 = lp*lp;
            op_jkml[rbl2+lp+0] = B->get_vectorized(lp,0);
        }
    }
    __syncthreads();

    ssize_t l_,t1,ll,mm,f,f1;
    ssize_t k_start,k_end;
    t1 = tx-l;
    l_ = t1/p1;
    f = 1-(l-l_);
    f1 = 1-f;
    ll = f1*l_ + f*m;
    mm = f1*m + f*l_;

    complex_type mu_l_m(0.);

    for (ssize_t j=0; j<=p; j++)
    {
        k_start = f*j;
        k_end = f*j + j + f;

        int j_ = static_cast<int>(j);
        ssize_t jj2 = static_cast<ssize_t>(j_*j_);

        ssize_t jl = j+ll;
        ssize_t jl2 = jl*jl;
        ssize_t lk= 0;

        complex_type mu_l_m_j(0.);
        ssize_t jl2_jl_m = jl2+jl+mm;
        ssize_t jj2_j = jj2+j;

        for (ssize_t k_ = k_start; k_ < k_end; k_++)
        {
            lk = k_-j;
            complex_type op(op_jkml[jl2_jl_m+lk]);
            complex_type om(omega_jk[jj2_j+lk]);
            mu_l_m_j += op*om;
        }
        mu_l_m +=cuda_toggle_sign_if_odd(j, mu_l_m_j);
    }
    if(SoA)
    {
        const size_t lm_index = (ll * (ll + 1)) / 2 + mm;
        (muSoA->get_lin_SoA_val(lm_index, 0)) %= mu_l_m;
    }
    else
    {
        (*mu)(ll, mm) %= mu_l_m;
    }
}

template <typename CoefficientMatrix, typename CoefficientMatrixSoA, typename Real, typename Real3, typename complex_type, bool SoA = true>
__global__ void
__lattice_no_shared(
         CoefficientMatrix *Omega,
         CoefficientMatrix *mu,
         CoefficientMatrixSoA *muSoA,
         CoefficientMatrix * B,
         const size_t p
         )

{
    const ssize_t m = threadIdx.x;
    const ssize_t l = blockIdx.x;

    if(m <= l)
    {
        complex_type mu_l_m(0.);

        for (ssize_t j=0; j<=p; j++)
        {
            complex_type mu_l_m_j(0.);
            for (ssize_t k = -j; k <= j; k++)
            {
                mu_l_m_j += Omega->get_vectorized(j,k) * B->get_vectorized(l+j,m+k);
            }
            mu_l_m +=cuda_toggle_sign_if_odd(j, mu_l_m_j);
        }
        if(SoA)
        {
            const size_t lm_index = (l * (l + 1)) / 2 + m;
            (muSoA->get_lin_SoA_val(lm_index, 0)) %= mu_l_m;
        }
        else
        {
            (*mu)(l, m) %= mu_l_m;
        }
    }
}

}//namespace end

#endif // CUDA_LATTICE_HPP
