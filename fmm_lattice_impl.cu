#include "fmm.hpp"
#include "cuda_LATTICE.hpp"
#include "cuda_lib.hpp"

namespace gmx_gpu_fmm{

void fmm_algorithm::lattice_impl(){

    typedef typename CoeffMatrix::value_type complex_type;
    CoeffMatrix* mudummy = nullptr;

    //wait for m2l
    for(size_t i = 0; i < STREAMS; ++i)
    {
        cudaStreamWaitEvent(priority_streams[current_priority_stream], priority_events[i], 0);
    }

    size_t op_p1xx2 = ( (2*p+1) * (2*p+1) );
    if(depth > 0)
    {
        __SoA2AoS_omega__<CoeffMatrixSoA, Box>(box, omegaSoA, 1, p1xp2_2, priority_streams[current_priority_stream]);
    }

    __copy_root_expasion<CoeffMatrix><<<(p1xp2_2-1)/64 + 1, 64, 0, priority_streams[current_priority_stream]>>>(omega_dipole, omega, p1xp2_2);

    if(dipole_compensation)
    {
        __P2M_P2L_dipole_corr<<<1, 64, 0, priority_streams[current_priority_stream]>>>(&q0abc[0], omega_dipole, &fake_particles[0], mu, p, fake_particle_size);
    }

    if (!open_boundary_conditions)
    {
#ifndef GMX_FMM_DOUBLE
        dim3 grid(1,1,1);
        dim3 block(p1+1,p1,1);
        __lattice<CoeffMatrix, CoeffMatrixSoA, Real, Real3, complex_type>
        <<<grid,block,(p1*p1+op_p1xx2)*sizeof(complex_type), priority_streams[current_priority_stream]>>>
        (omega_dipole, mudummy, muSoA, Lattice, p, p1, p1xx2, op_p1xx2);
#else
        if(p < MAXP)
        {
            dim3 grid(1,1,1);
            dim3 block(p1+1,p1,1);
            __lattice<CoeffMatrix, CoeffMatrixSoA, Real, Real3, complex_type>
            <<<grid,block,(p1*p1+op_p1xx2)*sizeof(complex_type), priority_streams[current_priority_stream]>>>
            (omega_dipole, mudummy, muSoA, Lattice, p, p1, p1xx2, op_p1xx2);
        }
        else
        {
            dim3 griD(p1,1,1);
            dim3 blocK(p1,1,1);
            __lattice_no_shared<CoeffMatrix, CoeffMatrixSoA, Real, Real3, complex_type>
            <<<griD,blocK,0,priority_streams[current_priority_stream]>>>
            (omega_dipole, mudummy, muSoA, Lattice, p);
        }
#endif
        if(dipole_compensation)
        {
            __AoS_addto_SoA_mu__(box, muSoA, 1, p1xp2_2, priority_streams[current_priority_stream]);
            cudaEventRecord(priority_events[current_priority_stream], priority_streams[current_priority_stream]);
        }
        else
        {
            cudaEventRecord(priority_events[current_priority_stream], priority_streams[current_priority_stream]);
        }
    }

}

}//namespace end
