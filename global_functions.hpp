#ifndef GLOBAL_FUNCTIONS_HPP
#define GLOBAL_FUNCTIONS_HPP

#include <math.h>
#include <vector>
#include <algorithm>
#include <cassert>
#include <stdlib.h>
#include <fstream>
#include "linear_algebra.hpp"
#include "xyzq.hpp"
#include "xyz.hpp"

namespace gmx_gpu_fmm{

extern int get_env_int(int dflt, const char* vrnm);

extern size_t boxes_above_depth(size_t d);

extern size_t boxes_on_depth(size_t d);

extern size_t make_boxid(size_t x, size_t y, size_t z, unsigned depth);

template <typename Real>
inline Real reciprocal(Real a)
{
    return  Real(1.) / a;
}

template <typename Real>
inline Real rsqrt(Real r)
{
    return Real(1.) / std::sqrt(r);
}

template <typename Real33>
Real33 change_of_basis_from_standard_basis(const Real33 & abc)
{
    typedef typename Real33::value_type Real3;
    typedef typename Real3::value_type Real;

    Real sizex = reciprocal(abc.a.x);
    Real sizey = reciprocal(abc.b.y);
    Real sizez = reciprocal(abc.c.z);

    Real33 result = Real33(Real3(sizex, 0., 0.),
                           Real3(0., sizey, 0.),
                           Real3(0., 0., sizez));


    return result;
}

template <typename Real33>
Real33 change_of_basis_to_standard_basis(const Real33 & abc)
{
    //std::cout<<abc.a<<abc.b<<abc.c<<std::endl;
    return abc;
}

template <typename RealOut, typename RealIn = RealOut>
struct AoS
{
    typedef RealOut Real;
    typedef const RealIn* const_pointer;
    typedef XYZQ<Real> Real4;
    typedef Real4 value_type;
    typedef RealIn vec[3];

    const vec* xyz;
    const_pointer x_, y_, z_, q_;
    size_t n;
    size_t offset;

    AoS(const vec* xyz, const RealIn* qq, const size_t n)
        : xyz(xyz), q_(qq), n(n), offset(3)
    {
        x_ = &xyz[0][0];
        y_ = &xyz[0][1];
        z_ = &xyz[0][2];
    }

    AoS(const_pointer x, const_pointer y, const_pointer z, const_pointer qq, size_t n)
        : x_(x), y_(y), z_(z), q_(qq), n(n), offset(1)
    {}

    const RealOut& x(size_t i) const
    {
        return *(x_+offset*i);
    }

    const RealOut& y(size_t i) const
    {
        return *(y_+offset*i);
    }

    const RealOut& z(size_t i) const
    {
        return *(z_+offset*i);
    }

    const RealOut& q(size_t i) const
    {
        return q_[i];
    }
};

}//namespace end

#endif // GLOBAL_FUNCTIONS_HPP
