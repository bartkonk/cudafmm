#ifndef CUDAVECTOR_HPP
#define CUDAVECTOR_HPP

namespace gmx_gpu_fmm{

template<typename T, typename Allocator>
class cudavector
{

public:
    T* p;
    typedef T value_type;
    //typedef std::vector<T, Allocator> base;

    cudavector()
    {
        mysize = 0;
    }

    cudavector(size_t size, T /*val*/)
    {
        mysize = size;
        p = allocator.allocate(size);
        mysize = size;
    }

    ~cudavector()
    {
        // Deallocate only if not default constructed
        if(mysize != 0) {
            allocator.deallocate(p,size());
        }
    }

    void resize(size_t size_)
    {
        if(size() >0)
            allocator.deallocate(p, size());
        mysize = size_;
        p = allocator.allocate(size_);
    }

    T& operator[](size_t position)
    {
        return p[position];
    }

    size_t size()
    {
        return mysize;
    }

private:
    size_t mysize = 0;
    Allocator allocator;
};

}//namespace end
#endif // CUDAVECTOR_HPP

