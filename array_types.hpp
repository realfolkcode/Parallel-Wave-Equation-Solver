#ifndef ARRAY_TYPES_H

#define ARRAY_TYPES_H
#include <memory>
#include <utility>
#include <cstring>

using ptrdiff_t = std::ptrdiff_t;

template <class T>
class vec final
{
private:
    ptrdiff_t len;
    std::shared_ptr<T[]> data;

public:
    vec(ptrdiff_t n) : len(n), data(new T[n]) {};
    ~vec() = default;
    ptrdiff_t length() { return len; }
    T* raw_ptr() { return data.get(); }
    T& operator()(ptrdiff_t idx) { return data[idx]; }
};

template <class T>
class matrix final
{
private:
    ptrdiff_t nr_, nc_;
    std::shared_ptr<T[]> data;

public:
    matrix(ptrdiff_t nr, ptrdiff_t nc) : nr_(nr), nc_(nc), data(new T[nr * nc]) {};
    ~matrix() = default;
    ptrdiff_t length() { return nr_ * nc_; }
    ptrdiff_t nrows() { return nr_; }
    ptrdiff_t ncols() { return nc_; }
    T* raw_ptr() { return data.get(); }
    T& operator()(ptrdiff_t row, ptrdiff_t col) { return data[row * nc_ + col]; }
    T& operator()(ptrdiff_t idx) { return data[idx]; }
    vec<T> row(ptrdiff_t);
    vec<T> col(ptrdiff_t);
};

template <class T>
vec<T> matrix<T>::row(ptrdiff_t r)
{
    vec<T> v(nc_);
    std::memcpy(v.raw_ptr(), raw_ptr() + r * nc_, nc_ * sizeof(T));
    return v;
}

template <class T>
vec<T> matrix<T>::col(ptrdiff_t c)
{
    vec<T> v(nr_);
    for (ptrdiff_t i = 0; i < nr_; i++)
    {
        v(i) = data[i * nc_ + c];
    }
    return v;
}

#endif