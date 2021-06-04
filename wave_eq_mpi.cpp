#include "array_types.hpp"
#include <mpi.h>
#include <iostream>
#include <fstream>
#include <utility>
#include <string>
#include <stdint.h>
#include <limits.h>

#if SIZE_MAX == UCHAR_MAX
    #define my_MPI_SIZE_T MPI_UNSIGNED_CHAR
#elif SIZE_MAX == USHRT_MAX
    #define my_MPI_SIZE_T MPI_UNSIGNED_SHORT
#elif SIZE_MAX == UINT_MAX
    #define my_MPI_SIZE_T MPI_UNSIGNED
#elif SIZE_MAX == ULONG_MAX
    #define my_MPI_SIZE_T MPI_UNSIGNED_LONG
#elif SIZE_MAX == ULLONG_MAX
    #define my_MPI_SIZE_T MPI_UNSIGNED_LONG_LONG
#else
    #error "something went wrong"
#endif

using size_t = std::size_t;
using ptrdiff_t = std::ptrdiff_t;

struct wave_eq_problem
{
    vec<double> u_prev;
    vec<double> u;
    double step_x;
};

void print_layer(vec<double> layer)
{
    size_t nx = layer.length() - 1;
    for (size_t i = 0; i <= nx; ++i)
    {
        std::cout << layer(i) << " ";
    }
    std::cout << "\n";
}

void apply_initial_conditions(wave_eq_problem& prob, vec<double> u_t, double u_L, double u_R, double tau)
{
    vec<double> u_prev = prob.u_prev;
    vec<double> u = prob.u;
    size_t nx = u.length() - 1;
    double h = prob.step_x;
    double courant = tau * tau / (h * h);
    u(0) = u_L;
    u(nx) = u_R;
    for (size_t i = 1; i < nx; ++i)
    {
        u(i) = u_prev(i) + tau * u_t(i) + courant / 2 * (u_prev(i - 1) - 2 * u_prev(i) + u_prev(i + 1));
    }
}

void explicit_step(wave_eq_problem& prob, double u_L, double u_R, double tau,
                    int comm_size, int myrank, ptrdiff_t ncells_each)
{
    vec<double> u_prev = prob.u_prev;
    vec<double> u = prob.u;
    ptrdiff_t nstart = myrank * ncells_each + 1;
    ptrdiff_t nstop = nstart + ncells_each;

    MPI_Status status;
    double* u_ptr = u.raw_ptr() + nstart;

    if (myrank > 0)
    {
        MPI_Send(u_ptr, 1, MPI_DOUBLE, myrank - 1, 0, MPI_COMM_WORLD);
        MPI_Recv(u_ptr - 1, 1, MPI_DOUBLE, myrank - 1, 1, MPI_COMM_WORLD, &status);
    }
    if (myrank < comm_size - 1)
    {
        MPI_Recv(u_ptr + ncells_each, 1, MPI_DOUBLE, myrank + 1, 0, MPI_COMM_WORLD, &status);
        MPI_Send(u_ptr + ncells_each - 1, 1, MPI_DOUBLE, myrank + 1, 1, MPI_COMM_WORLD);
    }

    size_t nx = u.length() - 1;
    double h = prob.step_x;
    double courant = (tau * tau) / (h * h);

    u_prev(0) = u_L;
    u_prev(nx) = u_R;

    for (ptrdiff_t i = nstart; i < nstop; ++i)
    {
        u_prev(i) = -u_prev(i) + 2 * u(i) + courant * (u(i - 1) - 2 * u(i) + u(i + 1));
    }
    vec<double> tmp = u;
    prob.u = u_prev;
    prob.u_prev = tmp;
}

void explicit_solver(wave_eq_problem& prob,
    vec<double> u_left, vec<double> u_right, vec<double> u_t,
    double tau, size_t nt)
{
    // входные параметры:
    // prob - начальная конфигурация
    // u_left - левое граничное условие
    // u_right - правое граничное условие
    // u_t - начальное условие на производную
    // tau - шаг по времени
    // nt - число шагов по времени
    // log_file - файловый дескриптор для вывода информации

    ptrdiff_t nx = prob.u.length() - 1;

    int myrank, comm_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    ptrdiff_t ncells_each = (nx - 1) / comm_size;

    // эволюция по времени
    for (size_t step = 2; step <= nt; ++step)
    {
        explicit_step(prob, u_left(step), u_right(step), tau, comm_size, myrank, ncells_each);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    double* u_ptr = prob.u.raw_ptr() + 1 + ncells_each * myrank;
    MPI_Gather(u_ptr, ncells_each, MPI_DOUBLE,
               u_ptr, ncells_each, MPI_DOUBLE,
               0, MPI_COMM_WORLD);
    if (myrank == 0)
    {
        print_layer(prob.u);
    }
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    size_t nx, nt;
    double h, tau;

    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    if (myrank == 0)
    {
        std::cin >> nx >> nt >> h >> tau;
    }
    MPI_Bcast(&nx, 1, my_MPI_SIZE_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nt, 1, my_MPI_SIZE_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&h, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&tau, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    vec<double> u_prev(nx + 1), u(nx + 1);
    vec<double> u_left(nt + 1), u_right(nt + 1);
    vec<double> u_t(nx + 1);
    if (myrank == 0)
    {
        for (size_t i = 0; i <= nx; ++i)
        {
            std::cin >> u_prev(i);
        }
        for (size_t i = 0; i <= nx; ++i)
        {
            std::cin >> u_t(i);
        }
    }
    double* u_prev_ptr = u_prev.raw_ptr();
    double* u_t_ptr = u_t.raw_ptr();
    MPI_Bcast(u_prev_ptr, nx + 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(u_t_ptr, nx + 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    u_left(0) = u_prev(0);
    u_right(0) = u_prev(nx);
    if (myrank == 0)
    {
        for (size_t i = 1; i <= nt; ++i)
        {
            std::cin >> u_left(i) >> u_right(i);
        }
    }
    double* u_left_ptr = u_left.raw_ptr();
    double* u_right_ptr = u_right.raw_ptr();
    MPI_Bcast(u_left_ptr, nt + 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(u_right_ptr, nt + 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    wave_eq_problem prob = { u_prev, u, h };

    apply_initial_conditions(prob, u_t, u_left(1), u_right(1), tau);
    MPI_Barrier(MPI_COMM_WORLD);

    double t0 = MPI_Wtime()
    explicit_solver(prob, u_left, u_right, u_t, tau, nt);
    double t1 = MPI_Wtime()

    if (myrank == 0)
    {
        std::cerr << "Execution time: " << t1 - t0 << "\n";
    }

    MPI_Finalize();

    return 0;
}