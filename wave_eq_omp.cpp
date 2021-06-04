#include "array_types.hpp"
#include <iostream>
#include <fstream>
#include <utility>
#include <string>
#include <omp.h>

using size_t = std::size_t;
using ptrdiff_t = std::ptrdiff_t;

struct wave_eq_problem
{
    vec<double> u;
    vec<double> u_t;
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

    // эволюция по времени
    size_t i;
    size_t nx = prob.u.length() - 1;
    omp_set_num_threads(8);
#pragma omp parallel for
    for (i = 1; i < prob.u.length() - 1; ++i)
    {
        prob.u_t(i) += (prob.u(i - 1) - 2 * prob.u(i) + prob.u(i + 1)) * (tau / (prob.step_x * prob.step_x)) / 2;
    }
#pragma omp parallel default(shared) private(i)
    {
        for (size_t step = 1; step <= nt; ++step)
        {
#pragma omp master
            {
                prob.u(0) = u_left(step);
                prob.u(nx) = u_right(step);
            }
#pragma omp for
            for (i = 1; i < prob.u.length() - 1; ++i)
            {
                prob.u(i) += prob.u_t(i) * tau;
            }
#pragma omp for
            for (i = 1; i < prob.u.length() - 1; ++i)
            {
                prob.u_t(i) += (prob.u(i - 1) - 2 * prob.u(i) + prob.u(i + 1)) * tau / (prob.step_x * prob.step_x);
            }
#pragma omp barrier
        }
    }
    print_layer(prob.u);
}

int main(int argc, char* argv[])
{
    size_t nx, nt;
    double h, tau;
    std::cin >> nx >> nt >> h >> tau;

    vec<double> u(nx + 1);
    vec<double> u_left(nt + 1), u_right(nt + 1);
    for (size_t i = 0; i <= nx; ++i)
    {
        std::cin >> u(i);
    }

    vec<double> u_t(nx + 1);
    for (size_t i = 0; i <= nx; ++i)
    {
        std::cin >> u_t(i);
    }

    u_left(0) = u(0);
    u_right(0) = u(nx);
    for (size_t i = 1; i <= nt; ++i)
    {
        std::cin >> u_left(i) >> u_right(i);
    }

    wave_eq_problem prob = { u, u_t, h };

    //double t0 = omp_get_wtime();
    explicit_solver(prob, u_left, u_right, u_t, tau, nt);
    //double t1 = omp_get_wtime();
    //std::cout << "Execution time: " << t1 - t0 << "\n";

    return 0;
}