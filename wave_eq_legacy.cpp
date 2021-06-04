#include "array_types.hpp"
#include <iostream>
#include <fstream>
#include <utility>
#include <string>

using size_t = std::size_t;
using ptrdiff_t = std::ptrdiff_t;

struct wave_eq_problem
{
    vec<double> u_prev;
    vec<double> u;
    double step_x;
};

void print_layer(vec<double> layer, std::ofstream& log_file)
{
    size_t nx = layer.length() - 1;
    for (size_t i = 0; i <= nx; ++i)
    {
        log_file << layer(i) << " ";
    }
    log_file << "\n";
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

void explicit_step(wave_eq_problem& prob, double u_L, double u_R, double tau)
{
    vec<double> u_prev = prob.u_prev;
    vec<double> u = prob.u;
    size_t nx = u.length() - 1;
    double h = prob.step_x;
    double courant = (tau * tau) / (h * h);
    u_prev(0) = u_L;
    u_prev(nx) = u_R;
    for (size_t i = 1; i < nx; ++i)
    {
        u_prev(i) = -u_prev(i) + 2 * u(i) + courant * (u(i - 1) - 2 * u(i) + u(i + 1));
    }
    vec<double> tmp = u;
    prob.u = u_prev;
    prob.u_prev = tmp;
}

void explicit_solver(wave_eq_problem& prob,
    vec<double> u_left, vec<double> u_right, vec<double> u_t,
    double tau, size_t nt,
    std::ofstream& log_file)
{
    // входные параметры:
    // prob - начальная конфигурация
    // u_left - левое граничное условие
    // u_right - правое граничное условие
    // u_t - начальное условие на производную
    // tau - шаг по времени
    // nt - число шагов по времени
    // log_file - файловый дескриптор для вывода информации

    print_layer(prob.u_prev, log_file);
    apply_initial_conditions(prob, u_t, u_left(1), u_right(1), tau);
    print_layer(prob.u, log_file);

    // эволюция по времени
    for (size_t step = 2; step <= nt; ++step)
    {
        explicit_step(prob, u_left(step), u_right(step), tau);
        print_layer(prob.u, log_file);
    }
}

int main(int argc, char* argv[])
{
    if (argc < 1)
    {
        return 1;
    }

    std::ifstream in_file(argv[1]);

    size_t nx, nt;
    double h, tau, coeff_a;
    in_file >> nx >> nt >> h >> tau;

    vec<double> u_prev(nx + 1), u(nx + 1);
    vec<double> u_left(nt + 1), u_right(nt + 1);
    for (size_t i = 0; i <= nx; ++i)
    {
        in_file >> u_prev(i);
    }

    vec<double> u_t(nx + 1);
    for (size_t i = 0; i <= nx; ++i)
    {
        in_file >> u_t(i);
    }

    u_left(0) = u_prev(0);
    u_right(0) = u_prev(nx);
    for (size_t i = 1; i <= nt; ++i)
    {
        in_file >> u_left(i) >> u_right(i);
    }

    wave_eq_problem prob = { u_prev, u, h };

    std::ofstream log_file("wave_eq.out");
    explicit_solver(prob, u_left, u_right, u_t, tau, nt, log_file);

    return 0;
}