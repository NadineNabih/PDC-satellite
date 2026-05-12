#include <iostream>
#include <omp.h>

int main() {
    const int N = 1000000;
    double sum = 0.0;

#pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < N; ++i) {
        sum += 1.0;
    }

    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif

    std::cout << "Sum=" << sum << " using " << threads << " threads\n";
    return 0;
}
