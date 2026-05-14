/*
 * ============================================================
 *  AID323 – Parallel and Distributed Computing
 *  Project : Satellite Image Analysis (Hybrid MPI + OpenMP)
 *  Language: C++17
 *  Author  : [Your Names]
 *
 *  Parallelisation strategy
 *  ─────────────────────────
 *  MPI    : Row-wise domain decomposition across P processes.
 *           Each process owns floor(H/P) or ceil(H/P) rows.
 *
 *  OpenMP : Thread-level parallelism inside each MPI process
 *           for NDVI computation and Gaussian convolution.
 *
 *  Halo exchange
 *  ─────────────
 *  Before smoothing, every process needs KERNEL_RADIUS ghost
 *  rows from its upper and lower neighbours (exchanged via
 *  MPI_Sendrecv — deadlock-free, no explicit non-blocking needed).
 *  Boundary ranks mirror their own edge rows instead.
 * ============================================================
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <cstring>
#include <mpi.h>
#include <omp.h>

using namespace std;

/* ── Configuration ─────────────────────────────────────────── */
constexpr int   IMAGE_HEIGHT  = 4096;
constexpr int   IMAGE_WIDTH   = 4096;
constexpr int   KERNEL_SIZE   = 15;
constexpr int   KERNEL_RADIUS = KERNEL_SIZE / 2;
constexpr float SIGMA         = 2.0f;

/* ── Gaussian kernel ───────────────────────────────────────── */
vector<float> make_gaussian_kernel(int size, float sigma) {
    int r = size / 2;
    vector<float> kernel(size * size);
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            float x = static_cast<float>(i - r);
            float y = static_cast<float>(j - r);
            float val = exp(-(x*x + y*y) / (2.0f * sigma * sigma));
            kernel[i * size + j] = val;
            sum += val;
        }
    }
    for (auto& v : kernel) v /= sum;
    return kernel;
}

/* ── Reproducible band generation ──────────────────────────── */
/*
 * Uses the same seed formula as sequential.cpp so both versions
 * produce identical pixel data — correctness can be verified
 * by comparing printed statistics.
 */
vector<float> generate_band_rows(int start_row, int num_rows,
                                  int width, float base,
                                  unsigned int seed) {
    vector<float> band(static_cast<size_t>(num_rows) * width);
    for (int i = 0; i < num_rows; i++) {
        unsigned int row_seed = seed + static_cast<unsigned int>(start_row + i) * 1000u;
        srand(row_seed);
        for (int j = 0; j < width; j++) {
            float v = base + 0.3f * (static_cast<float>(rand()) / RAND_MAX) - 0.15f;
            v = max(0.0f, min(1.0f, v));
            band[i * width + j] = v;
        }
    }
    return band;
}

/* ── OpenMP-parallel NDVI ──────────────────────────────────── */
vector<float> compute_ndvi_omp(const vector<float>& nir,
                                const vector<float>& red,
                                int rows, int width) {
    vector<float> ndvi(static_cast<size_t>(rows) * width);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < width; j++) {
            int idx = i * width + j;
            float d = nir[idx] + red[idx];
            ndvi[idx] = (d > 1e-6f) ? (nir[idx] - red[idx]) / d : 0.0f;
        }
    }
    return ndvi;
}

/* ── OpenMP-parallel Gaussian convolution (with halo) ─────── */
/*
 * halo_buf layout : [top_ghost(R rows) | local(num_rows) | bot_ghost(R rows)]
 * total rows in halo_buf = num_rows + 2*KERNEL_RADIUS
 */
vector<float> apply_gaussian_omp(const vector<float>& halo_buf,
                                  const vector<float>& kernel,
                                  int num_rows, int width) {
    vector<float> output(static_cast<size_t>(num_rows) * width);
    int r          = KERNEL_RADIUS;
    int total_rows = num_rows + 2 * r;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < width; j++) {
            float acc = 0.0f;
            for (int ki = -r; ki <= r; ki++) {
                int ni = max(0, min(total_rows - 1, (i + r) + ki));
                for (int kj = -r; kj <= r; kj++) {
                    int nj = max(0, min(width - 1, j + kj));
                    acc += halo_buf[ni * width + nj]
                         * kernel[(ki + r) * KERNEL_SIZE + (kj + r)];
                }
            }
            output[i * width + j] = acc;
        }
    }
    return output;
}

/* ── main ──────────────────────────────────────────────────── */
int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int H = IMAGE_HEIGHT, W = IMAGE_WIDTH;
    int halo = KERNEL_RADIUS;

    /* ── Row distribution ── */
    int base_rows  = H / nprocs;
    int remainder  = H % nprocs;
    int local_rows = base_rows + (rank < remainder ? 1 : 0);
    int start_row  = rank * base_rows + (rank < remainder ? rank : remainder);

    if (rank == 0) {
        cout << "╔══════════════════════════════════════════════╗\n"
             << "║  Parallel Satellite Analysis (MPI + OpenMP)  ║\n"
             << "╚══════════════════════════════════════════════╝\n"
             << "  Image   : " << H << " x " << W << "\n"
             << "  MPI     : " << nprocs << " processes\n"
             << "  OpenMP  : " << omp_get_max_threads() << " threads/process\n"
             << "  Kernel  : " << KERNEL_SIZE << " x " << KERNEL_SIZE
             << "  (sigma = " << SIGMA << ")\n" << flush;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_total_start = MPI_Wtime();

    /* ── Build kernel (same on every rank) ── */
    auto kernel = make_gaussian_kernel(KERNEL_SIZE, SIGMA);

    /* ── Stage 1: Generate local band data ── */
    auto local_nir = generate_band_rows(start_row, local_rows, W, 0.6f,  42u);
    auto local_red = generate_band_rows(start_row, local_rows, W, 0.3f, 137u);

    /* ── Stage 2: NDVI (OpenMP) ── */
    MPI_Barrier(MPI_COMM_WORLD);
    double t_ndvi_s = MPI_Wtime();
    auto local_ndvi = compute_ndvi_omp(local_nir, local_red, local_rows, W);
    double t_ndvi_e = MPI_Wtime();

    /* ── Stage 3: Halo exchange ── */
    MPI_Barrier(MPI_COMM_WORLD);
    double t_comm_s = MPI_Wtime();

    /* Allocate halo buffer and copy local data into the centre */
    int halo_rows = local_rows + 2 * halo;
    vector<float> halo_buf(static_cast<size_t>(halo_rows) * W, 0.0f);
    copy(local_ndvi.begin(), local_ndvi.end(),
         halo_buf.begin() + (long)halo * W);

    MPI_Status status;
    int up   = (rank > 0)          ? rank - 1 : MPI_PROC_NULL;
    int down = (rank < nprocs - 1) ? rank + 1 : MPI_PROC_NULL;

    /*
     * Top ghost exchange:
     *   send my first `halo` rows  → rank-1
     *   recv rank-1's last  `halo` rows → my top ghost
     */
    MPI_Sendrecv(
        local_ndvi.data(),             halo * W, MPI_FLOAT, up,   10,
        halo_buf.data(),               halo * W, MPI_FLOAT, up,   11,
        MPI_COMM_WORLD, &status
    );

    /*
     * Bottom ghost exchange:
     *   send my last  `halo` rows  → rank+1
     *   recv rank+1's first `halo` rows → my bottom ghost
     */
    MPI_Sendrecv(
        local_ndvi.data() + (long)(local_rows - halo) * W,
                                       halo * W, MPI_FLOAT, down, 11,
        halo_buf.data()  + (long)(local_rows + halo) * W,
                                       halo * W, MPI_FLOAT, down, 10,
        MPI_COMM_WORLD, &status
    );

    /* Boundary mirror: replicate edge rows where no neighbour exists */
    if (rank == 0)
        copy(local_ndvi.begin(),
             local_ndvi.begin() + (long)halo * W,
             halo_buf.begin());

    if (rank == nprocs - 1)
        copy(local_ndvi.end()   - (long)halo * W,
             local_ndvi.end(),
             halo_buf.begin()   + (long)(local_rows + halo) * W);

    double t_comm_e = MPI_Wtime();

    /* ── Stage 4: Gaussian smoothing (OpenMP) ── */
    MPI_Barrier(MPI_COMM_WORLD);
    double t_conv_s = MPI_Wtime();
    auto local_smooth = apply_gaussian_omp(halo_buf, kernel, local_rows, W);
    double t_conv_e   = MPI_Wtime();

    double t_total_end = MPI_Wtime();

    /* ── Gather at rank 0 ── */
    vector<int> recvcounts(nprocs), displs(nprocs);
    vector<float> global_result;

    if (rank == 0) {
        int offset = 0;
        for (int r = 0; r < nprocs; r++) {
            int rows_r     = base_rows + (r < remainder ? 1 : 0);
            recvcounts[r]  = rows_r * W;
            displs[r]      = offset;
            offset        += recvcounts[r];
        }
        global_result.resize(static_cast<size_t>(H) * W);
    }

    MPI_Gatherv(local_smooth.data(), local_rows * W, MPI_FLOAT,
                global_result.data(), recvcounts.data(), displs.data(),
                MPI_FLOAT, 0, MPI_COMM_WORLD);

    /* ── Reduce timings (max across ranks = wall-clock time) ── */
    double loc_ndvi  = t_ndvi_e  - t_ndvi_s;
    double loc_comm  = t_comm_e  - t_comm_s;
    double loc_conv  = t_conv_e  - t_conv_s;
    double loc_total = t_total_end - t_total_start;

    double max_ndvi, max_comm, max_conv, max_total;
    MPI_Reduce(&loc_ndvi,  &max_ndvi,  1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&loc_comm,  &max_comm,  1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&loc_conv,  &max_conv,  1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&loc_total, &max_total, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        float mn = *min_element(global_result.begin(), global_result.end());
        float mx = *max_element(global_result.begin(), global_result.end());
        double mean = accumulate(global_result.begin(),
                                 global_result.end(), 0.0) / global_result.size();

        cout << "\n┌─────────────────────────────────────────┐\n"
             << "│  Results                                │\n"
             << "├─────────────────────────────────────────┤\n"
             << fixed << setprecision(4)
             << "│  NDVI min   : " << setw(8) << mn   << "                  │\n"
             << "│  NDVI max   : " << setw(8) << mx   << "                  │\n"
             << "│  NDVI mean  : " << setw(8) << mean << "                  │\n"
             << "├─────────────────────────────────────────┤\n"
             << setprecision(3)
             << "│  NDVI comp  : " << setw(8) << max_ndvi  << " s               │\n"
             << "│  Halo exch  : " << setw(8) << max_comm  << " s               │\n"
             << "│  Gauss conv : " << setw(8) << max_conv  << " s               │\n"
             << "│  Total      : " << setw(8) << max_total << " s               │\n"
             << "└─────────────────────────────────────────┘\n";

        /* Append timing row for benchmark script */
        ofstream fp("parallel_time.txt", ios::app);
        if (fp) {
            fp << fixed << setprecision(6)
               << nprocs << " " << omp_get_max_threads() << " "
               << max_total << " " << max_comm << " "
               << max_ndvi  << " " << max_conv  << "\n";
        }
    }

    MPI_Finalize();
    return 0;
}
