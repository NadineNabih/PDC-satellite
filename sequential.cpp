/*
 * ============================================================
 *  AID323 – Parallel and Distributed Computing
 *  Project : Satellite Image Analysis (Sequential Baseline)
 *  Language: C++17
 *  Author  : [Your Names]
 *
 *  Pipeline:
 *    1. Generate synthetic NIR + RED satellite bands
 *    2. Compute NDVI = (NIR - RED) / (NIR + RED)
 *    3. Apply 15x15 Gaussian smoothing filter
 *    4. Report timing and image statistics
 * ============================================================
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <iomanip>

using namespace std;
using namespace chrono;

/* ── Configuration ─────────────────────────────────────────── */
constexpr int   IMAGE_HEIGHT  = 4096;
constexpr int   IMAGE_WIDTH   = 4096;
constexpr int   KERNEL_SIZE   = 15;
constexpr int   KERNEL_RADIUS = KERNEL_SIZE / 2;
constexpr float SIGMA         = 2.0f;

/* ── Timing helper ─────────────────────────────────────────── */
static double elapsed_sec(steady_clock::time_point t0) {
    return duration<double>(steady_clock::now() - t0).count();
}

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

/* ── Synthetic band generation ─────────────────────────────── */
vector<float> generate_band(int height, int width,
                             float base, unsigned int seed) {
    vector<float> band(static_cast<size_t>(height) * width);
    for (int i = 0; i < height; i++) {
        unsigned int row_seed = seed + static_cast<unsigned int>(i) * 1000u;
        srand(row_seed);
        for (int j = 0; j < width; j++) {
            float v = base + 0.3f * (static_cast<float>(rand()) / RAND_MAX) - 0.15f;
            v = max(0.0f, min(1.0f, v));
            band[i * width + j] = v;
        }
    }
    return band;
}

/* ── NDVI ──────────────────────────────────────────────────── */
vector<float> compute_ndvi(const vector<float>& nir,
                            const vector<float>& red,
                            int height, int width) {
    vector<float> ndvi(static_cast<size_t>(height) * width);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int idx = i * width + j;
            float d = nir[idx] + red[idx];
            ndvi[idx] = (d > 1e-6f) ? (nir[idx] - red[idx]) / d : 0.0f;
        }
    }
    return ndvi;
}

/* ── Gaussian 2-D convolution ──────────────────────────────── */
vector<float> apply_gaussian(const vector<float>& input,
                              const vector<float>& kernel,
                              int height, int width) {
    vector<float> output(static_cast<size_t>(height) * width);
    int r = KERNEL_RADIUS;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            float acc = 0.0f;
            for (int ki = -r; ki <= r; ki++) {
                int ni = max(0, min(height - 1, i + ki));
                for (int kj = -r; kj <= r; kj++) {
                    int nj = max(0, min(width  - 1, j + kj));
                    acc += input[ni * width + nj]
                         * kernel[(ki + r) * KERNEL_SIZE + (kj + r)];
                }
            }
            output[i * width + j] = acc;
        }
    }
    return output;
}

/* ── Statistics ────────────────────────────────────────────── */
struct Stats { float mn, mx; double mean; };

Stats compute_stats(const vector<float>& data) {
    Stats s;
    s.mn   = *min_element(data.begin(), data.end());
    s.mx   = *max_element(data.begin(), data.end());
    s.mean = accumulate(data.begin(), data.end(), 0.0) / data.size();
    return s;
}

/* ── main ──────────────────────────────────────────────────── */
int main() {
    int H = IMAGE_HEIGHT, W = IMAGE_WIDTH;
    double mem_mb = 5.0 * H * W * sizeof(float) / (1024.0 * 1024.0);

    cout << "╔══════════════════════════════════════════════╗\n"
         << "║  Sequential Satellite Image Analysis (C++)   ║\n"
         << "╚══════════════════════════════════════════════╝\n"
         << "  Image  : " << H << " x " << W << "  (" << (long)H*W << " pixels)\n"
         << "  Kernel : " << KERNEL_SIZE << " x " << KERNEL_SIZE
         << "  (sigma = " << SIGMA << ")\n"
         << fixed << setprecision(0)
         << "  Memory : ~" << mem_mb << " MB\n\n";

    auto kernel = make_gaussian_kernel(KERNEL_SIZE, SIGMA);

    auto t0 = steady_clock::now();
    cout << "[1/4] Generating synthetic satellite bands ... " << flush;
    auto nir = generate_band(H, W, 0.6f,  42u);
    auto red = generate_band(H, W, 0.3f, 137u);
    cout << fixed << setprecision(3) << elapsed_sec(t0) << " s\n";

    t0 = steady_clock::now();
    cout << "[2/4] Computing NDVI ...                        " << flush;
    auto ndvi = compute_ndvi(nir, red, H, W);
    double ndvi_time = elapsed_sec(t0);
    cout << ndvi_time << " s\n";

    t0 = steady_clock::now();
    cout << "[3/4] Applying Gaussian smoothing ...           " << flush;
    auto smoothed = apply_gaussian(ndvi, kernel, H, W);
    double conv_time = elapsed_sec(t0);
    cout << conv_time << " s\n";

    cout << "[4/4] Computing statistics ...                  ";
    Stats s = compute_stats(smoothed);
    cout << "done\n";

    double total = ndvi_time + conv_time;

    cout << "\n┌─────────────────────────────────┐\n"
         << "│  Results                        │\n"
         << "├─────────────────────────────────┤\n"
         << fixed << setprecision(4)
         << "│  NDVI min   : " << setw(8) << s.mn   << "           │\n"
         << "│  NDVI max   : " << setw(8) << s.mx   << "           │\n"
         << "│  NDVI mean  : " << setw(8) << s.mean << "           │\n"
         << "├─────────────────────────────────┤\n"
         << setprecision(3)
         << "│  NDVI time  : " << setw(8) << ndvi_time << " s        │\n"
         << "│  Gauss time : " << setw(8) << conv_time << " s        │\n"
         << "│  Total      : " << setw(8) << total     << " s        │\n"
         << "└─────────────────────────────────┘\n";

    ofstream f("sequential_time.txt");
    if (f) f << fixed << setprecision(6) << total << "\n";

    return 0;
}
