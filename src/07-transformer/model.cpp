/*
 * Transformer encoder forward pass in C++17 — no external dependencies.
 * Covers: matrix multiply, softmax, layer norm, scaled dot-product attention,
 *         FFN, full encoder layer, sinusoidal PE.
 *
 * Compile: g++ -O2 -std=c++17 -o model_cpp model.cpp && ./model_cpp
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <numeric>
#include <algorithm>
#include <random>
#include <iomanip>
#include <string>

// ---------------------------------------------------------------------------
// Matrix type: row-major 2D array stored as 1D vector
// ---------------------------------------------------------------------------

struct Matrix {
    int rows, cols;
    std::vector<double> data;

    Matrix(int r, int c, double fill = 0.0)
        : rows(r), cols(c), data(r * c, fill) {}

    double& at(int r, int c)       { return data[r * cols + c]; }
    double  at(int r, int c) const { return data[r * cols + c]; }

    // Row slice [r, r+1, ..., r+n-1]
    Matrix row_slice(int r, int n) const {
        Matrix out(n, cols);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < cols; ++j)
                out.at(i, j) = at(r + i, j);
        return out;
    }
};

// ---------------------------------------------------------------------------
// Linear algebra
// ---------------------------------------------------------------------------

// C = A @ B
Matrix matmul(const Matrix& A, const Matrix& B) {
    assert(A.cols == B.rows);
    Matrix C(A.rows, B.cols, 0.0);
    for (int i = 0; i < A.rows; ++i)
        for (int k = 0; k < A.cols; ++k)
            for (int j = 0; j < B.cols; ++j)
                C.at(i, j) += A.at(i, k) * B.at(k, j);
    return C;
}

// C = A @ B^T
Matrix matmul_BT(const Matrix& A, const Matrix& B) {
    assert(A.cols == B.cols);
    Matrix C(A.rows, B.rows, 0.0);
    for (int i = 0; i < A.rows; ++i)
        for (int j = 0; j < B.rows; ++j)
            for (int k = 0; k < A.cols; ++k)
                C.at(i, j) += A.at(i, k) * B.at(j, k);
    return C;
}

Matrix add(const Matrix& A, const Matrix& B) {
    assert(A.rows == B.rows && A.cols == B.cols);
    Matrix C(A.rows, A.cols);
    for (size_t i = 0; i < A.data.size(); ++i)
        C.data[i] = A.data[i] + B.data[i];
    return C;
}

// Add bias vector (broadast over rows)
Matrix add_bias(const Matrix& A, const std::vector<double>& b) {
    assert((int)b.size() == A.cols);
    Matrix C = A;
    for (int i = 0; i < A.rows; ++i)
        for (int j = 0; j < A.cols; ++j)
            C.at(i, j) += b[j];
    return C;
}

Matrix scale(const Matrix& A, double s) {
    Matrix C = A;
    for (auto& v : C.data) v *= s;
    return C;
}

// ---------------------------------------------------------------------------
// Activations
// ---------------------------------------------------------------------------

// Row-wise softmax (in-place on a copy)
Matrix softmax_rows(Matrix M) {
    for (int i = 0; i < M.rows; ++i) {
        double mx = *std::max_element(M.data.begin() + i * M.cols,
                                      M.data.begin() + (i + 1) * M.cols);
        double sum = 0.0;
        for (int j = 0; j < M.cols; ++j) {
            M.at(i, j) = std::exp(M.at(i, j) - mx);
            sum += M.at(i, j);
        }
        for (int j = 0; j < M.cols; ++j) M.at(i, j) /= sum;
    }
    return M;
}

Matrix relu(Matrix M) {
    for (auto& v : M.data) v = std::max(0.0, v);
    return M;
}

// ---------------------------------------------------------------------------
// Layer Norm
// ---------------------------------------------------------------------------

Matrix layer_norm(const Matrix& X,
                  const std::vector<double>& gamma,
                  const std::vector<double>& beta,
                  double eps = 1e-5) {
    Matrix out(X.rows, X.cols);
    for (int i = 0; i < X.rows; ++i) {
        double mu = 0.0;
        for (int j = 0; j < X.cols; ++j) mu += X.at(i, j);
        mu /= X.cols;
        double var = 0.0;
        for (int j = 0; j < X.cols; ++j) {
            double d = X.at(i, j) - mu;
            var += d * d;
        }
        var /= X.cols;
        double std_inv = 1.0 / std::sqrt(var + eps);
        for (int j = 0; j < X.cols; ++j)
            out.at(i, j) = gamma[j] * (X.at(i, j) - mu) * std_inv + beta[j];
    }
    return out;
}

// ---------------------------------------------------------------------------
// Sinusoidal PE
// ---------------------------------------------------------------------------

Matrix sinusoidal_pe(int max_len, int d_model) {
    Matrix pe(max_len, d_model, 0.0);
    for (int pos = 0; pos < max_len; ++pos) {
        for (int i = 0; i < d_model / 2; ++i) {
            double w = 1.0 / std::pow(10000.0, 2.0 * i / d_model);
            pe.at(pos, 2 * i)     = std::sin(pos * w);
            pe.at(pos, 2 * i + 1) = std::cos(pos * w);
        }
    }
    return pe;
}

// ---------------------------------------------------------------------------
// Scaled Dot-Product Attention
// ---------------------------------------------------------------------------

// Q, K, V: (seq, d_k)  →  output (seq, d_v)
Matrix attention(const Matrix& Q, const Matrix& K, const Matrix& V,
                 bool causal = false) {
    double scale_f = 1.0 / std::sqrt(static_cast<double>(Q.cols));
    Matrix scores = matmul_BT(Q, K);            // (seq_q, seq_k)
    scores = ::scale(scores, scale_f);

    // Optional causal mask (upper triangle → -inf)
    if (causal) {
        for (int i = 0; i < scores.rows; ++i)
            for (int j = i + 1; j < scores.cols; ++j)
                scores.at(i, j) = -1e9;
    }

    Matrix W = softmax_rows(scores);            // (seq_q, seq_k)
    return matmul(W, V);                        // (seq_q, d_v)
}

// ---------------------------------------------------------------------------
// Multi-Head Attention (simplified: single-head for clarity)
// ---------------------------------------------------------------------------

// Full MHA would split Q,K,V into h heads, apply attention per head, concat.
// Here we implement 1-head to keep the C++ readable and dependency-free.

struct MHAWeights {
    Matrix WQ, WK, WV, WO;
    int n_heads;
    MHAWeights(int d_model, int h, std::mt19937& rng)
        : WQ(d_model, d_model), WK(d_model, d_model),
          WV(d_model, d_model), WO(d_model, d_model), n_heads(h) {
        std::uniform_real_distribution<double> dist(-0.1, 0.1);
        for (auto* M : {&WQ, &WK, &WV, &WO})
            for (auto& v : M->data) v = dist(rng);
    }
};

Matrix mha_forward(const Matrix& X, const MHAWeights& W, bool causal = false) {
    int T = X.rows, D = X.cols;
    int h = W.n_heads, dk = D / h;
    Matrix Q = matmul(X, W.WQ);  // (T, D)
    Matrix K = matmul(X, W.WK);
    Matrix V = matmul(X, W.WV);

    // Split into heads, attend, concat
    Matrix concat_out(T, D, 0.0);
    for (int head = 0; head < h; ++head) {
        // Extract head's columns: [head*dk .. (head+1)*dk)
        Matrix Qh(T, dk), Kh(T, dk), Vh(T, dk);
        for (int t = 0; t < T; ++t)
            for (int d = 0; d < dk; ++d) {
                Qh.at(t, d) = Q.at(t, head * dk + d);
                Kh.at(t, d) = K.at(t, head * dk + d);
                Vh.at(t, d) = V.at(t, head * dk + d);
            }
        Matrix head_out = attention(Qh, Kh, Vh, causal);
        for (int t = 0; t < T; ++t)
            for (int d = 0; d < dk; ++d)
                concat_out.at(t, head * dk + d) = head_out.at(t, d);
    }
    return matmul(concat_out, W.WO);
}

// ---------------------------------------------------------------------------
// FFN
// ---------------------------------------------------------------------------

struct FFNWeights {
    Matrix W1, W2;
    std::vector<double> b1, b2;
    FFNWeights(int d_model, int d_ff, std::mt19937& rng)
        : W1(d_model, d_ff), W2(d_ff, d_model),
          b1(d_ff, 0.0), b2(d_model, 0.0) {
        std::uniform_real_distribution<double> dist(-0.1, 0.1);
        for (auto& v : W1.data) v = dist(rng);
        for (auto& v : W2.data) v = dist(rng);
    }
};

Matrix ffn_forward(const Matrix& X, const FFNWeights& W) {
    return matmul(relu(add_bias(matmul(X, W.W1), W.b1)), W.W2);
}

// ---------------------------------------------------------------------------
// Encoder Layer
// ---------------------------------------------------------------------------

struct EncoderLayerWeights {
    MHAWeights  mha;
    FFNWeights  ffn;
    std::vector<double> gamma1, beta1, gamma2, beta2;

    EncoderLayerWeights(int d_model, int n_heads, int d_ff, std::mt19937& rng)
        : mha(d_model, n_heads, rng), ffn(d_model, d_ff, rng),
          gamma1(d_model, 1.0), beta1(d_model, 0.0),
          gamma2(d_model, 1.0), beta2(d_model, 0.0) {}
};

Matrix encoder_layer(const Matrix& X, const EncoderLayerWeights& W) {
    // Pre-LN + Self-Attention + residual
    Matrix x_n = layer_norm(X, W.gamma1, W.beta1);
    Matrix attn_out = mha_forward(x_n, W.mha);
    Matrix x1 = add(X, attn_out);

    // Pre-LN + FFN + residual
    Matrix x1_n = layer_norm(x1, W.gamma2, W.beta2);
    Matrix ffn_out = ffn_forward(x1_n, W.ffn);
    return add(x1, ffn_out);
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

void print_section(const std::string& title) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "  " << title << "\n";
    std::cout << std::string(60, '=') << "\n";
}

double matrix_max_abs(const Matrix& M) {
    double mx = 0.0;
    for (auto v : M.data) mx = std::max(mx, std::abs(v));
    return mx;
}

double matrix_mean(const Matrix& M) {
    return std::accumulate(M.data.begin(), M.data.end(), 0.0) / M.data.size();
}

bool is_finite(const Matrix& M) {
    for (auto v : M.data)
        if (!std::isfinite(v)) return false;
    return true;
}

void print_matrix_sample(const Matrix& M, int rows = 3, int cols = 4) {
    int show_r = std::min(M.rows, rows);
    int show_c = std::min(M.cols, cols);
    for (int i = 0; i < show_r; ++i) {
        std::cout << "    [";
        for (int j = 0; j < show_c; ++j)
            std::cout << std::fixed << std::setprecision(4) << M.at(i, j)
                      << (j < show_c - 1 ? ", " : "");
        std::cout << (show_c < M.cols ? ", ...]" : "]") << "\n";
    }
}

// Check row sums ≈ 1 (for softmax output)
bool check_row_sums(const Matrix& W, double tol = 1e-6) {
    for (int i = 0; i < W.rows; ++i) {
        double s = 0.0;
        for (int j = 0; j < W.cols; ++j) s += W.at(i, j);
        if (std::abs(s - 1.0) > tol) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-0.5, 0.5);

    const int VOCAB  = 50;
    const int D      = 32;    // d_model
    const int D_FF   = 64;    // d_ff
    const int H      = 4;     // n_heads
    const int T      = 8;     // sequence length
    const int N_LAYERS = 2;

    print_section("SINUSOIDAL POSITIONAL ENCODING");
    Matrix pe = sinusoidal_pe(T, D);
    std::cout << "\n  PE shape: " << pe.rows << " × " << pe.cols << "\n";
    std::cout << "  First 3 positions, first 6 dims:\n";
    for (int i = 0; i < 3; ++i) {
        std::cout << "    pos " << i << ": ";
        for (int j = 0; j < 6; ++j)
            std::cout << std::fixed << std::setprecision(4) << pe.at(i, j) << " ";
        std::cout << "...\n";
    }

    // -----------------------------------------------------------------------
    print_section("SCALED DOT-PRODUCT ATTENTION");
    // -----------------------------------------------------------------------
    Matrix Q(T, D / H), K(T, D / H), V(T, D / H);
    for (auto& v : Q.data) v = dist(rng);
    for (auto& v : K.data) v = dist(rng);
    for (auto& v : V.data) v = dist(rng);

    Matrix attn_out = attention(Q, K, V, false);
    // Compute and show attention weights (for display)
    double sc = 1.0 / std::sqrt((double)(D / H));
    Matrix scores_disp = ::scale(matmul_BT(Q, K), sc);
    Matrix W_disp = softmax_rows(scores_disp);

    std::cout << "\n  Q, K, V shape: " << T << " × " << (D / H) << "\n";
    std::cout << "  Attention output shape: " << attn_out.rows << " × " << attn_out.cols << "\n";
    std::cout << "  Attention weights (first 4×4):\n";
    for (int i = 0; i < 4; ++i) {
        std::cout << "    [";
        for (int j = 0; j < 4; ++j)
            std::cout << std::fixed << std::setprecision(3) << W_disp.at(i, j)
                      << (j < 3 ? ", " : "");
        std::cout << ", ...]\n";
    }
    std::cout << "  Row sums = 1.0: " << (check_row_sums(W_disp) ? "YES" : "NO") << "\n";
    std::cout << "  All finite    : " << (is_finite(attn_out) ? "YES" : "NO") << "\n";

    // -----------------------------------------------------------------------
    print_section("CAUSAL (MASKED) ATTENTION");
    // -----------------------------------------------------------------------
    Matrix attn_causal = attention(Q, K, V, true);
    // Recompute scores with mask for display
    Matrix scores_causal = ::scale(matmul_BT(Q, K), sc);
    for (int i = 0; i < scores_causal.rows; ++i)
        for (int j = i + 1; j < scores_causal.cols; ++j)
            scores_causal.at(i, j) = -1e9;
    Matrix W_causal = softmax_rows(scores_causal);

    std::cout << "\n  Causal attention weights (4×4 excerpt):\n";
    for (int i = 0; i < 4; ++i) {
        std::cout << "    [";
        for (int j = 0; j < 4; ++j)
            std::cout << std::fixed << std::setprecision(3) << W_causal.at(i, j)
                      << (j < 3 ? ", " : "");
        std::cout << ", ...]\n";
    }
    std::cout << "  Upper triangle is zero: ";
    bool upper_zero = true;
    for (int i = 0; i < W_causal.rows; ++i)
        for (int j = i + 1; j < W_causal.cols; ++j)
            if (W_causal.at(i, j) > 1e-8) { upper_zero = false; break; }
    std::cout << (upper_zero ? "YES" : "NO") << "\n";

    // -----------------------------------------------------------------------
    print_section("LAYER NORM");
    // -----------------------------------------------------------------------
    Matrix x_test(3, D);
    for (auto& v : x_test.data) v = dist(rng) * 5.0;
    std::vector<double> gamma(D, 1.0), beta(D, 0.0);
    Matrix x_ln = layer_norm(x_test, gamma, beta);

    std::cout << "\n  Input  sample (row 0, first 6): ";
    for (int j = 0; j < 6; ++j) std::cout << std::fixed << std::setprecision(3) << x_test.at(0, j) << " ";
    std::cout << "\n  Output sample (row 0, first 6): ";
    for (int j = 0; j < 6; ++j) std::cout << std::fixed << std::setprecision(3) << x_ln.at(0, j) << " ";
    std::cout << "\n";
    for (int i = 0; i < std::min(3, x_ln.rows); ++i) {
        double mu = 0.0, var = 0.0;
        for (int j = 0; j < D; ++j) mu += x_ln.at(i, j);
        mu /= D;
        for (int j = 0; j < D; ++j) var += (x_ln.at(i, j) - mu) * (x_ln.at(i, j) - mu);
        var /= D;
        std::cout << "  Row " << i << ": mean=" << std::fixed << std::setprecision(5) << mu
                  << "  std=" << std::sqrt(var) << "\n";
    }

    // -----------------------------------------------------------------------
    print_section("MULTI-HEAD ATTENTION");
    // -----------------------------------------------------------------------
    MHAWeights mha_w(D, H, rng);
    Matrix x_mha(T, D);
    for (auto& v : x_mha.data) v = dist(rng);
    Matrix mha_out = mha_forward(x_mha, mha_w);
    std::cout << "\n  d_model=" << D << ", n_heads=" << H << ", d_k=" << D/H << "\n";
    std::cout << "  Input  shape: " << x_mha.rows << " × " << x_mha.cols << "\n";
    std::cout << "  Output shape: " << mha_out.rows << " × " << mha_out.cols << "\n";
    std::cout << "  All finite  : " << (is_finite(mha_out) ? "YES" : "NO") << "\n";
    std::cout << "  Mean abs val: " << std::fixed << std::setprecision(4)
              << matrix_max_abs(mha_out) << "\n";

    // -----------------------------------------------------------------------
    print_section("FFN SUBLAYER");
    // -----------------------------------------------------------------------
    FFNWeights ffn_w(D, D_FF, rng);
    Matrix ffn_out = ffn_forward(x_mha, ffn_w);
    std::cout << "\n  d_model=" << D << ", d_ff=" << D_FF << "\n";
    std::cout << "  Input  shape: " << x_mha.rows << " × " << x_mha.cols << "\n";
    std::cout << "  Output shape: " << ffn_out.rows << " × " << ffn_out.cols << "\n";
    std::cout << "  All finite  : " << (is_finite(ffn_out) ? "YES" : "NO") << "\n";
    // ReLU: should have non-negative outputs
    double min_val = *std::min_element(ffn_out.data.begin(), ffn_out.data.end());
    std::cout << "  Min val (ReLU applied in W1 output, so final ≥ 0 is not guaranteed): "
              << min_val << "\n";

    // -----------------------------------------------------------------------
    print_section("FULL ENCODER LAYER");
    // -----------------------------------------------------------------------
    EncoderLayerWeights enc_w(D, H, D_FF, rng);
    Matrix x_enc(T, D);
    for (auto& v : x_enc.data) v = dist(rng);
    // Add sinusoidal PE
    for (int i = 0; i < T; ++i)
        for (int j = 0; j < D; ++j)
            x_enc.at(i, j) += pe.at(i, j);

    Matrix enc_out = encoder_layer(x_enc, enc_w);
    std::cout << "\n  Input  shape: " << x_enc.rows << " × " << x_enc.cols << "\n";
    std::cout << "  Output shape: " << enc_out.rows << " × " << enc_out.cols << "\n";
    std::cout << "  All finite  : " << (is_finite(enc_out) ? "YES" : "NO") << "\n";
    std::cout << "  Mean        : " << std::fixed << std::setprecision(4)
              << matrix_mean(enc_out) << "\n";
    std::cout << "\n  Output sample (first 3 tokens, first 4 dims):\n";
    print_matrix_sample(enc_out, 3, 4);

    // -----------------------------------------------------------------------
    print_section("STACKED ENCODER (" + std::to_string(N_LAYERS) + " layers)");
    // -----------------------------------------------------------------------
    std::vector<EncoderLayerWeights> layers;
    for (int l = 0; l < N_LAYERS; ++l)
        layers.emplace_back(D, H, D_FF, rng);

    Matrix x_stack(T, D);
    for (auto& v : x_stack.data) v = dist(rng);
    for (int i = 0; i < T; ++i)
        for (int j = 0; j < D; ++j)
            x_stack.at(i, j) += pe.at(i, j);

    for (int l = 0; l < N_LAYERS; ++l) {
        x_stack = encoder_layer(x_stack, layers[l]);
        std::cout << "  Layer " << l+1 << " output: mean="
                  << std::fixed << std::setprecision(4) << matrix_mean(x_stack)
                  << "  max_abs=" << matrix_max_abs(x_stack)
                  << "  finite=" << (is_finite(x_stack) ? "YES" : "NO") << "\n";
    }

    // -----------------------------------------------------------------------
    print_section("COMPLEXITY ANALYSIS (C++ perspective)");
    // -----------------------------------------------------------------------
    std::cout << R"(
  Attention QK^T: O(T^2 * d_k) per head, O(T^2 * d_model) total
  FFN           : O(T * d_model * d_ff)
  Crossover at T ≈ d_ff = 4*d_model (for this model: T ≈ )" << D_FF << R"()

  This C++ implementation uses naive O(N^3) matmul.
  Production inference uses:
    - BLAS (OpenBLAS, MKL) for GEMM: highly optimised cache-aware kernels
    - CUDA cublas for GPU execution
    - FlashAttention: fused attention kernel, O(N) memory via tiling

  Memory layout: row-major (C order) — optimal for row-wise operations.
  PyTorch uses row-major by default; BLAS routines are column-major (Fortran).
  PyTorch handles this via transposition flags in GEMM calls.
)";

    // -----------------------------------------------------------------------
    print_section("PARAMETER COUNT");
    // -----------------------------------------------------------------------
    long long attn_params = 4LL * D * D;          // WQ + WK + WV + WO
    long long ffn_params  = 2LL * D * D_FF + D_FF + D;  // W1,b1,W2,b2
    long long ln_params   = 4LL * D;              // two LNs: gamma + beta each
    long long per_layer   = attn_params + ffn_params + ln_params;
    long long total       = N_LAYERS * per_layer;

    std::cout << "\n  d_model=" << D << ", d_ff=" << D_FF << ", n_layers=" << N_LAYERS << "\n";
    std::cout << "  Attention params per layer : " << attn_params << "\n";
    std::cout << "  FFN params per layer       : " << ffn_params  << "\n";
    std::cout << "  LayerNorm params per layer : " << ln_params   << "\n";
    std::cout << "  Total params per layer     : " << per_layer   << "\n";
    std::cout << "  Total (all layers)         : " << total       << "\n";
    std::cout << "  Theory (12 * L * d^2)      : " << 12LL * N_LAYERS * D * D << "\n";

    std::cout << "\n[C++ Transformer encoder completed successfully]\n";
    return 0;
}
