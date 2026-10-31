// Original wrapper using the public cuBLASLt and PyTorch extension APIs.
// No timing, tuning loops, input inspection, or reference/harness access.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublasLt.h>
#include <cuda_runtime_api.h>
#include <array>
#include <cstdint>
#include <memory>
#include <sstream>
#include <string>

namespace py = pybind11;

static void check_lt(cublasStatus_t status, const char* operation) {
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS,
                operation, " failed: cuBLAS status ", static_cast<int>(status));
}

class LtPlan {
    cublasLtHandle_t handle_ = nullptr;
    cublasLtMatmulDesc_t operation_ = nullptr;
    cublasLtMatrixLayout_t a_ = nullptr, b_ = nullptr, c_ = nullptr;
    cublasLtMatmulPreference_t preference_ = nullptr;
    cublasLtMatmulAlgo_t algorithm_{};
    int64_t m_ = 0, n_ = 0, k_ = 0;
    int device_ = 0;
    bool transposed_storage_ = false;
    bool ready_ = false;
    size_t workspace_size_ = 0;
    int algorithm_id_ = -1, tile_ = -1, stages_ = -1;
    uint32_t split_k_ = 0, reduction_ = 0;

public:
    LtPlan() = default;
    LtPlan(const LtPlan&) = delete;
    LtPlan& operator=(const LtPlan&) = delete;

    ~LtPlan() noexcept {
        try {
            c10::cuda::CUDAGuard guard(c10::Device(c10::kCUDA, device_));
            if (preference_) cublasLtMatmulPreferenceDestroy(preference_);
            if (c_) cublasLtMatrixLayoutDestroy(c_);
            if (b_) cublasLtMatrixLayoutDestroy(b_);
            if (a_) cublasLtMatrixLayoutDestroy(a_);
            if (operation_) cublasLtMatmulDescDestroy(operation_);
            if (handle_) cublasLtDestroy(handle_);
        } catch (...) {
            // Python/CUDA teardown must not throw through a destructor.
        }
    }

    void initialize(int64_t m, int64_t n, int64_t k,
                    bool transposed_storage, int rank, int64_t workspace_budget) {
        TORCH_CHECK(m > 0 && n > 0 && k > 0 && rank >= 0 && rank < 32);
        TORCH_CHECK(workspace_budget >= 0);
        m_ = m; n_ = n; k_ = k; transposed_storage_ = transposed_storage;
        TORCH_CHECK(cudaGetDevice(&device_) == cudaSuccess, "cudaGetDevice failed");
        check_lt(cublasLtCreate(&handle_), "create handle");
        check_lt(cublasLtMatmulDescCreate(&operation_, CUBLAS_COMPUTE_32F,
                                         CUDA_R_32F), "create operation");

        // Row-major Y[M,N] is column-major Y^T[N,M]. Compute W * X^T.
        // Original W[N,K] storage is column-major [K,N], requiring transpose.
        // Prepared W^T[K,N] storage is column-major [N,K], requiring no transpose.
        cublasOperation_t trans_a = transposed_storage ? CUBLAS_OP_N : CUBLAS_OP_T;
        cublasOperation_t trans_b = CUBLAS_OP_N;
        check_lt(cublasLtMatmulDescSetAttribute(operation_, CUBLASLT_MATMUL_DESC_TRANSA,
                                              &trans_a, sizeof(trans_a)), "set transpose A");
        check_lt(cublasLtMatmulDescSetAttribute(operation_, CUBLASLT_MATMUL_DESC_TRANSB,
                                              &trans_b, sizeof(trans_b)), "set transpose B");
        check_lt(cublasLtMatrixLayoutCreate(&a_, CUDA_R_16BF,
                  transposed_storage ? n : k, transposed_storage ? k : n,
                  transposed_storage ? n : k), "create A layout");
        check_lt(cublasLtMatrixLayoutCreate(&b_, CUDA_R_16BF, k, m, k), "create B layout");
        check_lt(cublasLtMatrixLayoutCreate(&c_, CUDA_R_16BF, n, m, n), "create C/D layout");
        check_lt(cublasLtMatmulPreferenceCreate(&preference_), "create preference");
        size_t budget = static_cast<size_t>(workspace_budget);
        check_lt(cublasLtMatmulPreferenceSetAttribute(preference_,
                  CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &budget, sizeof(budget)),
                  "set workspace budget");

        // Python dispatch checks pointer alignment and falls back otherwise.
        uint32_t alignment = 256;
        for (auto attr : {CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,
                          CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
                          CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,
                          CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES}) {
            check_lt(cublasLtMatmulPreferenceSetAttribute(preference_, attr,
                      &alignment, sizeof(alignment)), "set alignment preference");
        }
        std::array<cublasLtMatmulHeuristicResult_t, 32> choices{};
        int returned = 0;
        auto status = cublasLtMatmulAlgoGetHeuristic(handle_, operation_, a_, b_, c_, c_,
                       preference_, static_cast<int>(choices.size()), choices.data(), &returned);
        if (status == CUBLAS_STATUS_NOT_SUPPORTED) return;
        check_lt(status, "query heuristic choices");
        int eligible = 0;
        for (int i = 0; i < returned; ++i) {
            const auto& choice = choices[i];
            if (choice.state != CUBLAS_STATUS_SUCCESS || choice.workspaceSize > budget) continue;
            uint32_t scheme = 0;
            size_t written = 0;
            check_lt(cublasLtMatmulAlgoConfigGetAttribute(&choice.algo,
                      CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &scheme, sizeof(scheme), &written),
                      "read reduction scheme");
            // Do not use BF16 output-type or in-place intermediate reductions.
            if (scheme != CUBLASLT_REDUCTION_SCHEME_NONE &&
                scheme != CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE) continue;
            if (eligible++ != rank) continue;
            algorithm_ = choice.algo;
            workspace_size_ = choice.workspaceSize;
            reduction_ = scheme;
            check_lt(cublasLtMatmulAlgoConfigGetAttribute(&algorithm_, CUBLASLT_ALGO_CONFIG_ID,
                      &algorithm_id_, sizeof(algorithm_id_), &written), "read algorithm id");
            check_lt(cublasLtMatmulAlgoConfigGetAttribute(&algorithm_, CUBLASLT_ALGO_CONFIG_TILE_ID,
                      &tile_, sizeof(tile_), &written), "read tile id");
            check_lt(cublasLtMatmulAlgoConfigGetAttribute(&algorithm_, CUBLASLT_ALGO_CONFIG_STAGES_ID,
                      &stages_, sizeof(stages_), &written), "read stages id");
            check_lt(cublasLtMatmulAlgoConfigGetAttribute(&algorithm_, CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
                      &split_k_, sizeof(split_k_), &written), "read split K");
            ready_ = true;
            break;
        }
    }

    bool ready() const { return ready_; }
    std::string info() const {
        std::ostringstream out;
        out << "ready=" << ready_ << ",algo=" << algorithm_id_ << ",tile=" << tile_
            << ",stages=" << stages_ << ",splitK=" << split_k_ << ",reduction=" << reduction_
            << ",workspace=" << workspace_size_;
        return out.str();
    }

    at::Tensor run(const at::Tensor& x, const at::Tensor& weight) const {
        TORCH_CHECK(ready_, "No usable cuBLASLt algorithm for this plan");
        TORCH_CHECK(x.is_cuda() && weight.is_cuda() && x.device() == weight.device());
        TORCH_CHECK(x.get_device() == device_, "Plan belongs to another GPU");
        TORCH_CHECK(x.scalar_type() == at::kBFloat16 && weight.scalar_type() == at::kBFloat16);
        TORCH_CHECK(x.is_contiguous() && weight.is_contiguous());
        TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 256 == 0 &&
                    reinterpret_cast<uintptr_t>(weight.data_ptr()) % 256 == 0);
        TORCH_CHECK(x.dim() == 2 && x.size(0) == m_ && x.size(1) == k_);
        TORCH_CHECK(weight.dim() == 2 &&
                    weight.size(0) == (transposed_storage_ ? k_ : n_) &&
                    weight.size(1) == (transposed_storage_ ? n_ : k_));
        c10::cuda::CUDAGuard guard(x.device());
        auto output = at::empty({m_, n_}, x.options());
        // Each invocation owns its workspace, including under graph capture.
        // PyTorch's stream-aware allocator keeps asynchronous uses safe.
        auto scratch = at::empty({static_cast<int64_t>(workspace_size_)},
                                 x.options().dtype(at::kByte));
        const float alpha = 1.0f, beta = 0.0f;
        auto stream = c10::cuda::getCurrentCUDAStream(device_);
        check_lt(cublasLtMatmul(handle_, operation_, &alpha, weight.data_ptr(), a_,
                  x.data_ptr(), b_, &beta, output.data_ptr(), c_, output.data_ptr(), c_,
                  &algorithm_, workspace_size_ ? scratch.data_ptr() : nullptr,
                  workspace_size_, stream.stream()), "cuBLASLt matmul");
        return output;
    }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    py::class_<LtPlan, std::shared_ptr<LtPlan>>(module, "Plan")
        .def(py::init([](int64_t m, int64_t n, int64_t k, bool transposed,
                         int rank, int64_t workspace) {
            auto plan = std::make_shared<LtPlan>();
            plan->initialize(m, n, k, transposed, rank, workspace);
            return plan;
        }))
        .def("ready", &LtPlan::ready)
        .def("info", &LtPlan::info)
        .def("run", &LtPlan::run);
    module.def("version", []() { return cublasLtGetVersion(); });
}
