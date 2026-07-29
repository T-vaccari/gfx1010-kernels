#include "residual_layer_norm.h"

#include <ATen/Context.h>
#include <ATen/core/Generator.h>
#include <ATen/hip/HIPGeneratorImpl.h>
#include <ATen/hip/PhiloxUtils.cuh>
#include <c10/cuda/CUDAGuard.h>
#include <c10/hip/HIPGraphsC10Utils.h>
#include <c10/hip/HIPStream.h>
#include <c10/hip/HIPException.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#include <algorithm>
#include <mutex>
#include <type_traits>


namespace {

constexpr int kWaveSize = 32;
constexpr int kMaxWaves = 8;


void check_forward_inputs(
    const at::Tensor& x,
    const at::Tensor& branch,
    const at::Tensor& weight,
    const at::Tensor& bias,
    double dropout_p,
    double eps,
    int64_t threads) {
    TORCH_CHECK(x.is_cuda(), "x must be a HIP tensor");
    TORCH_CHECK(
        branch.is_cuda() && weight.is_cuda() && bias.is_cuda(),
        "branch, weight and bias must be HIP tensors");
    TORCH_CHECK(
        x.device() == branch.device()
            && x.device() == weight.device()
            && x.device() == bias.device(),
        "all tensors must be on the same HIP device");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must use float32");
    TORCH_CHECK(
        branch.scalar_type() == at::kHalf
            || branch.scalar_type() == at::kFloat,
        "branch must use float16 or float32");
    TORCH_CHECK(
        weight.scalar_type() == at::kFloat
            && bias.scalar_type() == at::kFloat,
        "weight and bias must use float32");
    TORCH_CHECK(x.dim() >= 2, "x must have at least two dimensions");
    TORCH_CHECK(branch.sizes() == x.sizes(), "branch must match x");
    const int64_t hidden = x.size(-1);
    TORCH_CHECK(
        weight.sizes() == at::IntArrayRef({hidden})
            && bias.sizes() == at::IntArrayRef({hidden}),
        "weight and bias must match the hidden size");
    TORCH_CHECK(
        x.is_contiguous() && branch.is_contiguous()
            && weight.is_contiguous() && bias.is_contiguous(),
        "all tensors must be contiguous");
    TORCH_CHECK(x.numel() > 0, "empty tensors are unsupported");
    TORCH_CHECK(
        hidden == 128 || hidden == 256 || hidden == 384
            || hidden == 512 || hidden == 768 || hidden == 1024,
        "unsupported native hidden size: ",
        hidden);
    TORCH_CHECK(
        threads == 128 || threads == 192 || threads == 256,
        "threads must be 128, 192 or 256");
    TORCH_CHECK(
        dropout_p >= 0.0 && dropout_p < 1.0,
        "dropout_p must be in [0, 1)");
    TORCH_CHECK(eps > 0.0, "eps must be positive");
}


void check_backward_inputs(
    const at::Tensor& grad_updated,
    const at::Tensor& grad_normalized,
    const at::Tensor& updated,
    const at::Tensor& weight,
    const at::Tensor& mean,
    const at::Tensor& rstd,
    const at::Tensor& dropout_mask,
    double dropout_p,
    bool training,
    int64_t threads,
    int64_t block_rows,
    int64_t parameter_threads,
    int64_t parameter_groups) {
    TORCH_CHECK(updated.is_cuda(), "updated must be a HIP tensor");
    TORCH_CHECK(
        grad_updated.device() == updated.device()
            && grad_normalized.device() == updated.device()
            && weight.device() == updated.device()
            && mean.device() == updated.device()
            && rstd.device() == updated.device(),
        "all backward tensors must be on the same HIP device");
    TORCH_CHECK(
        grad_updated.scalar_type() == at::kFloat
            && grad_normalized.scalar_type() == at::kFloat
            && updated.scalar_type() == at::kFloat
            && weight.scalar_type() == at::kFloat
            && mean.scalar_type() == at::kFloat
            && rstd.scalar_type() == at::kFloat,
        "all backward tensors must use float32");
    TORCH_CHECK(
        updated.dim() >= 2,
        "updated must have at least two dimensions");
    TORCH_CHECK(updated.numel() > 0, "empty tensors are unsupported");
    TORCH_CHECK(
        grad_updated.sizes() == updated.sizes()
            && grad_normalized.sizes() == updated.sizes(),
        "output gradients must match updated");
    const int64_t hidden = updated.size(-1);
    TORCH_CHECK(
        hidden == 128 || hidden == 256 || hidden == 384
            || hidden == 512 || hidden == 768 || hidden == 1024,
        "unsupported native hidden size: ",
        hidden);
    const int64_t row_count = updated.numel() / hidden;
    TORCH_CHECK(
        weight.sizes() == at::IntArrayRef({hidden}),
        "weight must match the hidden size");
    TORCH_CHECK(
        mean.numel() == row_count && rstd.numel() == row_count,
        "mean and rstd must match the row count");
    TORCH_CHECK(
        grad_updated.is_contiguous() && grad_normalized.is_contiguous()
            && updated.is_contiguous() && weight.is_contiguous()
            && mean.is_contiguous() && rstd.is_contiguous(),
        "all backward tensors must be contiguous");
    TORCH_CHECK(
        threads == 128 || threads == 192 || threads == 256,
        "threads must be 128, 192 or 256");
    TORCH_CHECK(
        block_rows == 16 || block_rows == 32
            || block_rows == 64 || block_rows == 128,
        "block_rows must be 16, 32, 64 or 128");
    TORCH_CHECK(
        parameter_threads == 64 || parameter_threads == 128
            || parameter_threads == 256,
        "parameter_threads must be 64, 128 or 256");
    TORCH_CHECK(
        parameter_groups > 0 && parameter_groups <= row_count,
        "parameter_groups must be in [1, row_count]");
    TORCH_CHECK(
        dropout_p >= 0.0 && dropout_p < 1.0,
        "dropout_p must be in [0, 1)");
    const bool dropout = training && dropout_p > 0.0;
    if (dropout) {
        TORCH_CHECK(
            dropout_mask.is_cuda()
                && dropout_mask.device() == updated.device(),
            "dropout_mask must be on the same HIP device");
        TORCH_CHECK(
            dropout_mask.scalar_type() == at::kInt
                && dropout_mask.is_contiguous(),
            "dropout_mask must be a contiguous int32 tensor");
        TORCH_CHECK(
            dropout_mask.numel() == row_count * (hidden / kWaveSize),
            "dropout_mask has the wrong size");
    }
}


template <int Threads>
__device__ __forceinline__ float block_sum(float value) {
    static_assert(Threads % kWaveSize == 0);
    __shared__ float wave_values[kMaxWaves];
    __shared__ float result;

#pragma unroll
    for (int offset = kWaveSize / 2; offset > 0; offset /= 2) {
        value += __shfl_down(value, offset, kWaveSize);
    }

    const int lane = threadIdx.x & (kWaveSize - 1);
    const int wave = threadIdx.x / kWaveSize;
    if (lane == 0) {
        wave_values[wave] = value;
    }
    __syncthreads();

    if (wave == 0) {
        float wave_sum =
            lane < Threads / kWaveSize ? wave_values[lane] : 0.0f;
#pragma unroll
        for (int offset = kWaveSize / 2; offset > 0; offset /= 2) {
            wave_sum += __shfl_down(wave_sum, offset, kWaveSize);
        }
        if (lane == 0) {
            result = wave_sum;
        }
    }
    __syncthreads();
    return result;
}


template <int Threads>
__device__ __forceinline__ float2 block_sum_pair(
    float first,
    float second) {
    static_assert(Threads % kWaveSize == 0);
    __shared__ float2 wave_values[kMaxWaves];
    __shared__ float2 result;

#pragma unroll
    for (int offset = kWaveSize / 2; offset > 0; offset /= 2) {
        first += __shfl_down(first, offset, kWaveSize);
        second += __shfl_down(second, offset, kWaveSize);
    }

    const int lane = threadIdx.x & (kWaveSize - 1);
    const int wave = threadIdx.x / kWaveSize;
    if (lane == 0) {
        wave_values[wave] = make_float2(first, second);
    }
    __syncthreads();

    if (wave == 0) {
        float2 wave_sum =
            lane < Threads / kWaveSize
            ? wave_values[lane]
            : make_float2(0.0f, 0.0f);
#pragma unroll
        for (int offset = kWaveSize / 2; offset > 0; offset /= 2) {
            wave_sum.x += __shfl_down(
                wave_sum.x,
                offset,
                kWaveSize);
            wave_sum.y += __shfl_down(
                wave_sum.y,
                offset,
                kWaveSize);
        }
        if (lane == 0) {
            result = wave_sum;
        }
    }
    __syncthreads();
    return result;
}


template <typename Branch>
__device__ __forceinline__ float load_branch(
    const Branch* branch,
    int64_t index) {
    return static_cast<float>(branch[index]);
}


template <>
__device__ __forceinline__ float load_branch<__half>(
    const __half* branch,
    int64_t index) {
    return __half2float(branch[index]);
}


template <typename Branch>
__device__ __forceinline__ float round_branch(float value) {
    return value;
}


template <>
__device__ __forceinline__ float round_branch<__half>(float value) {
    return __half2float(__float2half_rn(value));
}


template <typename Branch>
__device__ __forceinline__ void store_branch(
    Branch* branch,
    int64_t index,
    float value) {
    branch[index] = static_cast<Branch>(value);
}


template <>
__device__ __forceinline__ void store_branch<__half>(
    __half* branch,
    int64_t index,
    float value) {
    branch[index] = __float2half_rn(value);
}


struct alignas(8) Half4 {
    __half2 low;
    __half2 high;
};


template <typename Branch>
__device__ __forceinline__ float4 load_branch4(
    const Branch* branch);


template <>
__device__ __forceinline__ float4 load_branch4<float>(
    const float* branch) {
    return *reinterpret_cast<const float4*>(branch);
}


template <>
__device__ __forceinline__ float4 load_branch4<__half>(
    const __half* branch) {
    const Half4 packed = *reinterpret_cast<const Half4*>(branch);
    const float2 low = __half22float2(packed.low);
    const float2 high = __half22float2(packed.high);
    return make_float4(low.x, low.y, high.x, high.y);
}


template <typename Branch>
__device__ __forceinline__ void store_branch4(
    Branch* branch,
    const float4& value);


template <>
__device__ __forceinline__ void store_branch4<float>(
    float* branch,
    const float4& value) {
    *reinterpret_cast<float4*>(branch) = value;
}


template <>
__device__ __forceinline__ void store_branch4<__half>(
    __half* branch,
    const float4& value) {
    const Half4 packed{
        __halves2half2(
            __float2half_rn(value.x),
            __float2half_rn(value.y)),
        __halves2half2(
            __float2half_rn(value.z),
            __float2half_rn(value.w)),
    };
    *reinterpret_cast<Half4*>(branch) = packed;
}


template <int Hidden, typename Branch>
__global__ __launch_bounds__(Hidden / 4)
void residual_layer_norm_forward_vector4_kernel(
    const float* __restrict__ x,
    const Branch* __restrict__ branch,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ updated,
    float* __restrict__ normalized,
    float* __restrict__ mean,
    float* __restrict__ rstd,
    float eps) {
    constexpr int Threads = Hidden / 4;
    constexpr float InvHidden = 1.0f / static_cast<float>(Hidden);
    static_assert(Hidden % 4 == 0);
    static_assert(
        Threads == 32 || Threads == 64 || Threads == 96
            || Threads == 128 || Threads == 192 || Threads == 256);
    const int row = blockIdx.x;
    const int vector_column = threadIdx.x;
    const int64_t row_offset = static_cast<int64_t>(row) * Hidden;
    const float* x_row = x + row_offset;
    const Branch* branch_row = branch + row_offset;
    float* updated_row = updated + row_offset;
    float* normalized_row = normalized + row_offset;

    const float4 x_values =
        reinterpret_cast<const float4*>(x_row)[vector_column];
    const float4 branch_values =
        load_branch4(branch_row + vector_column * 4);
    const float4 values = make_float4(
        x_values.x + branch_values.x,
        x_values.y + branch_values.y,
        x_values.z + branch_values.z,
        x_values.w + branch_values.w);
    reinterpret_cast<float4*>(updated_row)[vector_column] = values;

    const float local_sum =
        (values.x + values.y) + (values.z + values.w);
    const float row_mean =
        block_sum<Threads>(local_sum) * InvHidden;
    const float4 centered = make_float4(
        values.x - row_mean,
        values.y - row_mean,
        values.z - row_mean,
        values.w - row_mean);
    const float local_square_sum =
        (centered.x * centered.x + centered.y * centered.y)
        + (centered.z * centered.z + centered.w * centered.w);
    const float variance =
        block_sum<Threads>(local_square_sum) * InvHidden;
    const float row_rstd = rsqrtf(variance + eps);

    if (threadIdx.x == 0) {
        mean[row] = row_mean;
        rstd[row] = row_rstd;
    }

    const float4 weights =
        reinterpret_cast<const float4*>(weight)[vector_column];
    const float4 biases =
        reinterpret_cast<const float4*>(bias)[vector_column];
    const float4 normalized_values = make_float4(
        centered.x * row_rstd * weights.x + biases.x,
        centered.y * row_rstd * weights.y + biases.y,
        centered.z * row_rstd * weights.z + biases.z,
        centered.w * row_rstd * weights.w + biases.w);
    reinterpret_cast<float4*>(
        normalized_row)[vector_column] = normalized_values;
}


template <int Hidden, int Threads, typename Branch, bool Dropout>
__global__ __launch_bounds__(Threads)
void residual_layer_norm_forward_kernel(
    const float* __restrict__ x,
    const Branch* __restrict__ branch,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ updated,
    float* __restrict__ normalized,
    float* __restrict__ mean,
    float* __restrict__ rstd,
    uint32_t* __restrict__ dropout_mask,
    float dropout_scale,
    uint32_t dropout_threshold,
    float eps,
    uint64_t seed,
    uint64_t offset) {
    constexpr int Items = (Hidden + Threads - 1) / Threads;
    constexpr float InvHidden = 1.0f / static_cast<float>(Hidden);
    const int row = blockIdx.x;
    const int64_t row_offset = static_cast<int64_t>(row) * Hidden;
    float values[Items];
    uint32_t random_values[Items];
    float local_sum = 0.0f;
    hiprandStatePhilox4_32_10_t random_state;

    if constexpr (Dropout) {
        hiprand_init(
            seed,
            static_cast<uint64_t>(row) * Threads + threadIdx.x,
            offset,
            &random_state);
        const uint4 first = hiprand4(&random_state);
        random_values[0] = first.x;
        if constexpr (Items > 1) {
            random_values[1] = first.y;
        }
        if constexpr (Items > 2) {
            random_values[2] = first.z;
        }
        if constexpr (Items > 3) {
            random_values[3] = first.w;
        }
        if constexpr (Items > 4) {
            const uint4 second = hiprand4(&random_state);
            random_values[4] = second.x;
            if constexpr (Items > 5) {
                random_values[5] = second.y;
            }
            if constexpr (Items > 6) {
                random_values[6] = second.z;
            }
            if constexpr (Items > 7) {
                random_values[7] = second.w;
            }
        }
    }

#pragma unroll
    for (int item = 0; item < Items; ++item) {
        const int column = threadIdx.x + item * Threads;
        bool keep = column < Hidden;
        float value = 0.0f;
        if (column < Hidden) {
            const int64_t index = row_offset + column;
            float branch_value = load_branch(branch, index);
            if constexpr (Dropout) {
                keep = random_values[item] >= dropout_threshold;
                branch_value = round_branch<Branch>(
                    branch_value * static_cast<float>(keep)
                    * dropout_scale);
            }
            value = x[index] + branch_value;
            updated[index] = value;
        }
        if constexpr (Dropout) {
            const uint32_t bits =
                static_cast<uint32_t>(__ballot(keep));
            const int lane = threadIdx.x & (kWaveSize - 1);
            const int word =
                item * (Threads / kWaveSize) + threadIdx.x / kWaveSize;
            if (lane == 0 && word < Hidden / kWaveSize) {
                dropout_mask[
                    static_cast<int64_t>(row) * (Hidden / kWaveSize) + word
                ] = bits;
            }
        }
        values[item] = value;
        local_sum += value;
    }

    const float row_mean = block_sum<Threads>(local_sum) * InvHidden;
    float local_square_sum = 0.0f;
#pragma unroll
    for (int item = 0; item < Items; ++item) {
        const int column = threadIdx.x + item * Threads;
        if (column < Hidden) {
            const float centered = values[item] - row_mean;
            local_square_sum += centered * centered;
        }
    }
    const float variance =
        block_sum<Threads>(local_square_sum) * InvHidden;
    const float row_rstd = rsqrtf(variance + eps);

    if (threadIdx.x == 0) {
        mean[row] = row_mean;
        rstd[row] = row_rstd;
    }

#pragma unroll
    for (int item = 0; item < Items; ++item) {
        const int column = threadIdx.x + item * Threads;
        if (column < Hidden) {
            const int64_t index = row_offset + column;
            normalized[index] =
                (values[item] - row_mean) * row_rstd * weight[column]
                + bias[column];
        }
    }
}


template <int Hidden, int Threads, typename Branch, bool Dropout>
__global__ __launch_bounds__(Threads)
void residual_layer_norm_backward_input_kernel(
    const float* __restrict__ grad_updated,
    const float* __restrict__ grad_normalized,
    const float* __restrict__ updated,
    const float* __restrict__ weight,
    const float* __restrict__ mean,
    const float* __restrict__ rstd,
    const uint32_t* __restrict__ dropout_mask,
    float* __restrict__ grad_x,
    Branch* __restrict__ grad_branch,
    float dropout_scale) {
    constexpr int Items = (Hidden + Threads - 1) / Threads;
    constexpr float InvHidden = 1.0f / static_cast<float>(Hidden);
    const int row = blockIdx.x;
    const int64_t row_offset = static_cast<int64_t>(row) * Hidden;
    const float row_mean = mean[row];
    const float row_rstd = rstd[row];
    float xhat_values[Items];
    float weighted_grad_values[Items];
    float local_grad_sum = 0.0f;
    float local_projection_sum = 0.0f;
#pragma unroll
    for (int item = 0; item < Items; ++item) {
        const int column = threadIdx.x + item * Threads;
        float xhat = 0.0f;
        float weighted_grad = 0.0f;
        if (column < Hidden) {
            const int64_t index = row_offset + column;
            xhat = (updated[index] - row_mean) * row_rstd;
            weighted_grad = grad_normalized[index] * weight[column];
            local_grad_sum += weighted_grad;
            local_projection_sum += weighted_grad * xhat;
        }
        xhat_values[item] = xhat;
        weighted_grad_values[item] = weighted_grad;
    }

    const float2 sums = block_sum_pair<Threads>(
        local_grad_sum,
        local_projection_sum);
    const float grad_mean = sums.x * InvHidden;
    const float projection_mean = sums.y * InvHidden;

#pragma unroll
    for (int item = 0; item < Items; ++item) {
        const int column = threadIdx.x + item * Threads;
        if (column < Hidden) {
            const int64_t index = row_offset + column;
            const float normalized_grad =
                row_rstd
                * (weighted_grad_values[item] - grad_mean
                   - xhat_values[item] * projection_mean);
            const float total_grad =
                grad_updated[index] + normalized_grad;
            grad_x[index] = total_grad;
            float branch_grad = total_grad;
            if constexpr (Dropout) {
                const uint32_t word = dropout_mask[
                    static_cast<int64_t>(row) * (Hidden / kWaveSize)
                    + column / kWaveSize
                ];
                const float keep = static_cast<float>(
                    (word >> (column & (kWaveSize - 1))) & 1U);
                branch_grad *= keep * dropout_scale;
            }
            store_branch(grad_branch, index, branch_grad);
        }
    }
}


template <int Hidden, typename Branch, bool Dropout>
__global__ __launch_bounds__(Hidden / 4)
void residual_layer_norm_backward_input_vector4_kernel(
    const float* __restrict__ grad_updated,
    const float* __restrict__ grad_normalized,
    const float* __restrict__ updated,
    const float* __restrict__ weight,
    const float* __restrict__ mean,
    const float* __restrict__ rstd,
    const uint32_t* __restrict__ dropout_mask,
    float* __restrict__ grad_x,
    Branch* __restrict__ grad_branch,
    float dropout_scale) {
    constexpr int Threads = Hidden / 4;
    constexpr float InvHidden = 1.0f / static_cast<float>(Hidden);
    static_assert(Hidden % 4 == 0);
    static_assert(
        Threads == 64 || Threads == 96 || Threads == 128
            || Threads == 192 || Threads == 256);
    const int row = blockIdx.x;
    const int vector_column = threadIdx.x;
    const int column = vector_column * 4;
    const int64_t row_offset = static_cast<int64_t>(row) * Hidden;
    const float* grad_updated_row = grad_updated + row_offset;
    const float* grad_normalized_row = grad_normalized + row_offset;
    const float* updated_row = updated + row_offset;
    float* grad_x_row = grad_x + row_offset;
    Branch* grad_branch_row = grad_branch + row_offset;
    const float row_mean = mean[row];
    const float row_rstd = rstd[row];

    const float4 updated_values =
        reinterpret_cast<const float4*>(updated_row)[vector_column];
    const float4 normalized_gradients =
        reinterpret_cast<const float4*>(
            grad_normalized_row)[vector_column];
    const float4 weights =
        reinterpret_cast<const float4*>(weight)[vector_column];
    const float4 xhat_values = make_float4(
        (updated_values.x - row_mean) * row_rstd,
        (updated_values.y - row_mean) * row_rstd,
        (updated_values.z - row_mean) * row_rstd,
        (updated_values.w - row_mean) * row_rstd);
    const float4 weighted_gradients = make_float4(
        normalized_gradients.x * weights.x,
        normalized_gradients.y * weights.y,
        normalized_gradients.z * weights.z,
        normalized_gradients.w * weights.w);
    const float local_grad_sum =
        (weighted_gradients.x + weighted_gradients.y)
        + (weighted_gradients.z + weighted_gradients.w);
    const float local_projection_sum =
        (weighted_gradients.x * xhat_values.x
         + weighted_gradients.y * xhat_values.y)
        + (weighted_gradients.z * xhat_values.z
           + weighted_gradients.w * xhat_values.w);
    const float2 sums = block_sum_pair<Threads>(
        local_grad_sum,
        local_projection_sum);
    const float grad_mean = sums.x * InvHidden;
    const float projection_mean = sums.y * InvHidden;
    const float4 direct_gradients =
        reinterpret_cast<const float4*>(
            grad_updated_row)[vector_column];
    const float4 total_gradients = make_float4(
        direct_gradients.x
            + row_rstd
                * (weighted_gradients.x - grad_mean
                   - xhat_values.x * projection_mean),
        direct_gradients.y
            + row_rstd
                * (weighted_gradients.y - grad_mean
                   - xhat_values.y * projection_mean),
        direct_gradients.z
            + row_rstd
                * (weighted_gradients.z - grad_mean
                   - xhat_values.z * projection_mean),
        direct_gradients.w
            + row_rstd
                * (weighted_gradients.w - grad_mean
                   - xhat_values.w * projection_mean));
    reinterpret_cast<float4*>(grad_x_row)[vector_column] =
        total_gradients;

    float4 branch_gradients = total_gradients;
    if constexpr (Dropout) {
        const uint32_t word = dropout_mask[
            static_cast<int64_t>(row) * (Hidden / kWaveSize)
            + column / kWaveSize
        ];
        const int bit = column & (kWaveSize - 1);
        branch_gradients.x *=
            static_cast<float>((word >> bit) & 1U) * dropout_scale;
        branch_gradients.y *=
            static_cast<float>((word >> (bit + 1)) & 1U) * dropout_scale;
        branch_gradients.z *=
            static_cast<float>((word >> (bit + 2)) & 1U) * dropout_scale;
        branch_gradients.w *=
            static_cast<float>((word >> (bit + 3)) & 1U) * dropout_scale;
    }
    store_branch4(
        grad_branch_row + column,
        branch_gradients);
}


template <int Hidden, int BlockRows, int Threads>
__global__ __launch_bounds__(Threads)
void residual_layer_norm_backward_partials_kernel(
    const float* __restrict__ grad_normalized,
    const float* __restrict__ updated,
    const float* __restrict__ mean,
    const float* __restrict__ rstd,
    float* __restrict__ partial_weight,
    float* __restrict__ partial_bias,
    int64_t row_count) {
    const int column = blockIdx.x * Threads + threadIdx.x;
    const int64_t first_row =
        static_cast<int64_t>(blockIdx.y) * BlockRows;
    float weight_sum = 0.0f;
    float bias_sum = 0.0f;

    if (column < Hidden) {
#pragma unroll
        for (int row_delta = 0; row_delta < BlockRows; ++row_delta) {
            const int64_t row = first_row + row_delta;
            if (row < row_count) {
                const int64_t index = row * Hidden + column;
                const float gradient = grad_normalized[index];
                const float xhat =
                    (updated[index] - mean[row]) * rstd[row];
                weight_sum += gradient * xhat;
                bias_sum += gradient;
            }
        }
        const int64_t partial_index =
            static_cast<int64_t>(blockIdx.y) * Hidden + column;
        partial_weight[partial_index] = weight_sum;
        partial_bias[partial_index] = bias_sum;
    }
}


template <int Hidden, int Threads>
__global__ __launch_bounds__(Threads)
void residual_layer_norm_backward_reduce_kernel(
    const float* __restrict__ partial_weight,
    const float* __restrict__ partial_bias,
    float* __restrict__ grad_weight,
    float* __restrict__ grad_bias,
    int64_t group_count) {
    const int column = blockIdx.x * Threads + threadIdx.x;
    if (column >= Hidden) {
        return;
    }

    float weight_sum = 0.0f;
    float bias_sum = 0.0f;
    for (int64_t group = 0; group < group_count; ++group) {
        const int64_t index = group * Hidden + column;
        weight_sum += partial_weight[index];
        bias_sum += partial_bias[index];
    }
    grad_weight[column] = weight_sum;
    grad_bias[column] = bias_sum;
}


template <int Hidden, int Threads>
__global__ __launch_bounds__(Threads)
void residual_layer_norm_backward_atomic_kernel(
    const float* __restrict__ grad_normalized,
    const float* __restrict__ updated,
    const float* __restrict__ mean,
    const float* __restrict__ rstd,
    float* __restrict__ grad_weight,
    float* __restrict__ grad_bias,
    int64_t row_count) {
    const int column = blockIdx.x * Threads + threadIdx.x;
    if (column >= Hidden) {
        return;
    }

    float weight_sum = 0.0f;
    float bias_sum = 0.0f;
    for (
        int64_t row = static_cast<int64_t>(blockIdx.y);
        row < row_count;
        row += static_cast<int64_t>(gridDim.y)) {
        const int64_t index = row * Hidden + column;
        const float gradient = grad_normalized[index];
        const float xhat = (updated[index] - mean[row]) * rstd[row];
        weight_sum += gradient * xhat;
        bias_sum += gradient;
    }
    atomicAdd(grad_weight + column, weight_sum);
    atomicAdd(grad_bias + column, bias_sum);
}


template <int Hidden, int Waves>
__global__ __launch_bounds__(kWaveSize * Waves)
void residual_layer_norm_backward_grouped_atomic_kernel(
    const float* __restrict__ grad_normalized,
    const float* __restrict__ updated,
    const float* __restrict__ mean,
    const float* __restrict__ rstd,
    float* __restrict__ grad_weight,
    float* __restrict__ grad_bias,
    int64_t row_count,
    int64_t parameter_groups) {
    const int lane = threadIdx.x & (kWaveSize - 1);
    const int wave = threadIdx.x / kWaveSize;
    const int column = blockIdx.x * kWaveSize + lane;
    const int64_t group =
        static_cast<int64_t>(blockIdx.y) * Waves + wave;
    float weight_sum = 0.0f;
    float bias_sum = 0.0f;

    if (column < Hidden && group < parameter_groups) {
        for (
            int64_t row = group;
            row < row_count;
            row += parameter_groups) {
            const int64_t index = row * Hidden + column;
            const float gradient = grad_normalized[index];
            const float xhat =
                (updated[index] - mean[row]) * rstd[row];
            weight_sum += gradient * xhat;
            bias_sum += gradient;
        }
    }

    __shared__ float weight_partials[Waves][kWaveSize];
    __shared__ float bias_partials[Waves][kWaveSize];
    weight_partials[wave][lane] = weight_sum;
    bias_partials[wave][lane] = bias_sum;
    __syncthreads();

    if (wave == 0 && column < Hidden) {
#pragma unroll
        for (int source_wave = 1; source_wave < Waves; ++source_wave) {
            weight_sum += weight_partials[source_wave][lane];
            bias_sum += bias_partials[source_wave][lane];
        }
        atomicAdd(grad_weight + column, weight_sum);
        atomicAdd(grad_bias + column, bias_sum);
    }
}


template <int Hidden, int Threads, typename Branch>
void launch_forward_typed(
    const at::Tensor& x,
    const at::Tensor& branch,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::Tensor& updated,
    at::Tensor& normalized,
    at::Tensor& mean,
    at::Tensor& rstd,
    at::Tensor& dropout_mask,
    float dropout_p,
    float eps,
    bool dropout,
    uint64_t seed,
    uint64_t offset,
    hipStream_t stream) {
    const int64_t row_count = x.numel() / Hidden;
    const dim3 grid(static_cast<unsigned int>(row_count));
    const dim3 block(Threads);
    const Branch* branch_ptr =
        reinterpret_cast<const Branch*>(branch.data_ptr());
    auto* dropout_mask_ptr =
        reinterpret_cast<uint32_t*>(dropout_mask.data_ptr<int32_t>());
    const uint32_t dropout_threshold = static_cast<uint32_t>(
        static_cast<double>(dropout_p) * 4294967296.0);
    const float dropout_scale =
        dropout ? 1.0f / (1.0f - dropout_p) : 1.0f;

    if (dropout) {
        hipLaunchKernelGGL(
            (residual_layer_norm_forward_kernel<
                Hidden,
                Threads,
                Branch,
                true>),
            grid,
            block,
            0,
            stream,
            x.data_ptr<float>(),
            branch_ptr,
            weight.data_ptr<float>(),
            bias.data_ptr<float>(),
            updated.data_ptr<float>(),
            normalized.data_ptr<float>(),
            mean.data_ptr<float>(),
            rstd.data_ptr<float>(),
            dropout_mask_ptr,
            dropout_scale,
            dropout_threshold,
            eps,
            seed,
            offset);
    } else {
        hipLaunchKernelGGL(
            (residual_layer_norm_forward_kernel<
                Hidden,
                Threads,
                Branch,
                false>),
            grid,
            block,
            0,
            stream,
            x.data_ptr<float>(),
            branch_ptr,
            weight.data_ptr<float>(),
            bias.data_ptr<float>(),
            updated.data_ptr<float>(),
            normalized.data_ptr<float>(),
            mean.data_ptr<float>(),
            rstd.data_ptr<float>(),
            dropout_mask_ptr,
            dropout_scale,
            dropout_threshold,
            eps,
            seed,
            offset);
    }
}


template <int Hidden, int Threads>
void launch_forward_hidden(
    const at::Tensor& x,
    const at::Tensor& branch,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::Tensor& updated,
    at::Tensor& normalized,
    at::Tensor& mean,
    at::Tensor& rstd,
    at::Tensor& dropout_mask,
    float dropout_p,
    float eps,
    bool dropout,
    uint64_t seed,
    uint64_t offset,
    hipStream_t stream) {
    if (branch.scalar_type() == at::kHalf) {
        launch_forward_typed<Hidden, Threads, __half>(
            x,
            branch,
            weight,
            bias,
            updated,
            normalized,
            mean,
            rstd,
            dropout_mask,
            dropout_p,
            eps,
            dropout,
            seed,
            offset,
            stream);
    } else {
        launch_forward_typed<Hidden, Threads, float>(
            x,
            branch,
            weight,
            bias,
            updated,
            normalized,
            mean,
            rstd,
            dropout_mask,
            dropout_p,
            eps,
            dropout,
            seed,
            offset,
            stream);
    }
}


template <int Hidden, typename Branch>
void launch_forward_vector4_typed(
    const at::Tensor& x,
    const at::Tensor& branch,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::Tensor& updated,
    at::Tensor& normalized,
    at::Tensor& mean,
    at::Tensor& rstd,
    float eps,
    hipStream_t stream) {
    constexpr int Threads = Hidden / 4;
    const int64_t row_count = x.numel() / Hidden;
    const dim3 grid(static_cast<unsigned int>(row_count));
    const dim3 block(Threads);
    hipLaunchKernelGGL(
        (residual_layer_norm_forward_vector4_kernel<Hidden, Branch>),
        grid,
        block,
        0,
        stream,
        x.data_ptr<float>(),
        reinterpret_cast<const Branch*>(branch.data_ptr()),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        updated.data_ptr<float>(),
        normalized.data_ptr<float>(),
        mean.data_ptr<float>(),
        rstd.data_ptr<float>(),
        eps);
}


template <int Hidden>
void launch_forward_vector4(
    const at::Tensor& x,
    const at::Tensor& branch,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::Tensor& updated,
    at::Tensor& normalized,
    at::Tensor& mean,
    at::Tensor& rstd,
    float eps,
    hipStream_t stream) {
    if (branch.scalar_type() == at::kHalf) {
        launch_forward_vector4_typed<Hidden, __half>(
            x,
            branch,
            weight,
            bias,
            updated,
            normalized,
            mean,
            rstd,
            eps,
            stream);
    } else {
        launch_forward_vector4_typed<Hidden, float>(
            x,
            branch,
            weight,
            bias,
            updated,
            normalized,
            mean,
            rstd,
            eps,
            stream);
    }
}


template <int Alignment>
bool tensor_pointer_is_aligned(const at::Tensor& tensor) {
    static_assert((Alignment & (Alignment - 1)) == 0);
    return (
        reinterpret_cast<std::uintptr_t>(tensor.data_ptr())
        & (Alignment - 1)
    ) == 0;
}


bool can_use_forward_vector4(
    const at::Tensor& x,
    const at::Tensor& branch,
    const at::Tensor& weight,
    const at::Tensor& bias) {
    return tensor_pointer_is_aligned<16>(x)
        && tensor_pointer_is_aligned<16>(weight)
        && tensor_pointer_is_aligned<16>(bias)
        && (
            branch.scalar_type() == at::kHalf
                ? tensor_pointer_is_aligned<8>(branch)
                : tensor_pointer_is_aligned<16>(branch));
}


template <int Hidden>
bool forward_vector4_is_profitable(int64_t row_count) {
    if constexpr (Hidden == 128) {
        return row_count >= 2048;
    } else if constexpr (Hidden == 256 || Hidden == 384) {
        return row_count >= 1024;
    } else if constexpr (Hidden == 512 || Hidden == 768) {
        return row_count >= 512;
    } else {
        return row_count >= 256;
    }
}


template <int Hidden, int Threads, typename Branch>
void launch_backward_input_typed(
    const at::Tensor& grad_updated,
    const at::Tensor& grad_normalized,
    const at::Tensor& updated,
    const at::Tensor& weight,
    const at::Tensor& mean,
    const at::Tensor& rstd,
    const at::Tensor& dropout_mask,
    at::Tensor& grad_x,
    at::Tensor& grad_branch,
    float dropout_scale,
    bool dropout,
    hipStream_t stream) {
    const int64_t row_count = updated.numel() / Hidden;
    const dim3 grid(static_cast<unsigned int>(row_count));
    const dim3 block(Threads);
    Branch* grad_branch_ptr =
        reinterpret_cast<Branch*>(grad_branch.data_ptr());

    if (dropout) {
        hipLaunchKernelGGL(
            (residual_layer_norm_backward_input_kernel<
                Hidden,
                Threads,
                Branch,
                true>),
            grid,
            block,
            0,
            stream,
            grad_updated.data_ptr<float>(),
            grad_normalized.data_ptr<float>(),
            updated.data_ptr<float>(),
            weight.data_ptr<float>(),
            mean.data_ptr<float>(),
            rstd.data_ptr<float>(),
            reinterpret_cast<const uint32_t*>(
                dropout_mask.data_ptr<int32_t>()),
            grad_x.data_ptr<float>(),
            grad_branch_ptr,
            dropout_scale);
    } else {
        hipLaunchKernelGGL(
            (residual_layer_norm_backward_input_kernel<
                Hidden,
                Threads,
                Branch,
                false>),
            grid,
            block,
            0,
            stream,
            grad_updated.data_ptr<float>(),
            grad_normalized.data_ptr<float>(),
            updated.data_ptr<float>(),
            weight.data_ptr<float>(),
            mean.data_ptr<float>(),
            rstd.data_ptr<float>(),
            reinterpret_cast<const uint32_t*>(
                dropout_mask.data_ptr<int32_t>()),
            grad_x.data_ptr<float>(),
            grad_branch_ptr,
            dropout_scale);
    }
}


template <int Hidden, int Threads>
void launch_backward_input_hidden(
    const at::Tensor& grad_updated,
    const at::Tensor& grad_normalized,
    const at::Tensor& updated,
    const at::Tensor& weight,
    const at::Tensor& mean,
    const at::Tensor& rstd,
    const at::Tensor& dropout_mask,
    at::Tensor& grad_x,
    at::Tensor& grad_branch,
    float dropout_scale,
    bool dropout,
    hipStream_t stream) {
    if (grad_branch.scalar_type() == at::kHalf) {
        launch_backward_input_typed<Hidden, Threads, __half>(
            grad_updated,
            grad_normalized,
            updated,
            weight,
            mean,
            rstd,
            dropout_mask,
            grad_x,
            grad_branch,
            dropout_scale,
            dropout,
            stream);
    } else {
        launch_backward_input_typed<Hidden, Threads, float>(
            grad_updated,
            grad_normalized,
            updated,
            weight,
            mean,
            rstd,
            dropout_mask,
            grad_x,
            grad_branch,
            dropout_scale,
            dropout,
            stream);
    }
}


template <int Hidden, int BlockRows, int Threads>
void launch_backward_parameters_hidden(
    const at::Tensor& grad_normalized,
    const at::Tensor& updated,
    const at::Tensor& mean,
    const at::Tensor& rstd,
    at::Tensor& partial_weight,
    at::Tensor& partial_bias,
    at::Tensor& grad_weight,
    at::Tensor& grad_bias,
    int64_t row_count,
    int64_t group_count,
    hipStream_t stream) {
    const dim3 partial_grid(
        (Hidden + Threads - 1) / Threads,
        static_cast<unsigned int>(group_count));
    const dim3 partial_block(Threads);
    hipLaunchKernelGGL(
        (residual_layer_norm_backward_partials_kernel<
            Hidden,
            BlockRows,
            Threads>),
        partial_grid,
        partial_block,
        0,
        stream,
        grad_normalized.data_ptr<float>(),
        updated.data_ptr<float>(),
        mean.data_ptr<float>(),
        rstd.data_ptr<float>(),
        partial_weight.data_ptr<float>(),
        partial_bias.data_ptr<float>(),
        row_count);

    constexpr int ReduceThreads = 256;
    const dim3 reduce_grid((Hidden + ReduceThreads - 1) / ReduceThreads);
    const dim3 reduce_block(ReduceThreads);
    hipLaunchKernelGGL(
        (residual_layer_norm_backward_reduce_kernel<
            Hidden,
            ReduceThreads>),
        reduce_grid,
        reduce_block,
        0,
        stream,
        partial_weight.data_ptr<float>(),
        partial_bias.data_ptr<float>(),
        grad_weight.data_ptr<float>(),
        grad_bias.data_ptr<float>(),
        group_count);
}


template <int Hidden, int Threads>
void launch_backward_parameters_atomic(
    const at::Tensor& grad_normalized,
    const at::Tensor& updated,
    const at::Tensor& mean,
    const at::Tensor& rstd,
    at::Tensor& grad_weight,
    at::Tensor& grad_bias,
    int64_t row_count,
    int64_t parameter_groups,
    hipStream_t stream) {
    if constexpr (Threads == 64 || Threads == 128) {
        constexpr int Waves = Threads == 64 ? 4 : 2;
        const dim3 grid(
            (Hidden + kWaveSize - 1) / kWaveSize,
            static_cast<unsigned int>(
                (parameter_groups + Waves - 1) / Waves));
        const dim3 block(kWaveSize * Waves);
        hipLaunchKernelGGL(
            (residual_layer_norm_backward_grouped_atomic_kernel<
                Hidden,
                Waves>),
            grid,
            block,
            0,
            stream,
            grad_normalized.data_ptr<float>(),
            updated.data_ptr<float>(),
            mean.data_ptr<float>(),
            rstd.data_ptr<float>(),
            grad_weight.data_ptr<float>(),
            grad_bias.data_ptr<float>(),
            row_count,
            parameter_groups);
    } else {
        const dim3 grid(
            (Hidden + Threads - 1) / Threads,
            static_cast<unsigned int>(parameter_groups));
        const dim3 block(Threads);
        hipLaunchKernelGGL(
            (residual_layer_norm_backward_atomic_kernel<Hidden, Threads>),
            grid,
            block,
            0,
            stream,
            grad_normalized.data_ptr<float>(),
            updated.data_ptr<float>(),
            mean.data_ptr<float>(),
            rstd.data_ptr<float>(),
            grad_weight.data_ptr<float>(),
            grad_bias.data_ptr<float>(),
            row_count);
    }
}


template <int Hidden>
void dispatch_forward_threads(
    int64_t threads,
    const at::Tensor& x,
    const at::Tensor& branch,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::Tensor& updated,
    at::Tensor& normalized,
    at::Tensor& mean,
    at::Tensor& rstd,
    at::Tensor& dropout_mask,
    float dropout_p,
    float eps,
    bool dropout,
    uint64_t seed,
    uint64_t offset,
    hipStream_t stream) {
    const auto row_count = x.numel() / Hidden;
    if (!dropout
        && forward_vector4_is_profitable<Hidden>(row_count)
        && can_use_forward_vector4(
            x,
            branch,
            weight,
            bias)) {
        launch_forward_vector4<Hidden>(
            x,
            branch,
            weight,
            bias,
            updated,
            normalized,
            mean,
            rstd,
            eps,
            stream);
        return;
    }
    if (threads == 128) {
        launch_forward_hidden<Hidden, 128>(
            x, branch, weight, bias, updated, normalized, mean, rstd,
            dropout_mask,
            dropout_p, eps, dropout, seed, offset, stream);
    } else if (threads == 192) {
        launch_forward_hidden<Hidden, 192>(
            x, branch, weight, bias, updated, normalized, mean, rstd,
            dropout_mask,
            dropout_p, eps, dropout, seed, offset, stream);
    } else if (threads == 256) {
        launch_forward_hidden<Hidden, 256>(
            x, branch, weight, bias, updated, normalized, mean, rstd,
            dropout_mask,
            dropout_p, eps, dropout, seed, offset, stream);
    } else {
        TORCH_CHECK(false, "threads must be 128, 192 or 256");
    }
}


template <int Hidden, typename Branch>
void launch_backward_input_vector4_typed(
    const at::Tensor& grad_updated,
    const at::Tensor& grad_normalized,
    const at::Tensor& updated,
    const at::Tensor& weight,
    const at::Tensor& mean,
    const at::Tensor& rstd,
    const at::Tensor& dropout_mask,
    at::Tensor& grad_x,
    at::Tensor& grad_branch,
    float dropout_scale,
    bool dropout,
    hipStream_t stream) {
    constexpr int Threads = Hidden / 4;
    const int64_t row_count = updated.numel() / Hidden;
    const dim3 grid(static_cast<unsigned int>(row_count));
    const dim3 block(Threads);
    auto* grad_branch_ptr =
        reinterpret_cast<Branch*>(grad_branch.data_ptr());
    if (dropout) {
        hipLaunchKernelGGL(
            (residual_layer_norm_backward_input_vector4_kernel<
                Hidden,
                Branch,
                true>),
            grid,
            block,
            0,
            stream,
            grad_updated.data_ptr<float>(),
            grad_normalized.data_ptr<float>(),
            updated.data_ptr<float>(),
            weight.data_ptr<float>(),
            mean.data_ptr<float>(),
            rstd.data_ptr<float>(),
            reinterpret_cast<const uint32_t*>(
                dropout_mask.data_ptr<int32_t>()),
            grad_x.data_ptr<float>(),
            grad_branch_ptr,
            dropout_scale);
    } else {
        hipLaunchKernelGGL(
            (residual_layer_norm_backward_input_vector4_kernel<
                Hidden,
                Branch,
                false>),
            grid,
            block,
            0,
            stream,
            grad_updated.data_ptr<float>(),
            grad_normalized.data_ptr<float>(),
            updated.data_ptr<float>(),
            weight.data_ptr<float>(),
            mean.data_ptr<float>(),
            rstd.data_ptr<float>(),
            reinterpret_cast<const uint32_t*>(
                dropout_mask.data_ptr<int32_t>()),
            grad_x.data_ptr<float>(),
            grad_branch_ptr,
            dropout_scale);
    }
}


template <int Hidden>
void launch_backward_input_vector4(
    const at::Tensor& grad_updated,
    const at::Tensor& grad_normalized,
    const at::Tensor& updated,
    const at::Tensor& weight,
    const at::Tensor& mean,
    const at::Tensor& rstd,
    const at::Tensor& dropout_mask,
    at::Tensor& grad_x,
    at::Tensor& grad_branch,
    float dropout_scale,
    bool dropout,
    hipStream_t stream) {
    if (grad_branch.scalar_type() == at::kHalf) {
        launch_backward_input_vector4_typed<Hidden, __half>(
            grad_updated,
            grad_normalized,
            updated,
            weight,
            mean,
            rstd,
            dropout_mask,
            grad_x,
            grad_branch,
            dropout_scale,
            dropout,
            stream);
    } else {
        launch_backward_input_vector4_typed<Hidden, float>(
            grad_updated,
            grad_normalized,
            updated,
            weight,
            mean,
            rstd,
            dropout_mask,
            grad_x,
            grad_branch,
            dropout_scale,
            dropout,
            stream);
    }
}


bool can_use_backward_input_vector4(
    const at::Tensor& grad_updated,
    const at::Tensor& grad_normalized,
    const at::Tensor& updated,
    const at::Tensor& weight,
    const at::Tensor& grad_x,
    const at::Tensor& grad_branch) {
    return tensor_pointer_is_aligned<16>(grad_updated)
        && tensor_pointer_is_aligned<16>(grad_normalized)
        && tensor_pointer_is_aligned<16>(updated)
        && tensor_pointer_is_aligned<16>(weight)
        && tensor_pointer_is_aligned<16>(grad_x)
        && (
            grad_branch.scalar_type() == at::kHalf
                ? tensor_pointer_is_aligned<8>(grad_branch)
                : tensor_pointer_is_aligned<16>(grad_branch));
}


template <int Hidden>
bool backward_input_vector4_is_profitable(
    int64_t row_count,
    bool dropout) {
    if constexpr (Hidden == 256 || Hidden == 384) {
        return row_count >= 512;
    } else if constexpr (Hidden == 512 || Hidden == 768) {
        return !dropout || row_count >= 256;
    } else {
        return true;
    }
}


template <int Hidden>
void dispatch_backward_threads(
    int64_t threads,
    const at::Tensor& grad_updated,
    const at::Tensor& grad_normalized,
    const at::Tensor& updated,
    const at::Tensor& weight,
    const at::Tensor& mean,
    const at::Tensor& rstd,
    const at::Tensor& dropout_mask,
    at::Tensor& grad_x,
    at::Tensor& grad_branch,
    float dropout_scale,
    bool dropout,
    hipStream_t stream) {
    if constexpr (Hidden >= 256) {
        const int64_t row_count = updated.numel() / Hidden;
        if (backward_input_vector4_is_profitable<Hidden>(
                row_count,
                dropout)
            && can_use_backward_input_vector4(
                grad_updated,
                grad_normalized,
                updated,
                weight,
                grad_x,
                grad_branch)) {
            launch_backward_input_vector4<Hidden>(
                grad_updated,
                grad_normalized,
                updated,
                weight,
                mean,
                rstd,
                dropout_mask,
                grad_x,
                grad_branch,
                dropout_scale,
                dropout,
                stream);
            return;
        }
    }
    if (threads == 128) {
        launch_backward_input_hidden<Hidden, 128>(
            grad_updated, grad_normalized, updated, weight, mean, rstd,
            dropout_mask, grad_x, grad_branch, dropout_scale, dropout,
            stream);
    } else if (threads == 192) {
        launch_backward_input_hidden<Hidden, 192>(
            grad_updated, grad_normalized, updated, weight, mean, rstd,
            dropout_mask, grad_x, grad_branch, dropout_scale, dropout,
            stream);
    } else if (threads == 256) {
        launch_backward_input_hidden<Hidden, 256>(
            grad_updated, grad_normalized, updated, weight, mean, rstd,
            dropout_mask, grad_x, grad_branch, dropout_scale, dropout,
            stream);
    } else {
        TORCH_CHECK(false, "threads must be 128, 192 or 256");
    }
}


template <int Hidden, int Threads>
void dispatch_backward_parameter_rows(
    int64_t block_rows,
    const at::Tensor& grad_normalized,
    const at::Tensor& updated,
    const at::Tensor& mean,
    const at::Tensor& rstd,
    at::Tensor& partial_weight,
    at::Tensor& partial_bias,
    at::Tensor& grad_weight,
    at::Tensor& grad_bias,
    int64_t row_count,
    int64_t group_count,
    hipStream_t stream) {
    if (block_rows == 16) {
        launch_backward_parameters_hidden<Hidden, 16, Threads>(
            grad_normalized, updated, mean, rstd, partial_weight, partial_bias,
            grad_weight, grad_bias, row_count, group_count, stream);
    } else if (block_rows == 32) {
        launch_backward_parameters_hidden<Hidden, 32, Threads>(
            grad_normalized, updated, mean, rstd, partial_weight, partial_bias,
            grad_weight, grad_bias, row_count, group_count, stream);
    } else if (block_rows == 64) {
        launch_backward_parameters_hidden<Hidden, 64, Threads>(
            grad_normalized, updated, mean, rstd, partial_weight, partial_bias,
            grad_weight, grad_bias, row_count, group_count, stream);
    } else if (block_rows == 128) {
        launch_backward_parameters_hidden<Hidden, 128, Threads>(
            grad_normalized, updated, mean, rstd, partial_weight, partial_bias,
            grad_weight, grad_bias, row_count, group_count, stream);
    } else {
        TORCH_CHECK(false, "block_rows must be 16, 32, 64 or 128");
    }
}


template <int Hidden>
void dispatch_backward_parameters(
    int64_t block_rows,
    int64_t parameter_threads,
    const at::Tensor& grad_normalized,
    const at::Tensor& updated,
    const at::Tensor& mean,
    const at::Tensor& rstd,
    at::Tensor& partial_weight,
    at::Tensor& partial_bias,
    at::Tensor& grad_weight,
    at::Tensor& grad_bias,
    int64_t row_count,
    int64_t group_count,
    hipStream_t stream) {
    if (parameter_threads == 64) {
        dispatch_backward_parameter_rows<Hidden, 64>(
            block_rows, grad_normalized, updated, mean, rstd, partial_weight,
            partial_bias, grad_weight, grad_bias, row_count, group_count,
            stream);
    } else if (parameter_threads == 128) {
        dispatch_backward_parameter_rows<Hidden, 128>(
            block_rows, grad_normalized, updated, mean, rstd, partial_weight,
            partial_bias, grad_weight, grad_bias, row_count, group_count,
            stream);
    } else if (parameter_threads == 256) {
        dispatch_backward_parameter_rows<Hidden, 256>(
            block_rows, grad_normalized, updated, mean, rstd, partial_weight,
            partial_bias, grad_weight, grad_bias, row_count, group_count,
            stream);
    } else {
        TORCH_CHECK(false, "parameter_threads must be 64, 128 or 256");
    }
}


template <int Hidden>
void dispatch_backward_parameters_atomic(
    int64_t parameter_threads,
    const at::Tensor& grad_normalized,
    const at::Tensor& updated,
    const at::Tensor& mean,
    const at::Tensor& rstd,
    at::Tensor& grad_weight,
    at::Tensor& grad_bias,
    int64_t row_count,
    int64_t parameter_groups,
    hipStream_t stream) {
    if (parameter_threads == 64) {
        launch_backward_parameters_atomic<Hidden, 64>(
            grad_normalized, updated, mean, rstd, grad_weight, grad_bias,
            row_count, parameter_groups, stream);
    } else if (parameter_threads == 128) {
        launch_backward_parameters_atomic<Hidden, 128>(
            grad_normalized, updated, mean, rstd, grad_weight, grad_bias,
            row_count, parameter_groups, stream);
    } else if (parameter_threads == 256) {
        launch_backward_parameters_atomic<Hidden, 256>(
            grad_normalized, updated, mean, rstd, grad_weight, grad_bias,
            row_count, parameter_groups, stream);
    } else {
        TORCH_CHECK(false, "parameter_threads must be 64, 128 or 256");
    }
}


}  // namespace


std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
residual_layer_norm_forward_hip(
    const at::Tensor& x,
    const at::Tensor& branch,
    const at::Tensor& weight,
    const at::Tensor& bias,
    double dropout_p,
    double eps,
    bool training,
    int64_t threads) {
    check_forward_inputs(
        x,
        branch,
        weight,
        bias,
        dropout_p,
        eps,
        threads);
    c10::cuda::CUDAGuard guard(x.device());
    const int64_t hidden = x.size(-1);
    const int64_t row_count = x.numel() / hidden;
    auto updated = at::empty_like(x);
    auto normalized = at::empty_like(x);
    auto mean = at::empty({row_count}, x.options());
    auto rstd = at::empty({row_count}, x.options());
    const bool dropout = training && dropout_p > 0.0;
    auto dropout_mask = at::empty(
        {dropout ? row_count * (hidden / kWaveSize) : 0},
        x.options().dtype(at::kInt));
    uint64_t seed = 0;
    uint64_t offset = 0;

    if (dropout) {
        TORCH_CHECK(
            c10::hip::currentStreamCaptureStatusMayInitCtx()
                == c10::hip::CaptureStatus::None,
            "native residual LayerNorm dropout does not support HIP graph capture");
        auto* generator =
            at::get_generator_or_default<at::CUDAGeneratorImpl>(
                std::nullopt,
                at::cuda::detail::getDefaultCUDAGenerator(x.device().index()));
        at::PhiloxCudaState state;
        const uint64_t randoms_per_thread =
            (hidden + threads - 1) / threads;
        const uint64_t increment =
            ((randoms_per_thread + 3) / 4) * 4;
        {
            std::lock_guard<std::mutex> lock(generator->mutex_);
            state = generator->philox_cuda_state(increment);
        }
        TORCH_CHECK(
            !state.captured_,
            "native residual LayerNorm dropout does not support HIP graph capture");
        seed = state.seed_.val;
        offset = state.offset_.val;
    }

    const auto stream =
        c10::hip::getCurrentHIPStream(x.device().index()).stream();

    switch (hidden) {
        case 128:
            dispatch_forward_threads<128>(
                threads, x, branch, weight, bias, updated, normalized, mean,
                rstd, dropout_mask, static_cast<float>(dropout_p),
                static_cast<float>(eps), dropout, seed, offset, stream);
            break;
        case 256:
            dispatch_forward_threads<256>(
                threads, x, branch, weight, bias, updated, normalized, mean,
                rstd, dropout_mask, static_cast<float>(dropout_p),
                static_cast<float>(eps), dropout, seed, offset, stream);
            break;
        case 384:
            dispatch_forward_threads<384>(
                threads, x, branch, weight, bias, updated, normalized, mean,
                rstd, dropout_mask, static_cast<float>(dropout_p),
                static_cast<float>(eps), dropout, seed, offset, stream);
            break;
        case 512:
            dispatch_forward_threads<512>(
                threads, x, branch, weight, bias, updated, normalized, mean,
                rstd, dropout_mask, static_cast<float>(dropout_p),
                static_cast<float>(eps), dropout, seed, offset, stream);
            break;
        case 768:
            dispatch_forward_threads<768>(
                threads, x, branch, weight, bias, updated, normalized, mean,
                rstd, dropout_mask, static_cast<float>(dropout_p),
                static_cast<float>(eps), dropout, seed, offset, stream);
            break;
        case 1024:
            dispatch_forward_threads<1024>(
                threads, x, branch, weight, bias, updated, normalized, mean,
                rstd, dropout_mask, static_cast<float>(dropout_p),
                static_cast<float>(eps), dropout, seed, offset, stream);
            break;
        default:
            TORCH_CHECK(false, "unsupported native hidden size: ", hidden);
    }
    C10_HIP_KERNEL_LAUNCH_CHECK();
    return {updated, normalized, mean, rstd, dropout_mask};
}


std::vector<at::Tensor> residual_layer_norm_backward_hip(
    const at::Tensor& grad_updated,
    const at::Tensor& grad_normalized,
    const at::Tensor& updated,
    const at::Tensor& weight,
    const at::Tensor& mean,
    const at::Tensor& rstd,
    const at::Tensor& dropout_mask,
    double dropout_p,
    bool training,
    bool branch_is_half,
    int64_t threads,
    int64_t block_rows,
    int64_t parameter_threads,
    int64_t parameter_groups,
    bool compute_parameter_gradients) {
    check_backward_inputs(
        grad_updated,
        grad_normalized,
        updated,
        weight,
        mean,
        rstd,
        dropout_mask,
        dropout_p,
        training,
        threads,
        block_rows,
        parameter_threads,
        parameter_groups);
    c10::cuda::CUDAGuard guard(updated.device());
    const int64_t hidden = updated.size(-1);
    const int64_t row_count = updated.numel() / hidden;
    const int64_t group_count = (row_count + block_rows - 1) / block_rows;
    const bool deterministic_parameters =
        at::globalContext().deterministicAlgorithms();
    auto grad_x = at::empty_like(updated);
    auto grad_branch = at::empty(
        updated.sizes(),
        updated.options().dtype(branch_is_half ? at::kHalf : at::kFloat));
    auto partials = compute_parameter_gradients && deterministic_parameters
        ? at::empty({2, group_count, hidden}, updated.options())
        : at::Tensor();
    auto partial_weight = compute_parameter_gradients && deterministic_parameters
        ? partials.select(0, 0)
        : at::Tensor();
    auto partial_bias = compute_parameter_gradients && deterministic_parameters
        ? partials.select(0, 1)
        : at::Tensor();
    auto parameter_gradients = compute_parameter_gradients
        ? at::empty({2, hidden}, updated.options())
        : at::Tensor();
    auto grad_weight = compute_parameter_gradients
        ? parameter_gradients.select(0, 0)
        : at::Tensor();
    auto grad_bias = compute_parameter_gradients
        ? parameter_gradients.select(0, 1)
        : at::Tensor();
    const bool dropout = training && dropout_p > 0.0;
    const float dropout_scale =
        dropout
            ? 1.0f / (1.0f - static_cast<float>(dropout_p))
            : 1.0f;
    const auto stream =
        c10::hip::getCurrentHIPStream(updated.device().index()).stream();
    if (compute_parameter_gradients && !deterministic_parameters) {
        C10_HIP_CHECK(hipMemsetAsync(
            parameter_gradients.data_ptr<float>(),
            0,
            parameter_gradients.numel() * sizeof(float),
            stream));
    }

    switch (hidden) {
        case 128:
            dispatch_backward_threads<128>(
                threads, grad_updated, grad_normalized, updated, weight, mean,
                rstd, dropout_mask, grad_x, grad_branch,
                dropout_scale, dropout, stream);
            break;
        case 256:
            dispatch_backward_threads<256>(
                threads, grad_updated, grad_normalized, updated, weight, mean,
                rstd, dropout_mask, grad_x, grad_branch,
                dropout_scale, dropout, stream);
            break;
        case 384:
            dispatch_backward_threads<384>(
                threads, grad_updated, grad_normalized, updated, weight, mean,
                rstd, dropout_mask, grad_x, grad_branch,
                dropout_scale, dropout, stream);
            break;
        case 512:
            dispatch_backward_threads<512>(
                threads, grad_updated, grad_normalized, updated, weight, mean,
                rstd, dropout_mask, grad_x, grad_branch,
                dropout_scale, dropout, stream);
            break;
        case 768:
            dispatch_backward_threads<768>(
                threads, grad_updated, grad_normalized, updated, weight, mean,
                rstd, dropout_mask, grad_x, grad_branch,
                dropout_scale, dropout, stream);
            break;
        case 1024:
            dispatch_backward_threads<1024>(
                threads, grad_updated, grad_normalized, updated, weight, mean,
                rstd, dropout_mask, grad_x, grad_branch,
                dropout_scale, dropout, stream);
            break;
        default:
            TORCH_CHECK(false, "unsupported native hidden size: ", hidden);
    }
    if (compute_parameter_gradients) {
        if (deterministic_parameters) {
            switch (hidden) {
                case 128:
                    dispatch_backward_parameters<128>(
                        block_rows, parameter_threads, grad_normalized, updated,
                        mean, rstd, partial_weight, partial_bias, grad_weight,
                        grad_bias, row_count, group_count, stream);
                    break;
                case 256:
                    dispatch_backward_parameters<256>(
                        block_rows, parameter_threads, grad_normalized, updated,
                        mean, rstd, partial_weight, partial_bias, grad_weight,
                        grad_bias, row_count, group_count, stream);
                    break;
                case 384:
                    dispatch_backward_parameters<384>(
                        block_rows, parameter_threads, grad_normalized, updated,
                        mean, rstd, partial_weight, partial_bias, grad_weight,
                        grad_bias, row_count, group_count, stream);
                    break;
                case 512:
                    dispatch_backward_parameters<512>(
                        block_rows, parameter_threads, grad_normalized, updated,
                        mean, rstd, partial_weight, partial_bias, grad_weight,
                        grad_bias, row_count, group_count, stream);
                    break;
                case 768:
                    dispatch_backward_parameters<768>(
                        block_rows, parameter_threads, grad_normalized, updated,
                        mean, rstd, partial_weight, partial_bias, grad_weight,
                        grad_bias, row_count, group_count, stream);
                    break;
                case 1024:
                    dispatch_backward_parameters<1024>(
                        block_rows, parameter_threads, grad_normalized, updated,
                        mean, rstd, partial_weight, partial_bias, grad_weight,
                        grad_bias, row_count, group_count, stream);
                    break;
                default:
                    TORCH_CHECK(
                        false,
                        "unsupported native hidden size: ",
                        hidden);
            }
        } else {
            switch (hidden) {
                case 128:
                    dispatch_backward_parameters_atomic<128>(
                        parameter_threads, grad_normalized, updated, mean, rstd,
                        grad_weight, grad_bias, row_count, parameter_groups,
                        stream);
                    break;
                case 256:
                    dispatch_backward_parameters_atomic<256>(
                        parameter_threads, grad_normalized, updated, mean, rstd,
                        grad_weight, grad_bias, row_count, parameter_groups,
                        stream);
                    break;
                case 384:
                    dispatch_backward_parameters_atomic<384>(
                        parameter_threads, grad_normalized, updated, mean, rstd,
                        grad_weight, grad_bias, row_count, parameter_groups,
                        stream);
                    break;
                case 512:
                    dispatch_backward_parameters_atomic<512>(
                        parameter_threads, grad_normalized, updated, mean, rstd,
                        grad_weight, grad_bias, row_count, parameter_groups,
                        stream);
                    break;
                case 768:
                    dispatch_backward_parameters_atomic<768>(
                        parameter_threads, grad_normalized, updated, mean, rstd,
                        grad_weight, grad_bias, row_count, parameter_groups,
                        stream);
                    break;
                case 1024:
                    dispatch_backward_parameters_atomic<1024>(
                        parameter_threads, grad_normalized, updated, mean, rstd,
                        grad_weight, grad_bias, row_count, parameter_groups,
                        stream);
                    break;
                default:
                    TORCH_CHECK(
                        false,
                        "unsupported native hidden size: ",
                        hidden);
            }
        }
    }
    C10_HIP_KERNEL_LAUNCH_CHECK();
    return {grad_x, grad_branch, grad_weight, grad_bias};
}
