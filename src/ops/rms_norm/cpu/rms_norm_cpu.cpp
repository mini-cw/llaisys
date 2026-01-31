#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <cstring>

template <typename T>
void rms_norm_(T *out, const T *in, const T *weight, float eps, size_t num_rows, size_t dim) {
    for (size_t i = 0; i < num_rows; ++i) {
        float sum = 0.0;
        for (size_t j = 0; j < dim; ++j) {
            sum += llaisys::utils::cast<float>(in[i * dim + j]) *
                llaisys::utils::cast<float>(in[i * dim + j])
            ;
        }
        float rms = sqrt(eps + sum / dim);
        for (size_t j = 0; j < dim; ++j) {
            out[i * dim + j] = llaisys::utils::cast<T>(
                llaisys::utils::cast<float>(in[i * dim + j]) / rms
                * llaisys::utils::cast<float>(weight[j])
            );
        }
    }
}

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, float eps, llaisysDataType_t type, size_t num_rows, size_t dim) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
                    reinterpret_cast<const float *>(weight), eps, num_rows, dim);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in),
                    reinterpret_cast<const llaisys::bf16_t *>(weight), eps, num_rows, dim);
    case LLAISYS_DTYPE_F16:
        return rms_norm_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in),
                    reinterpret_cast<const llaisys::fp16_t *>(weight), eps, num_rows, dim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
