#include "argmax_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void argmax_(T *max_idx, T *max_val, const T *vals, size_t numel) {
    if (numel == 0) return;

    float current_max_v;
    size_t current_max_i = 0;

    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
        current_max_v = llaisys::utils::cast<float>(vals[0]);
    } else {
        current_max_v = static_cast<float>(vals[0]);
    }

    for (size_t i = 1; i < numel; i++) {
        float v_i;
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            v_i = llaisys::utils::cast<float>(vals[i]);
        } else {
            v_i = static_cast<float>(vals[i]);
        }

        if (v_i > current_max_v) {
            current_max_v = v_i;
            current_max_i = i;
        }
    }

    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
        *max_val = llaisys::utils::cast<T>(current_max_v);
        *max_idx = llaisys::utils::cast<T>(static_cast<float>(current_max_i));
    } else {
        *max_val = static_cast<T>(current_max_v);
        *max_idx = static_cast<T>(current_max_i);
    }
}

namespace llaisys::ops::cpu {
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t type, size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return argmax_(reinterpret_cast<float *>(max_idx), reinterpret_cast<float *>(max_val), reinterpret_cast<const float *>(vals), numel);
    case LLAISYS_DTYPE_BF16:
        return argmax_(reinterpret_cast<llaisys::bf16_t *>(max_idx), reinterpret_cast<llaisys::bf16_t *>(max_val),
                    reinterpret_cast<const llaisys::bf16_t *>(vals), numel);
    case LLAISYS_DTYPE_F16:
        return argmax_(reinterpret_cast<llaisys::fp16_t *>(max_idx), reinterpret_cast<llaisys::fp16_t *>(max_val),
                    reinterpret_cast<const llaisys::fp16_t *>(vals), numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
