#include "embedding_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <cstring>

template <typename T>
void embedding_(T *out, const int64_t *index, const T *weight, size_t ld, size_t len) {
    size_t bytes_per_row = sizeof(T) * ld;
    for (size_t i = 0; i < len; i++) {
        memcpy(out + i * ld, weight + index[i] * ld, bytes_per_row);
    }
}

namespace llaisys::ops::cpu {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight, llaisysDataType_t type, size_t ld, size_t len) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return embedding_(reinterpret_cast<float *>(out), reinterpret_cast<const int64_t *>(index), reinterpret_cast<const float *>(weight), ld, len);
    case LLAISYS_DTYPE_BF16:
        return embedding_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const int64_t *>(index),
                    reinterpret_cast<const llaisys::bf16_t *>(weight), ld, len);
    case LLAISYS_DTYPE_F16:
        return embedding_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const int64_t *>(index),
                    reinterpret_cast<const llaisys::fp16_t *>(weight), ld, len);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
