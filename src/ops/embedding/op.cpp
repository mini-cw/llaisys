#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/embedding_cpu.hpp"

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);
    ASSERT(index->dtype() == LLAISYS_DTYPE_I64, "Embedding: index dtype must be I64");
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());

    // always support cpu calculation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        if (weight->deviceType() == LLAISYS_DEVICE_CPU) {
            return cpu::embedding(out->data(), index->data(), weight->data(), weight->dtype(), weight->shape()[1], index->numel());
        }
    } else {
        ;
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::embedding(out->data(), index->data(), weight->data(), weight->dtype(), weight->shape()[1], index->numel());
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
