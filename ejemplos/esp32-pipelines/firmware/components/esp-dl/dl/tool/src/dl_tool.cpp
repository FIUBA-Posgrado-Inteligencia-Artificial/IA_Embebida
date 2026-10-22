#include "dl_tool.hpp"
#include <string.h>
#include "soc/soc_caps.h"

extern "C" {
#if CONFIG_XTENSA_BOOST
void dl_xtensa_bzero_32b(void *ptr, const int n);
#endif

#if CONFIG_PIE_V1_BOOST
void dl_tie728_bzero_128b(void *ptr, const int n);
void dl_tie728_bzero(void *ptr, const int n);
void dl_tie728_memcpy(void *dst, const void *src, const size_t n);
#endif

#if CONFIG_PIE_V2_BOOST
void dl_esp32p4_memcpy(void *dst, const void *src, const size_t n);
#endif
}

namespace dl {
namespace tool {

int round_half_even(float value)
{
#if CONFIG_PIE_V2_BOOST
    int ret;
    __asm__ volatile("fcvt.w.s %0, %1, rne" : "=r"(ret) : "f"(value));
    return ret;
#else
    float rounded;
    if (value < 0) {
        rounded = value - 0.5f;
    } else {
        rounded = value + 0.5f;
    }

    int int_part = (int)rounded;
    if (fabsf(rounded - int_part) < 1e-6) {
        if ((int_part & 1) != 0) {
            if (value < 0)
                int_part++;
            else
                int_part--;
        }
    }
    return int_part;
#endif
}

int round_half_even(double value)
{
    double rounded;
    if (value < 0) {
        rounded = value - 0.5;
    } else {
        rounded = value + 0.5;
    }

    int int_part = (int)rounded;
    if (fabs(rounded - int_part) < 1e-6) {
        if ((int_part & 1) != 0) {
            if (value < 0)
                int_part++;
            else
                int_part--;
        }
    }
    return int_part;
}

int round_half_up(float value)
{
    return (int)floorf(value + 0.5);
}

int round_half_up(double value)
{
    return (int)floor(value + 0.5);
}

template <typename T>
int round(T value)
{
#if CONFIG_ROUND_HALF_EVEN_ENABLED
    return round_half_even(value);
#else
    return round_half_up(value);
#endif
}

template int round(float value);
template int round(double value);

template <typename T>
T shift_and_round_half_even(T value, int shift)
{
    T ret = 0;

    if (shift <= 0) {
        int64_t shifted = 0;
        shifted = static_cast<int64_t>(value) << -shift;
        tool::truncate(ret, shifted);
    } else {
        T shifted = value >> shift;
        T remainder = value & ((static_cast<T>(1) << shift) - 1);
        T half = (static_cast<T>(1)) << (shift - 1);

        if (remainder > half) {
            shifted += 1;
        } else if (remainder == half) {
            if ((shifted & 1) != 0) {
                shifted += 1;
            }
        }
        ret = shifted;
    }
    return ret;
}

template <typename T>
T shift_and_round_half_up(T value, int shift)
{
    int64_t shifted = 0;
    T ret = 0;

    if (shift <= 0) {
        shifted = (static_cast<int64_t>(value)) << -shift;
        tool::truncate(ret, shifted);
    } else {
        int64_t half = (static_cast<int64_t>(1)) << (shift - 1);
        shifted = (value + half) >> shift;
        ret = static_cast<T>(shifted);
    }
    return ret;
}

template <typename T>
T shift_and_round(T value, int shift)
{
#if CONFIG_ROUND_HALF_EVEN_ENABLED
    return shift_and_round_half_even(value, shift);
#else
    return shift_and_round_half_up(value, shift);
#endif
}

template int32_t shift_and_round(int32_t value, int shift);
template int64_t shift_and_round(int64_t value, int shift);

void set_zero(void *ptr, const int n)
{
#if CONFIG_PIE_V1_BOOST
    dl_tie728_bzero(ptr, n);
#else
    bzero(ptr, n);
#endif
}

void copy_memory(void *dst, void *src, const size_t n)
{
#if CONFIG_PIE_V2_BOOST
    dl_esp32p4_memcpy(dst, src, n);
#elif CONFIG_PIE_V1_BOOST
    dl_tie728_memcpy(dst, src, n);
#else
    memcpy(dst, src, n);
#endif
}

memory_addr_type_t memory_addr_type(void *address)
{
#if defined(SOC_MEM_TCM_SUPPORTED)
    if (esp_ptr_in_tcm(address)) {
        return MEMORY_ADDR_TCM;
    }
#elif defined(SOC_MEM_SPM_SUPPORTED)
    if (esp_ptr_in_spm(address)) {
        return MEMORY_ADDR_TCM;
    }
#endif

#if CONFIG_SPIRAM_RODATA || CONFIG_SPIRAM_XIP_FROM_PSRAM
    esp_paddr_t paddr;
    mmu_target_t target;
    ESP_ERROR_CHECK(esp_mmu_vaddr_to_paddr(address, &paddr, &target));
    if (target == MMU_TARGET_PSRAM0) {
        return MEMORY_ADDR_PSRAM;
    } else if (target == MMU_TARGET_FLASH0) {
        return MEMORY_ADDR_FLASH;
    }
#else
    if (esp_ptr_external_ram(address)) {
        return MEMORY_ADDR_PSRAM;
    } else if (esp_ptr_in_drom(address)) {
        return MEMORY_ADDR_FLASH;
    }
#endif
    else if (esp_ptr_internal(address)) {
        return MEMORY_ADDR_INTERNAL;
    } else {
        return MEMORY_ADDR_UKN;
    }
}

// PATCH: local divergence from upstream esp-dl 3.3.9 (see components/README.md).
// esp-dl hardcodes MALLOC_CAP_SPIRAM in several places that have no business requiring
// external RAM (Module::operator new, TensorBase::operator new, the esp32s3 bias repack).
// On a board whose efuses declare PSRAM that does not actually respond, that capability has
// zero bytes and every such allocation returns NULL. `operator new` is not declared noexcept,
// so the compiler does not null-check it: the constructor runs on a NULL `this` and writes the
// vtable pointer to address 0 -> StoreProhibited, EXCVADDR 0x00000000, boot loop.
// Fix: when an allocation that asked for PSRAM fails, retry once from internal RAM before
// giving up. If the retry also fails we return NULL exactly like upstream, so callers that do
// check the result are unaffected.
static bool s_psram_fallback_warned = false;

static void *retry_from_internal(size_t total_size, uint32_t caps, bool zero)
{
    if (!(caps & MALLOC_CAP_SPIRAM)) {
        return nullptr;
    }
    // PATCH-NOTE (deferred): the retry drops MALLOC_CAP_SIMD, which the primary path adds under
    // SOC_SIMD_INSTRUCTION_SUPPORTED. Harmless on IDF 5.3.2 (that branch does not compile there);
    // revisit when moving to IDF >= 5.5.
    uint32_t internal_caps = (caps & ~(uint32_t)MALLOC_CAP_SPIRAM) | MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT;
    void *ret = zero ? heap_caps_aligned_calloc(16, 1, total_size, internal_caps)
                     : heap_caps_aligned_alloc(16, total_size, internal_caps);
    if (ret && !s_psram_fallback_warned) {
        s_psram_fallback_warned = true;
        ESP_LOGW(DL_LOG_TAG,
                 "PATCH: PSRAM unavailable, esp-dl allocations fall back to internal RAM (first hit: %u bytes).",
                 (unsigned)total_size);
    }
    return ret;
}

HEAP_IRAM_ATTR void *malloc_aligned(size_t size, uint32_t caps)
{
#if SOC_SIMD_INSTRUCTION_SUPPORTED
    void *ret = heap_caps_aligned_alloc(16, size, caps | MALLOC_CAP_SIMD);
#else
    void *ret = heap_caps_aligned_alloc(16, size, caps);
#endif
    if (!ret) {
        ret = retry_from_internal(size, caps, false); // PATCH
    }
    if (!ret) {
        int largest_free_block = heap_caps_get_largest_free_block(caps);
        if (largest_free_block < size) {
            ESP_LOGE(DL_LOG_TAG,
                     "Failed to alloc %.2fKB memory, largest available block size %.2fKB. ",
                     size / 1024.f,
                     largest_free_block / 1024.f);
        } else {
            ESP_LOGE(
                DL_LOG_TAG, "Input cap=0x%lx can not callocate with MALLOC_CAP_SIMD, please try other caps.", caps);
        }
    }

    return ret;
}

HEAP_IRAM_ATTR void *calloc_aligned(size_t n, size_t size, uint32_t caps)
{
#if SOC_SIMD_INSTRUCTION_SUPPORTED
    void *ret = heap_caps_aligned_calloc(16, n, size, caps | MALLOC_CAP_SIMD);
#else
    void *ret = heap_caps_aligned_calloc(16, n, size, caps);
#endif
    if (!ret) {
        ret = retry_from_internal(n * size, caps, true); // PATCH
    }
    if (!ret) {
        int largest_free_block = heap_caps_get_largest_free_block(caps);
        if (largest_free_block < size) {
            ESP_LOGE(DL_LOG_TAG,
                     "Failed to alloc %.2fKB memory, largest available block size %.2fKB. ",
                     size / 1024.f,
                     largest_free_block / 1024.f);
        } else {
            ESP_LOGE(
                DL_LOG_TAG, "Input cap=0x%lx can not callocate with MALLOC_CAP_SIMD, please try other caps.", caps);
        }
    }

    return ret;
}

size_t get_aligned_size(size_t size, int alignment)
{
    assert(alignment > 0 && (alignment & (alignment - 1)) == 0 && "alignment must be a power of 2");
    return (size + alignment - 1) & ~(alignment - 1);
}

float *gen_lut_8bit(float *table, int exponent, std::function<float(float)> func)
{
    if (table == nullptr) {
        return table;
    }
    float scale = DL_SCALE(exponent);
    for (int i = 0; i < 256; i++) {
        table[i] = func(scale * (i - 128));
    }
    return table;
}

} // namespace tool
} // namespace dl
