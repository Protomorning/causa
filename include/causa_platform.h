#ifndef CAUSA_PLATFORM_H
#define CAUSA_PLATFORM_H

#include <stddef.h>

#if defined(_MSC_VER)
    #define CAUSA_INLINE __forceinline
    #define CAUSA_ALIGN(n) __declspec(align(n))
    #define CAUSA_ALIGNOF(t) __alignof(t)
    #define CAUSA_THREAD_LOCAL __declspec(thread)
    #define CAUSA_RESTRICT __restrict
    #define CAUSA_NORETURN __declspec(noreturn)
    #define CAUSA_UNREACHABLE() __assume(0)
    #define CAUSA_LIKELY(x) (x)
    #define CAUSA_UNLIKELY(x) (x)

    #include <windows.h>
    #define causa_atomic_inc(ptr) (InterlockedIncrement((volatile LONG*)(ptr)))
    #define causa_atomic_dec(ptr) (InterlockedDecrement((volatile LONG*)(ptr)))
    #define causa_atomic_add(ptr, val) (InterlockedExchangeAdd((volatile LONG*)(ptr), (LONG)(val)) + (val))
    #define causa_atomic_sub(ptr, val) (InterlockedExchangeAdd((volatile LONG*)(ptr), -(LONG)(val)) - (val))

    #include <intrin.h>
    #pragma intrinsic(_ReturnAddress)
    #pragma intrinsic(_AddressOfReturnAddress)

#else
    #define CAUSA_INLINE inline __attribute__((always_inline))
    #define CAUSA_ALIGN(n) __attribute__((aligned(n)))
    #define CAUSA_ALIGNOF(t) __alignof__(t)
    #define CAUSA_THREAD_LOCAL __thread
    #define CAUSA_RESTRICT __restrict__
    #define CAUSA_NORETURN __attribute__((noreturn))
    #define CAUSA_UNREACHABLE() __builtin_unreachable()
    #define CAUSA_LIKELY(x) __builtin_expect(!!(x), 1)
    #define CAUSA_UNLIKELY(x) __builtin_expect(!!(x), 0)

    #define causa_atomic_inc(ptr) __sync_fetch_and_add((ptr), 1)
    #define causa_atomic_dec(ptr) __sync_sub_and_fetch((ptr), 1)
    #define causa_atomic_add(ptr, val) __sync_fetch_and_add((ptr), (val))
    #define causa_atomic_sub(ptr, val) __sync_sub_and_fetch((ptr), (val))
#endif

#if defined(_M_IX86) || defined(_M_X64) || defined(__i386__) || defined(__x86_64__)
    #define CAUSA_ARCH_X86 1
    #define CAUSA_HAS_SSE 1
    #if defined(__AVX2__) || defined(__AVX2__)
        #define CAUSA_HAS_AVX2 1
    #endif
    #if defined(__AVX512F__) || defined(__AVX512F__)
        #define CAUSA_HAS_AVX512 1
    #endif
#elif defined(__aarch64__) || defined(__arm64__)
    #define CAUSA_ARCH_ARM64 1
    #define CAUSA_HAS_NEON 1
#endif

#if defined(_WIN32)
    #define CAUSA_OS_WINDOWS 1
    #define CAUSA_PATH_SEP '\\'
    #define CAUSA_DLLEXPORT __declspec(dllexport)
    #define CAUSA_DLLIMPORT __declspec(dllimport)
#elif defined(__linux__)
    #define CAUSA_OS_LINUX 1
    #define CAUSA_PATH_SEP '/'
    #define CAUSA_DLLEXPORT __attribute__((visibility("default")))
    #define CAUSA_DLLIMPORT
#elif defined(__APPLE__)
    #define CAUSA_OS_MACOS 1
    #define CAUSA_PATH_SEP '/'
    #define CAUSA_DLLEXPORT __attribute__((visibility("default")))
    #define CAUSA_DLLIMPORT
#else
    #define CAUSA_PATH_SEP '/'
    #define CAUSA_DLLEXPORT
    #define CAUSA_DLLIMPORT
#endif

#ifndef CAUSA_API
    #ifdef CAUSA_BUILD_SHARED
        #define CAUSA_API CAUSA_DLLEXPORT
    #elif defined(CAUSA_USE_SHARED)
        #define CAUSA_API CAUSA_DLLIMPORT
    #else
        #define CAUSA_API
    #endif
#endif

// The largest number of variables a model can hold, and the largest number
// of causes one variable can have.
//
// These are here rather than in the causal header because the memory uses
// them as well: a model is built out of what the memory learned.
#define CAUSA_MAX_VARIABLES 64
#define CAUSA_MAX_PARENTS 16

#endif