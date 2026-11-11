#ifndef CAUSA_ARENA_H
#define CAUSA_ARENA_H

#include "causa_platform.h"
#include <stddef.h>
#include <stdint.h>
#include <stdalign.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CAUSA_ARENA_DEFAULT_SIZE (64 * 1024 * 1024)
#define CAUSA_ARENA_ALIGNMENT 64

// A block of memory.
//
// The data follows the header in the same allocation, so the block is one
// piece of memory rather than a header with a pointer to a second one.
//
// The size of the data is given by the arena when the block is made, and
// the array is declared with one element rather than none, because a
// zero-length array is not standard C and some compilers object to it.
typedef struct causa_arena_block_s {
    struct causa_arena_block_s* next;
    size_t capacity;
    size_t offset;
    uint8_t data[1];
} causa_arena_block_t;

typedef struct causa_arena_s {
    causa_arena_block_t* current;
    causa_arena_block_t* head;
    size_t block_size;
    size_t total_allocated;
    size_t peak_allocated;
    void* (*alloc_fn)(size_t size, void* ctx);
    void (*free_fn)(void* ptr, void* ctx);
    void* alloc_ctx;
} causa_arena_t;

causa_arena_t* causa_arena_new(size_t block_size);
causa_arena_t* causa_arena_new_custom(
    size_t block_size,
    void* (*alloc_fn)(size_t, void*),
    void (*free_fn)(void*, void*),
    void* ctx
);
void causa_arena_free(causa_arena_t* arena);

void* causa_arena_alloc(causa_arena_t* arena, size_t size, size_t alignment);
void* causa_arena_alloc_zero(causa_arena_t* arena, size_t size, size_t alignment);

#define causa_arena_alloc_type(arena, type) \
    (type*)causa_arena_alloc(arena, sizeof(type), CAUSA_ALIGNOF(type))

#define causa_arena_alloc_array(arena, type, count) \
    (type*)causa_arena_alloc(arena, sizeof(type) * (count), CAUSA_ALIGNOF(type))

void causa_arena_reset(causa_arena_t* arena);
size_t causa_arena_used(const causa_arena_t* arena);
size_t causa_arena_capacity(const causa_arena_t* arena);

typedef struct {
    causa_arena_t* arena;
    size_t checkpoint;
} causa_arena_checkpoint_t;

causa_arena_checkpoint_t causa_arena_checkpoint(causa_arena_t* arena);
void causa_arena_rewind(causa_arena_checkpoint_t* cp);

#ifdef __cplusplus
}
#endif

#endif