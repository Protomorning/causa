#include "causa_arena.h"
#include "causa_platform.h"
#include <stdlib.h>
#include <string.h>

static void* default_alloc(size_t size, void* ctx) {
    (void)ctx;
    return malloc(size);
}

static void default_free(void* ptr, void* ctx) {
    (void)ctx;
    free(ptr);
}

static causa_arena_block_t* arena_block_new(causa_arena_t* arena, size_t min_size) {
    size_t block_size = arena->block_size;
    if (min_size > block_size) {
        block_size = (min_size + CAUSA_ARENA_ALIGNMENT - 1) & ~(CAUSA_ARENA_ALIGNMENT - 1);
    }

    // The block's own header comes first, then the data. The data array is
    // declared with one element, so that one byte of it is already counted
    // in the header's size and must not be counted twice.
    size_t total_size = sizeof(causa_arena_block_t) + block_size - 1;
    causa_arena_block_t* block = (causa_arena_block_t*)arena->alloc_fn(total_size, arena->alloc_ctx);
    if (!block) return NULL;

    block->next = NULL;
    block->capacity = block_size;
    block->offset = 0;
    return block;
}

causa_arena_t* causa_arena_new(size_t block_size) {
    return causa_arena_new_custom(block_size, default_alloc, default_free, NULL);
}

causa_arena_t* causa_arena_new_custom(
    size_t block_size,
    void* (*alloc_fn)(size_t, void*),
    void (*free_fn)(void*, void*),
    void* ctx
) {
    if (block_size < 4096) block_size = 4096;
    block_size = (block_size + CAUSA_ARENA_ALIGNMENT - 1) & ~(CAUSA_ARENA_ALIGNMENT - 1);

    causa_arena_t* arena = (causa_arena_t*)alloc_fn(sizeof(causa_arena_t), ctx);
    if (!arena) return NULL;

    arena->block_size = block_size;
    arena->alloc_fn = alloc_fn;
    arena->free_fn = free_fn;
    arena->alloc_ctx = ctx;
    arena->total_allocated = 0;
    arena->peak_allocated = 0;
    arena->head = NULL;
    arena->current = NULL;

    arena->current = arena_block_new(arena, 0);
    if (!arena->current) {
        arena->free_fn(arena, ctx);
        return NULL;
    }
    arena->head = arena->current;

    return arena;
}

void causa_arena_free(causa_arena_t* arena) {
    if (!arena) return;

    causa_arena_block_t* block = arena->head;
    while (block) {
        causa_arena_block_t* next = block->next;
        arena->free_fn(block, arena->alloc_ctx);
        block = next;
    }
    arena->free_fn(arena, arena->alloc_ctx);
}

void* causa_arena_alloc(causa_arena_t* arena, size_t size, size_t alignment) {
    if (!arena || size == 0) return NULL;

    alignment = (alignment + CAUSA_ARENA_ALIGNMENT - 1) & ~(CAUSA_ARENA_ALIGNMENT - 1);
    if (alignment < CAUSA_ARENA_ALIGNMENT) alignment = CAUSA_ARENA_ALIGNMENT;

    causa_arena_block_t* block = arena->current;
    uintptr_t addr = (uintptr_t)(block->data + block->offset);
    uintptr_t aligned = (addr + alignment - 1) & ~(uintptr_t)(alignment - 1);
    size_t padding = aligned - addr;

    if (block->offset + padding + size > block->capacity) {
        causa_arena_block_t* new_block = arena_block_new(arena, padding + size);
        if (!new_block) return NULL;

        new_block->next = block->next;
        block->next = new_block;
        block = new_block;
        arena->current = block;

        addr = (uintptr_t)block->data;
        aligned = (addr + alignment - 1) & ~(uintptr_t)(alignment - 1);
        padding = aligned - addr;
    }

    block->offset += padding + size;
    arena->total_allocated += padding + size;
    if (arena->total_allocated > arena->peak_allocated) {
        arena->peak_allocated = arena->total_allocated;
    }

    return (void*)aligned;
}

void* causa_arena_alloc_zero(causa_arena_t* arena, size_t size, size_t alignment) {
    void* ptr = causa_arena_alloc(arena, size, alignment);
    if (ptr) memset(ptr, 0, size);
    return ptr;
}

void causa_arena_reset(causa_arena_t* arena) {
    if (!arena) return;

    causa_arena_block_t* block = arena->head;
    while (block) {
        block->offset = 0;
        block = block->next;
    }
    arena->current = arena->head;
    arena->total_allocated = 0;
}

size_t causa_arena_used(const causa_arena_t* arena) {
    return arena ? arena->total_allocated : 0;
}

size_t causa_arena_capacity(const causa_arena_t* arena) {
    if (!arena) return 0;
    size_t cap = 0;
    causa_arena_block_t* block = arena->head;
    while (block) {
        cap += block->capacity;
        block = block->next;
    }
    return cap;
}

causa_arena_checkpoint_t causa_arena_checkpoint(causa_arena_t* arena) {
    causa_arena_checkpoint_t cp = {0};
    if (!arena) return cp;

    cp.arena = arena;
    cp.checkpoint = arena->total_allocated;
    return cp;
}

void causa_arena_rewind(causa_arena_checkpoint_t* cp) {
    if (!cp || !cp->arena) return;

    causa_arena_t* arena = cp->arena;
    size_t target = cp->checkpoint;

    if (target >= arena->total_allocated) return;

    causa_arena_block_t* block = arena->head;
    size_t accumulated = 0;

    while (block) {
        size_t block_end = accumulated + block->offset;
        if (block_end > target) {
            block->offset = target - accumulated;
            arena->current = block;
            arena->total_allocated = target;
            return;
        }
        accumulated = block_end;
        block = block->next;
    }

    arena->current = arena->head;
    arena->total_allocated = 0;
}