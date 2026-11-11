#ifndef CAUSA_IO_H
#define CAUSA_IO_H

#include "causa_tensor.h"
#include "causa_graph.h"
#include "causa_memory.h"
#include "causa_scm.h"
#include <stdio.h>
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CAUSA_IO_MAGIC 0x52534E54u   // "TNSR"
#define CAUSA_IO_VERSION 3u
#define CAUSA_IO_MAX_STRING 256

// The largest number of items a section may hold. A file that says more is
// refused rather than believed: the count decides how much memory is
// allocated, so a wrong one would be a way to ask for anything.
#define CAUSA_IO_MAX_ITEMS 1048576u

// Section tags in the file
typedef enum {
    CAUSA_IO_SECTION_TENSORS = 1,
    CAUSA_IO_SECTION_GRAPH = 2,
    CAUSA_IO_SECTION_SCM = 3,
    CAUSA_IO_SECTION_MEMORY = 4,
    CAUSA_IO_SECTION_END = 255
} causa_io_section_t;

// Writer handle
typedef struct causa_writer_s causa_writer_t;
typedef struct causa_reader_s causa_reader_t;

// ===== Writer =====

causa_writer_t* causa_writer_open(const char* path);
void causa_writer_close(causa_writer_t* writer);
bool causa_writer_is_ok(const causa_writer_t* writer);
const char* causa_writer_error(const causa_writer_t* writer);

// Write a single tensor (with full metadata)
bool causa_write_tensor(causa_writer_t* writer, const causa_tensor_t* tensor);

// Write multiple tensors as a group (shared section)
bool causa_write_tensor_group(causa_writer_t* writer, causa_tensor_t** tensors, int32_t count);

// Write a graph: op structure + all referenced tensors
bool causa_write_graph(causa_writer_t* writer, const causa_graph_t* graph);

// Write a structural causal model
bool causa_write_scm(causa_writer_t* writer, const causa_scm_t* scm);

// Write hierarchical memory (all levels and slots)
bool causa_write_memory(causa_writer_t* writer, const causa_memory_t* memory);

// ===== Reader =====

causa_reader_t* causa_reader_open(const char* path);
void causa_reader_close(causa_reader_t* reader);
bool causa_reader_is_ok(const causa_reader_t* reader);
const char* causa_reader_error(const causa_reader_t* reader);

// Read a single tensor. Returns NULL on error.
causa_tensor_t* causa_read_tensor(
    causa_reader_t* reader,
    causa_arena_t* arena,
    causa_backend_t* backend
);

// Read a group of tensors
int32_t causa_read_tensor_group(
    causa_reader_t* reader,
    causa_arena_t* arena,
    causa_backend_t* backend,
    causa_tensor_t** out_tensors,
    int32_t max_count
);

// Read a graph. Requires the arena/backend to own all tensors.
causa_graph_t* causa_read_graph(
    causa_reader_t* reader,
    causa_arena_t* arena,
    causa_backend_t* backend
);

// Read a structural causal model
causa_scm_t* causa_read_scm(
    causa_reader_t* reader,
    causa_arena_t* arena,
    causa_backend_t* backend
);

// Read hierarchical memory
causa_memory_t* causa_read_memory(
    causa_reader_t* reader,
    causa_arena_t* arena,
    causa_backend_t* backend
);

// ===== Convenience (whole-file) =====

// Save a tensor to a file
bool causa_save_tensor(const char* path, const causa_tensor_t* tensor);

// Load a tensor from a file
causa_tensor_t* causa_load_tensor(
    const char* path,
    causa_arena_t* arena,
    causa_backend_t* backend
);

// Save a memory structure
bool causa_save_memory(const char* path, const causa_memory_t* memory);

// Load a memory structure
causa_memory_t* causa_load_memory(
    const char* path,
    causa_arena_t* arena,
    causa_backend_t* backend
);

// Save a SCM
bool causa_save_scm(const char* path, const causa_scm_t* scm);

// Load a SCM
causa_scm_t* causa_load_scm(
    const char* path,
    causa_arena_t* arena,
    causa_backend_t* backend
);

#ifdef __cplusplus
}
#endif

#endif