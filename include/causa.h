#ifndef CAUSA_H
#define CAUSA_H

#include "causa_platform.h"
#include "causa_arena.h"
#include "causa_tensor.h"
#include "causa_graph.h"
#include "causa_backend.h"
#include "causa_memory.h"
#include "causa_scm.h"
#include "causa_io.h"

#ifdef __cplusplus
extern "C" {
#endif

const char* causa_version_string(void);
int causa_init(void);
void causa_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif