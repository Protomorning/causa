#define BUILD_IMPLEMENTATION
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#ifdef _WIN32
    #define DLLEXPORT __declspec(dllexport)
    #define SHARED_LIB_EXT ".dll"
    #define STATIC_LIB_EXT ".lib"
    #define EXE_EXT ".exe"
    #define PATH_SEP '\\'
    #define DEFAULT_CC "cl"
#else
    #define DLLEXPORT __attribute__((visibility("default")))
    #define SHARED_LIB_EXT ".so"
    #define STATIC_LIB_EXT ".a"
    #define EXE_EXT ""
    #define PATH_SEP '/'
    #define DEFAULT_CC "cc"
#endif

#define MAX_CMD_LEN 8192
#define MAX_PATH_LEN 1024
#define MAX_SOURCES 64
#define MAX_INCLUDES 16
#define MAX_LIBS 16

typedef enum {
    BUILD_DEBUG = 0,
    BUILD_RELEASE = 1
} build_mode_t;

typedef enum {
    TARGET_STATIC = 0,
    TARGET_SHARED = 1,
    TARGET_EXECUTABLE = 2
} target_type_t;

typedef struct {
    char cc[MAX_PATH_LEN];
    char nvcc[MAX_PATH_LEN];
    char cflags[MAX_CMD_LEN];
    char nvccflags[MAX_CMD_LEN];
    char ldflags[MAX_CMD_LEN];
    char includes[MAX_INCLUDES][MAX_PATH_LEN];
    int num_includes;
    char sources[MAX_SOURCES][MAX_PATH_LEN];
    int num_sources;
    char cuda_sources[MAX_SOURCES][MAX_PATH_LEN];
    int num_cuda_sources;
    char libs[MAX_LIBS][MAX_PATH_LEN];
    int num_libs;
    char output[MAX_PATH_LEN];
    char build_dir[MAX_PATH_LEN];
    build_mode_t mode;
    target_type_t target;
    bool verbose;
    bool clean;
    bool run_after_build;
    bool has_cuda;
} build_config_t;

void config_init(build_config_t* cfg);
void config_add_include(build_config_t* cfg, const char* path);
void config_add_source(build_config_t* cfg, const char* path);
void config_add_lib(build_config_t* cfg, const char* lib);
void config_set_mode(build_config_t* cfg, build_mode_t mode);
void config_set_target(build_config_t* cfg, target_type_t target);
void config_set_output(build_config_t* cfg, const char* output);
void config_set_build_dir(build_config_t* cfg, const char* dir);

bool ensure_dir(const char* path);
void ensure_all_obj_dirs(build_config_t* cfg);
bool file_exists(const char* path);
bool run_command(const char* cmd, bool verbose);
char* get_obj_path(build_config_t* cfg, const char* source);
bool compile_source(build_config_t* cfg, const char* source, const char* obj_path);
bool link_target(build_config_t* cfg);
bool build(build_config_t* cfg);
bool run_executable(build_config_t* cfg);
void print_usage(const char* prog);

void config_init(build_config_t* cfg) {
    memset(cfg, 0, sizeof(build_config_t));

    const char* cc_env = getenv("CC");
    strncpy(cfg->cc, cc_env ? cc_env : DEFAULT_CC, MAX_PATH_LEN - 1);

    const char* nvcc_env = getenv("NVCC");
    strncpy(cfg->nvcc, nvcc_env ? nvcc_env : "nvcc", MAX_PATH_LEN - 1);

    cfg->mode = BUILD_DEBUG;
    cfg->target = TARGET_STATIC;
    cfg->verbose = false;
    cfg->clean = false;
    cfg->run_after_build = false;
    cfg->has_cuda = false;

    strncpy(cfg->build_dir, "build", MAX_PATH_LEN - 1);
    strncpy(cfg->output, "causa", MAX_PATH_LEN - 1);

    // Detect CUDA
    char test_cmd[MAX_CMD_LEN];
#ifdef _WIN32
    snprintf(test_cmd, MAX_CMD_LEN, "where nvcc >nul 2>nul");
#else
    snprintf(test_cmd, MAX_CMD_LEN, "which nvcc >/dev/null 2>&1");
#endif
    if (system(test_cmd) == 0) {
        cfg->has_cuda = true;
    }

#ifdef _WIN32
    // /std:c11 is needed for the flexible array member in the arena block.
    // /W4 is the level that matches what the other compilers are held to;
    // /Wall adds a page of notes about the optimiser that say nothing about
    // the code.
    strncat(cfg->cflags, " /nologo /W4 /std:c11 /WX- /O2 /D_CRT_SECURE_NO_WARNINGS", MAX_CMD_LEN - strlen(cfg->cflags) - 1);
    strncat(cfg->ldflags, " /nologo", MAX_CMD_LEN - strlen(cfg->ldflags) - 1);
    strncat(cfg->nvccflags, " -std=c++17 -O2", MAX_CMD_LEN - strlen(cfg->nvccflags) - 1);
#else
    strncat(cfg->cflags, " -std=c11 -Wall -Wextra -Wpedantic -Werror -O2 -fPIC", MAX_CMD_LEN - strlen(cfg->cflags) - 1);
    strncat(cfg->ldflags, "", MAX_CMD_LEN - strlen(cfg->ldflags) - 1);
    strncat(cfg->nvccflags, " -std=c++17 -O2", MAX_CMD_LEN - strlen(cfg->nvccflags) - 1);
#endif
}

void config_add_include(build_config_t* cfg, const char* path) {
    if (cfg->num_includes >= MAX_INCLUDES) return;
    strncpy(cfg->includes[cfg->num_includes++], path, MAX_PATH_LEN - 1);
}

void config_add_source(build_config_t* cfg, const char* path) {
    if (cfg->num_sources >= MAX_SOURCES) return;
    strncpy(cfg->sources[cfg->num_sources++], path, MAX_PATH_LEN - 1);
}

void config_add_cuda_source(build_config_t* cfg, const char* path) {
    if (cfg->num_cuda_sources >= MAX_SOURCES) return;
    strncpy(cfg->cuda_sources[cfg->num_cuda_sources++], path, MAX_PATH_LEN - 1);
}

void config_add_lib(build_config_t* cfg, const char* lib) {
    if (cfg->num_libs >= MAX_LIBS) return;
    strncpy(cfg->libs[cfg->num_libs++], lib, MAX_PATH_LEN - 1);
}

void config_set_mode(build_config_t* cfg, build_mode_t mode) {
    cfg->mode = mode;
    if (mode == BUILD_DEBUG) {
#ifdef _WIN32
        strncat(cfg->cflags, " /Zi /Od /DDEBUG", MAX_CMD_LEN - strlen(cfg->cflags) - 1);
        strncat(cfg->nvccflags, " -G -DDEBUG", MAX_CMD_LEN - strlen(cfg->nvccflags) - 1);
#else
        strncat(cfg->cflags, " -g -O0 -DDEBUG", MAX_CMD_LEN - strlen(cfg->cflags) - 1);
        strncat(cfg->nvccflags, " -G -DDEBUG", MAX_CMD_LEN - strlen(cfg->nvccflags) - 1);
#endif
    } else {
#ifdef _WIN32
        strncat(cfg->cflags, " /O2 /DNDEBUG", MAX_CMD_LEN - strlen(cfg->cflags) - 1);
        strncat(cfg->nvccflags, " -O2 -DNDEBUG", MAX_CMD_LEN - strlen(cfg->nvccflags) - 1);
#else
        strncat(cfg->cflags, " -O3 -DNDEBUG -flto", MAX_CMD_LEN - strlen(cfg->cflags) - 1);
        strncat(cfg->nvccflags, " -O3 -DNDEBUG", MAX_CMD_LEN - strlen(cfg->nvccflags) - 1);
#endif
    }
}

void config_set_target(build_config_t* cfg, target_type_t target) {
    cfg->target = target;
}

void config_set_output(build_config_t* cfg, const char* output) {
    strncpy(cfg->output, output, MAX_PATH_LEN - 1);
}

void config_set_build_dir(build_config_t* cfg, const char* dir) {
    strncpy(cfg->build_dir, dir, MAX_PATH_LEN - 1);
}

bool ensure_dir(const char* path) {
#ifdef _WIN32
    char cmd[MAX_CMD_LEN];
    char* p = _strdup(path);
    if (!p) return false;
    char* sep = strchr(p, '\\');
    while (sep) {
        *sep = '\0';
        snprintf(cmd, MAX_CMD_LEN, "if not exist \"%s\" mkdir \"%s\"", p, p);
        system(cmd);
        *sep = '\\';
        sep = strchr(sep + 1, '\\');
    }
    snprintf(cmd, MAX_CMD_LEN, "if not exist \"%s\" mkdir \"%s\"", path, path);
    bool ret = system(cmd) == 0;
    free(p);
    return ret;
#else
    char cmd[MAX_CMD_LEN];
    snprintf(cmd, MAX_CMD_LEN, "mkdir -p \"%s\"", path);
    return system(cmd) == 0;
#endif
}

void ensure_all_obj_dirs(build_config_t* cfg) {
    for (int i = 0; i < cfg->num_sources; i++) {
        char* obj = get_obj_path(cfg, cfg->sources[i]);
        char dir[MAX_PATH_LEN];
        strncpy(dir, obj, MAX_PATH_LEN - 1);
        char* last_sep = strrchr(dir, PATH_SEP);
        if (last_sep) {
            *last_sep = '\0';
            if (cfg->verbose) printf("Creating directory: %s\n", dir);
            ensure_dir(dir);
        }
    }
}

bool file_exists(const char* path) {
    FILE* f = fopen(path, "r");
    if (f) { fclose(f); return true; }
    return false;
}

bool run_command(const char* cmd, bool verbose) {
    if (verbose) printf("> %s\n", cmd);
    int ret = system(cmd);
    return ret == 0;
}

char* get_obj_path(build_config_t* cfg, const char* source) {
    static char obj_path[MAX_PATH_LEN];
    static char source_copy[MAX_PATH_LEN];
    strncpy(source_copy, source, MAX_PATH_LEN - 1);
    for (char* p = source_copy; *p; p++) {
        if (*p == '/') *p = PATH_SEP;
    }
    const char* base = strrchr(source_copy, PATH_SEP);
    base = base ? base + 1 : source_copy;
    char* dot = strrchr(base, '.');
    size_t len = dot ? (size_t)(dot - base) : strlen(base);

    char rel_path[MAX_PATH_LEN];
    strncpy(rel_path, source_copy, MAX_PATH_LEN - 1);
    if (dot) rel_path[dot - source_copy] = '\0';

    snprintf(obj_path, MAX_PATH_LEN, "%s%c%s.obj", cfg->build_dir, PATH_SEP, rel_path);
    return obj_path;
}

bool compile_cuda_source(build_config_t* cfg, const char* source, const char* obj_path) {
    char cmd[MAX_CMD_LEN] = {0};

    // The CUDA backend is one file that is compiled two ways. With nvcc it
    // becomes the real backend; with the C compiler it becomes a stub that
    // reports that there is no CUDA. The file decides which by checking
    // __CUDACC__, which only nvcc defines.
    if (cfg->has_cuda) {
        // -x cu says what the file is, so the extension does not have to.
        snprintf(cmd, MAX_CMD_LEN, "%s -x cu -c %s", cfg->nvcc, source);
        for (int i = 0; i < cfg->num_includes; i++) {
            char inc[MAX_PATH_LEN];
            snprintf(inc, MAX_PATH_LEN, " -I\"%s\"", cfg->includes[i]);
            strncat(cmd, inc, MAX_CMD_LEN - strlen(cmd) - 1);
        }
        strncat(cmd, " ", MAX_CMD_LEN - strlen(cmd) - 1);
        strncat(cmd, cfg->nvccflags, MAX_CMD_LEN - strlen(cmd) - 1);
        char out_flag[MAX_PATH_LEN];
        snprintf(out_flag, MAX_PATH_LEN, " -o \"%s\"", obj_path);
        strncat(cmd, out_flag, MAX_CMD_LEN - strlen(cmd) - 1);
        return run_command(cmd, cfg->verbose);
    }

#ifdef _WIN32
    snprintf(cmd, MAX_CMD_LEN, "%s /c %s", cfg->cc, source);
    for (int i = 0; i < cfg->num_includes; i++) {
        char inc[MAX_PATH_LEN];
        snprintf(inc, MAX_PATH_LEN, " /I\"%s\"", cfg->includes[i]);
        strncat(cmd, inc, MAX_CMD_LEN - strlen(cmd) - 1);
    }
    strncat(cmd, " ", MAX_CMD_LEN - strlen(cmd) - 1);
    strncat(cmd, cfg->cflags, MAX_CMD_LEN - strlen(cmd) - 1);
    char out_flag[MAX_PATH_LEN];
    snprintf(out_flag, MAX_PATH_LEN, " /Fo\"%s\"", obj_path);
    strncat(cmd, out_flag, MAX_CMD_LEN - strlen(cmd) - 1);
#else
    snprintf(cmd, MAX_CMD_LEN, "%s -x c -c %s", cfg->cc, source);
    for (int i = 0; i < cfg->num_includes; i++) {
        char inc[MAX_PATH_LEN];
        snprintf(inc, MAX_PATH_LEN, " -I\"%s\"", cfg->includes[i]);
        strncat(cmd, inc, MAX_CMD_LEN - strlen(cmd) - 1);
    }
    strncat(cmd, " ", MAX_CMD_LEN - strlen(cmd) - 1);
    strncat(cmd, cfg->cflags, MAX_CMD_LEN - strlen(cmd) - 1);
    char out_flag[MAX_PATH_LEN];
    snprintf(out_flag, MAX_PATH_LEN, " -o \"%s\"", obj_path);
    strncat(cmd, out_flag, MAX_CMD_LEN - strlen(cmd) - 1);
#endif

    return run_command(cmd, cfg->verbose);
}

bool compile_source(build_config_t* cfg, const char* source, const char* obj_path) {
    // Check if it's a CUDA file
    const char* ext = strrchr(source, '.');
    if (ext && (strcmp(ext, ".cu") == 0 || strcmp(ext, ".cuh") == 0)) {
        return compile_cuda_source(cfg, source, obj_path);
    }

    char cmd[MAX_CMD_LEN] = {0};

#ifdef _WIN32
    snprintf(cmd, MAX_CMD_LEN, "%s /c %s", cfg->cc, source);
    for (int i = 0; i < cfg->num_includes; i++) {
        char inc[MAX_PATH_LEN];
        snprintf(inc, MAX_PATH_LEN, " /I\"%s\"", cfg->includes[i]);
        strncat(cmd, inc, MAX_CMD_LEN - strlen(cmd) - 1);
    }
    strncat(cmd, " ", MAX_CMD_LEN - strlen(cmd) - 1);
    strncat(cmd, cfg->cflags, MAX_CMD_LEN - strlen(cmd) - 1);
    char out_flag[MAX_PATH_LEN];
    snprintf(out_flag, MAX_PATH_LEN, " /Fo\"%s\"", obj_path);
    strncat(cmd, out_flag, MAX_CMD_LEN - strlen(cmd) - 1);
#else
    snprintf(cmd, MAX_CMD_LEN, "%s -c %s", cfg->cc, source);
    for (int i = 0; i < cfg->num_includes; i++) {
        char inc[MAX_PATH_LEN];
        snprintf(inc, MAX_PATH_LEN, " -I\"%s\"", cfg->includes[i]);
        strncat(cmd, inc, MAX_CMD_LEN - strlen(cmd) - 1);
    }
    strncat(cmd, " ", MAX_CMD_LEN - strlen(cmd) - 1);
    strncat(cmd, cfg->cflags, MAX_CMD_LEN - strlen(cmd) - 1);
    char out_flag[MAX_PATH_LEN];
    snprintf(out_flag, MAX_PATH_LEN, " -o \"%s\"", obj_path);
    strncat(cmd, out_flag, MAX_CMD_LEN - strlen(cmd) - 1);
#endif

    return run_command(cmd, cfg->verbose);
}

bool link_target(build_config_t* cfg) {
    char cmd[MAX_CMD_LEN] = {0};
    char output_path[MAX_PATH_LEN];

    // The objects are worked out from the sources, so the paths follow the
    // platform rather than being written down.
    char obj_files[MAX_SOURCES][MAX_PATH_LEN];
    int num_obj_files = 0;

    for (int i = 0; i < cfg->num_sources; i++) {
        snprintf(obj_files[num_obj_files++], MAX_PATH_LEN, "%s", get_obj_path(cfg, cfg->sources[i]));
    }
    for (int i = 0; i < cfg->num_cuda_sources; i++) {
        snprintf(obj_files[num_obj_files++], MAX_PATH_LEN, "%s", get_obj_path(cfg, cfg->cuda_sources[i]));
    }

    if (cfg->target == TARGET_STATIC) {
#ifdef _WIN32
        snprintf(output_path, MAX_PATH_LEN, "%s%c%s%s", cfg->build_dir, PATH_SEP, cfg->output, STATIC_LIB_EXT);
        snprintf(cmd, MAX_CMD_LEN, "lib /nologo /OUT:\"%s\"", output_path);
        for (int i = 0; i < num_obj_files; i++) {
            char obj_arg[MAX_PATH_LEN];
            snprintf(obj_arg, MAX_PATH_LEN, " \"%s\"", obj_files[i]);
            strncat(cmd, obj_arg, MAX_CMD_LEN - strlen(cmd) - 1);
        }
#else
        snprintf(output_path, MAX_PATH_LEN, "%s%c%s%s", cfg->build_dir, PATH_SEP, cfg->output, STATIC_LIB_EXT);
        snprintf(cmd, MAX_CMD_LEN, "ar rcs \"%s\"", output_path);
        for (int i = 0; i < num_obj_files; i++) {
            char obj_arg[MAX_PATH_LEN];
            snprintf(obj_arg, MAX_PATH_LEN, " \"%s\"", obj_files[i]);
            strncat(cmd, obj_arg, MAX_CMD_LEN - strlen(cmd) - 1);
        }
#endif
    } else if (cfg->target == TARGET_SHARED) {
#ifdef _WIN32
        snprintf(output_path, MAX_PATH_LEN, "%s%c%s%s", cfg->build_dir, PATH_SEP, cfg->output, SHARED_LIB_EXT);
        snprintf(cmd, MAX_CMD_LEN, "%s /DLL /OUT:\"%s\"", cfg->cc, output_path);
        for (int i = 0; i < num_obj_files; i++) {
            char obj_arg[MAX_PATH_LEN];
            snprintf(obj_arg, MAX_PATH_LEN, " \"%s\"", obj_files[i]);
            strncat(cmd, obj_arg, MAX_CMD_LEN - strlen(cmd) - 1);
        }
        strncat(cmd, " ", MAX_CMD_LEN - strlen(cmd) - 1);
        strncat(cmd, cfg->ldflags, MAX_CMD_LEN - strlen(cmd) - 1);
        if (cfg->has_cuda) {
            strncat(cmd, " cudart.lib cublas.lib cudnn.lib", MAX_CMD_LEN - strlen(cmd) - 1);
        }
#else
        snprintf(output_path, MAX_PATH_LEN, "%s%c%s%s", cfg->build_dir, PATH_SEP, cfg->output, SHARED_LIB_EXT);
        snprintf(cmd, MAX_CMD_LEN, "%s -shared -o \"%s\"", cfg->cc, output_path);
        for (int i = 0; i < num_obj_files; i++) {
            char obj_arg[MAX_PATH_LEN];
            snprintf(obj_arg, MAX_PATH_LEN, " \"%s\"", obj_files[i]);
            strncat(cmd, obj_arg, MAX_CMD_LEN - strlen(cmd) - 1);
        }
        strncat(cmd, " ", MAX_CMD_LEN - strlen(cmd) - 1);
        strncat(cmd, cfg->ldflags, MAX_CMD_LEN - strlen(cmd) - 1);
        if (cfg->has_cuda) {
            strncat(cmd, " -lcudart -lcublas -lcudnn", MAX_CMD_LEN - strlen(cmd) - 1);
        }
#endif
    } else {
        snprintf(output_path, MAX_PATH_LEN, "%s%c%s%s", cfg->build_dir, PATH_SEP, cfg->output, EXE_EXT);
#ifdef _WIN32
        snprintf(cmd, MAX_CMD_LEN, "%s /OUT:\"%s\"", cfg->cc, output_path);
        for (int i = 0; i < num_obj_files; i++) {
            char obj_arg[MAX_PATH_LEN];
            snprintf(obj_arg, MAX_PATH_LEN, " \"%s\"", obj_files[i]);
            strncat(cmd, obj_arg, MAX_CMD_LEN - strlen(cmd) - 1);
        }
        strncat(cmd, " ", MAX_CMD_LEN - strlen(cmd) - 1);
        strncat(cmd, cfg->ldflags, MAX_CMD_LEN - strlen(cmd) - 1);
        if (cfg->has_cuda) {
            strncat(cmd, " cudart.lib cublas.lib cudnn.lib", MAX_CMD_LEN - strlen(cmd) - 1);
        }
#else
        snprintf(cmd, MAX_CMD_LEN, "%s -o \"%s\"", cfg->cc, output_path);
        for (int i = 0; i < num_obj_files; i++) {
            char obj_arg[MAX_PATH_LEN];
            snprintf(obj_arg, MAX_PATH_LEN, " \"%s\"", obj_files[i]);
            strncat(cmd, obj_arg, MAX_CMD_LEN - strlen(cmd) - 1);
        }
        strncat(cmd, " ", MAX_CMD_LEN - strlen(cmd) - 1);
        strncat(cmd, cfg->ldflags, MAX_CMD_LEN - strlen(cmd) - 1);
        if (cfg->has_cuda) {
            strncat(cmd, " -lcudart -lcublas -lcudnn", MAX_CMD_LEN - strlen(cmd) - 1);
        }
#endif
    }

    for (int i = 0; i < cfg->num_libs; i++) {
        char lib_arg[MAX_PATH_LEN];
#ifdef _WIN32
        snprintf(lib_arg, MAX_PATH_LEN, " %s.lib", cfg->libs[i]);
#else
        snprintf(lib_arg, MAX_PATH_LEN, " -l%s", cfg->libs[i]);
#endif
        strncat(cmd, lib_arg, MAX_CMD_LEN - strlen(cmd) - 1);
    }

    return run_command(cmd, cfg->verbose);
}

bool build(build_config_t* cfg) {
    if (cfg->clean) {
#ifdef _WIN32
        char cmd[MAX_CMD_LEN];
        snprintf(cmd, MAX_CMD_LEN, "rmdir /s /q \"%s\" 2>nul", cfg->build_dir);
        system(cmd);
#else
        char cmd[MAX_CMD_LEN];
        snprintf(cmd, MAX_CMD_LEN, "rm -rf \"%s\"", cfg->build_dir);
        system(cmd);
#endif
    }

    if (!ensure_dir(cfg->build_dir)) {
        fprintf(stderr, "Failed to create build directory: %s\n", cfg->build_dir);
        return false;
    }

    ensure_all_obj_dirs(cfg);

    for (int i = 0; i < cfg->num_sources; i++) {
        if (!file_exists(cfg->sources[i])) {
            fprintf(stderr, "Source file not found: %s\n", cfg->sources[i]);
            return false;
        }

        char* obj = get_obj_path(cfg, cfg->sources[i]);
        if (!compile_source(cfg, cfg->sources[i], obj)) {
            fprintf(stderr, "Failed to compile: %s\n", cfg->sources[i]);
            return false;
        }
    }

    // Compile CUDA sources
    for (int i = 0; i < cfg->num_cuda_sources; i++) {
        if (!file_exists(cfg->cuda_sources[i])) {
            fprintf(stderr, "CUDA source file not found: %s\n", cfg->cuda_sources[i]);
            return false;
        }

        char* obj = get_obj_path(cfg, cfg->cuda_sources[i]);
        if (!compile_source(cfg, cfg->cuda_sources[i], obj)) {
            fprintf(stderr, "Failed to compile CUDA: %s\n", cfg->cuda_sources[i]);
            return false;
        }
    }

    if (!link_target(cfg)) {
        fprintf(stderr, "Failed to link target\n");
        return false;
    }

    printf("Build successful: %s%c%s\n", cfg->build_dir, PATH_SEP, cfg->output);
    return true;
}

bool run_executable(build_config_t* cfg) {
    if (cfg->target != TARGET_EXECUTABLE) return false;

    char path[MAX_PATH_LEN];
    snprintf(path, MAX_PATH_LEN, "%s%c%s%s", cfg->build_dir, PATH_SEP, cfg->output, EXE_EXT);

    if (!file_exists(path)) {
        fprintf(stderr, "Executable not found: %s\n", path);
        return false;
    }

    return run_command(path, cfg->verbose);
}

void print_usage(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  --mode <debug|release>     Build mode (default: debug)\n");
    printf("  --target <static|shared|exe>  Target type (default: static)\n");
    printf("  --output <name>            Output name (default: causa)\n");
    printf("  --build-dir <dir>          Build directory (default: build)\n");
    printf("  --cc <compiler>            C compiler (default: %s)\n", DEFAULT_CC);
    printf("  --nvcc <compiler>          CUDA compiler (default: nvcc)\n");
    printf("  --clean                    Clean build directory before building\n");
    printf("  --verbose                  Verbose output\n");
    printf("  --run                      Run after build (executable target only)\n");
    printf("  --help                     Show this help\n");
}

int main(int argc, char** argv) {
    build_config_t cfg;
    config_init(&cfg);

    config_add_include(&cfg, "include");
    config_add_source(&cfg, "src/causa_arena.c");
    config_add_source(&cfg, "src/causa_tensor.c");
    config_add_source(&cfg, "src/causa_graph.c");
    config_add_source(&cfg, "src/causa_backend_cpu.c");
    config_add_source(&cfg, "src/causa_memory.c");
    config_add_source(&cfg, "src/causa_scm.c");
    config_add_source(&cfg, "src/causa_io.c");

    // The CUDA backend is always built. With nvcc it is the real backend;
    // without it, the same file gives a stub that reports that there is
    // no CUDA.
    config_add_cuda_source(&cfg, "src/causa_backend_cuda.c");

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            if (strcmp(argv[++i], "release") == 0) config_set_mode(&cfg, BUILD_RELEASE);
            else if (strcmp(argv[i], "debug") == 0) config_set_mode(&cfg, BUILD_DEBUG);
            else { fprintf(stderr, "Unknown mode: %s\n", argv[i]); return 1; }
        } else if (strcmp(argv[i], "--target") == 0 && i + 1 < argc) {
            if (strcmp(argv[++i], "static") == 0) config_set_target(&cfg, TARGET_STATIC);
            else if (strcmp(argv[i], "shared") == 0) config_set_target(&cfg, TARGET_SHARED);
            else if (strcmp(argv[i], "exe") == 0) config_set_target(&cfg, TARGET_EXECUTABLE);
            else { fprintf(stderr, "Unknown target: %s\n", argv[i]); return 1; }
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            config_set_output(&cfg, argv[++i]);
        } else if (strcmp(argv[i], "--build-dir") == 0 && i + 1 < argc) {
            config_set_build_dir(&cfg, argv[++i]);
        } else if (strcmp(argv[i], "--cc") == 0 && i + 1 < argc) {
            strncpy(cfg.cc, argv[++i], MAX_PATH_LEN - 1);
        } else if (strcmp(argv[i], "--clean") == 0) {
            cfg.clean = true;
        } else if (strcmp(argv[i], "--verbose") == 0) {
            cfg.verbose = true;
        } else if (strcmp(argv[i], "--run") == 0) {
            cfg.run_after_build = true;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    if (!build(&cfg)) {
        return 1;
    }

    if (cfg.run_after_build) {
        if (!run_executable(&cfg)) {
            return 1;
        }
    }

    return 0;
}