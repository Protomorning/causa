# causa

A portable C library for tensor computation with differentiable memory and
causal reasoning. No external dependencies.

Memory, causality and plasticity are one system. Working memory holds recent
episodes, episodic memory holds counterfactuals, semantic memory holds causal
laws, and consolidation is the process of discovering causal structure.
Interventions `do(X=x)` and counterfactuals `Y_x(u)` are graph operations, so
they can be differentiated rather than applied afterwards.

The procedural level says what to do: the laws say what causes what, and a
procedure says what to do about it.

```python
import causa

mem = causa.Memory(key_dim=3, value_dim=3)
for _ in range(400):
    mem.write([x0, x1, x2], [x1, x2, 0.0], level="episodic")

mem.consolidate_causal()                  # episodes -> causal laws
mem.predict(partial)                      # fill in what was not observed
mem.build_scm(3)                          # laws -> a model
mem.plan_intervention(state, goal)        # what to change to reach a goal
```

`examples/causal_memory.py` and `examples/causal_memory.c` run the whole
thing end to end.

## Layout

```
include/
  causa.h            master header
  causa_platform.h   cross-platform macros (atomics, alignment, SIMD)
  causa_arena.h      bump allocator with checkpoints
  causa_tensor.h     tensor type, ops, backward kernels
  causa_graph.h      computation DAG + autograd
  causa_backend.h    hardware abstraction
  causa_memory.h     hierarchical memory + plasticity
  causa_scm.h        SCM, interventions, counterfactuals, discovery
  causa_io.h         binary serialization
src/                 implementations, one per header
test/test_causa.c    C test suite
test_causa.py        Python binding tests
test_causa_scm.py    Python causal and memory tests
examples/            worked examples of the causal memory system
build.c              self-contained build script
CMakeLists.txt       CMake build
setup.py             Python extension build
```

## Building

With CMake:

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build
```

Options: `CAUSA_BUILD_TESTS` (on), `CAUSA_BUILD_SHARED` (off),
`CAUSA_WITH_CUDA` (off).

Or with the self-contained build script, which needs nothing but a C
compiler:

```
build.exe --mode release --target static --verbose
```

Targets: `static`, `shared`, `exe`. Modes: `debug`, `release`.

On Windows, `build.bat` finds the compiler, builds the library and runs the
C test suite.

The Python extension is built by setuptools:

```
python setup.py build_ext --inplace
python test_causa.py
python test_causa_scm.py
```

## C usage

```c
#include "causa.h"

causa_arena_t*  arena   = causa_arena_new(1024 * 1024);
causa_backend_t* backend = causa_backend_cpu_new();

int64_t shape[] = {2, 2};
causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
opts.requires_grad = true;

causa_tensor_t* a = causa_tensor_new(arena, backend, shape, 2, &opts);
causa_tensor_t* b = causa_tensor_new(arena, backend, shape, 2, &opts);

causa_tensor_t* c = causa_tensor_matmul(a, b);
causa_tensor_matmul_backward(a, b, seed_gradient);
```

## Python usage

```python
import causa

a = causa.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
b = (a * a).sum()
g = b.backward()          # gradient w.r.t. a

causa.save("weights.bin", a)
c = causa.load("weights.bin")
```

## Causal reasoning

```c
causa_scm_t* scm = causa_scm_new(arena, backend, NULL);
causa_scm_add_equation_op(scm, Y, parents, 1, noise_UY, CAUSA_OP_ADD, NULL, 0);
causa_scm_build(scm);

// do(X = 5)
causa_intervention_t* iv = causa_intervention_new(arena, X, value_5);
causa_scm_t* intervened = causa_scm_do(scm, iv);

// counterfactual Y_x(u)
causa_counterfactual_t* cf =
    causa_counterfactual_new(arena, Y, X, value_5, evidence);
causa_counterfactual_set_evidence(cf, observed_per_variable, n_vars);
causa_counterfactual_compute(scm, cf);
```

## Causal memory

```c
// Record an episode: the key holds the observed variables
causa_memory_write(mem, observation, outcome, CAUSA_MEMORY_EPISODIC);

// Consolidate: episodic observations become semantic causal laws
int32_t laws = causa_memory_consolidate_causal(mem, scm, &config);

// Read back what was learned about a variable
causa_tensor_t* found[16];
int32_t n = causa_memory_find_causal_laws(mem, child_variable, found, 16);

// Reason over the memory: predict the variables that were not observed.
// A NaN in the observation marks a variable that was not observed.
causa_memory_predict(mem, observation, prediction);

// Build a model out of what was learned, so it can be intervened on
causa_memory_build_scm(mem, learned_scm, num_variables);

// Group the episodes into procedures, which say what to do where
int32_t procedures = causa_memory_learn_procedures(mem, radius);

// Work out what to do to reach a goal. The goal holds the wanted value of
// the variable to be changed, and NaN for the rest.
causa_memory_plan_intervention(mem, state, goal, plan, &count);
```

## Python

```python
import causa

# Tensors
a = causa.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
b = (a * a).sum()
g = b.backward()          # gradient with respect to a

# Memory that learns causal structure
mem = causa.Memory(key_dim=3, value_dim=3)
for _ in range(200):
    mem.write([x0, x1, x2], [x1, x2, 0.0], level="episodic")
laws = mem.consolidate_causal(alpha=0.05, threshold=0.25)
print(mem.find_causal_laws(1))   # the laws about variable 1

# Reason over the memory: the unobserved variables are filled in
prediction = mem.predict(causa.tensor([x0, float("nan"), float("nan")]))
print(prediction.tolist())

# The learned laws become a model that can be reasoned about
scm = mem.build_scm(3)

# Procedures: what to do where
mem.learn_procedures(radius=0.5)
for trigger, action in mem.find_procedures(causa.tensor([x0, x1, x2])):
    print(trigger.tolist(), "->", action.tolist())

# What to do to reach a goal
plan, count = mem.plan_intervention(
    causa.tensor([1.0, float("nan"), float("nan")]),   # what is known
    causa.tensor([float("nan"), float("nan"), 10.0]))   # what is wanted
print(plan.tolist())

# A structural causal model
scm = causa.SCM()
x0 = scm.variable("x0")
x1 = scm.variable("x1")
u1 = scm.variable("u1", trainable=True)
scm.add_observed_variable(x0)
scm.add_equation(x1, [x0], u1, coefficient=1.5)
scm.build()

# An intervention gives a model of its own
after = scm.do(x0, 5.0)
print([v.tolist()[0] for v in after.evaluate()])   # 5.0, 7.5

# A counterfactual
cf = causa.Counterfactual(query=x1, intervention_var=x0,
                           intervention_val=causa.tensor([5.0]),
                           evidence=obs)
cf.set_evidence([obs_x0, obs_x1])
print(cf.compute(scm).tolist())
```

## Design notes

- **Arena allocation.** No `malloc` in hot paths. Checkpoints and rewind for
  scoped lifetimes.
- **Byte strides.** Tensors store strides in bytes; kernels convert to element
  units at the point of indexing.
- **Opaque types.** Public structs are opaque; internals are private.
- **No magic.** Every kernel is a plain C loop or a documented algorithm.
- **Structure and coefficients are found separately.** The optimisation finds
  which variables cause which; the coefficients are then estimated by
  regression, because the constraints that shape the optimisation pull the
  weights away from the fit.
- **A law is linear.** A law states the coefficient of each parent in the
  child's equation. A constant term has no place in that, so it is carried by
  the noise: the abduction infers it along with everything else.
- **A file is not trusted.** Every count in a serialized file decides how much
  memory is set aside, so each one is checked against a bound before it is
  believed, and the sizes in the file are checked against the shapes read
  alongside them.
- **A view carries its own offset.** The data pointer of a view starts at the
  offset, so every kernel reads from the right place without having to know
  that it is looking at a view.
- **A gradient is made where it is needed.** A tensor that is given a gradient
  for the first time has the buffer made for it. Otherwise the contribution
  would be dropped, and the gradient of a product would be lost.
- **A tape is scoped to its graph.** The Python tape is shared by the module,
  but a backward pass follows the inputs back from the tensor it is called on,
  so one graph never applies the operations of another.

## Tests

40 C tests, 66 Python binding assertions and 53 causal binding assertions, all
passing.

## Not yet implemented

- Single-device only. No distributed collectives.
- Metal and Vulkan backends are declared but not implemented.
- The CUDA backend is built when the CUDA toolkit is present. The same
  source compiles either way: with `nvcc` it is the real backend, without it
  the file gives a stub that reports that there is no CUDA. The Python build
  compiles the file with the host compiler, so it always gets the stub.
- The CUDA backend has not been run on a device, and it is not ready to be:
  its tensors are allocated in the host arena and their addresses are passed
  to the kernels as device pointers, the result of a CUDA call is not checked,
  and nothing waits for a kernel to finish before the result is read. It
  would give wrong answers rather than fail.
- The counterfactual gradient is not implemented. `causa_counterfactual_gradient`
  ignores the variable it is asked to differentiate with, and
  `causa_counterfactual_grad_wrt_intervention` gives nothing back, because the
  intervention value never reaches the graph the backward pass walks.
- `causa_memory_to_graph` builds a graph whose read operations have no
  forward function, so the graph computes nothing. It has no callers.
- The file format is written in the byte order and the word size of the
  machine that wrote it, and nothing in the file says which that was. A file
  does not move between machines of a different byte order.
- A counterfactual that is kept holds its result at the width of a key
  rather than the width of a value. The two are the same by default.
- The spike timing rule and the neurogenesis rule are declared but do
  nothing: the first needs the time a spike happened and the second needs
  room to be made, and neither fits the model as it is.
- The laws are linear, so a relation that is not linear in the parents is
  only described by its average. The episodes cover that gap when a
  prediction is made.
- The intervention plan acts on one variable, so a goal that needs several
  causes at once is only partly reached.
