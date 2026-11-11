# Causal design notes

Technical notes on the `causa_scm` module. For API reference see
`include/causa_scm.h`.

## Model

A structural causal model is a directed acyclic graph of variables, where each
variable `X` has:

- a set of causal parents `Pa(X)`
- an exogenous noise term `U_X`
- a structural equation `X := f(Pa(X), U_X)`

The SCM is stored in `causa_scm_t`, which holds a variable table, the exogenous
noise tensors, and a topological order over the variables.

## Interventions

`causa_scm_do(scm, intervention)` produces a new SCM in which the intervened
variable's structural equation is replaced by a constant. This is a graph
transformation, not an assignment of a value to an existing variable: the
variable's parents are cut, so any effect that flowed through them no longer
does.

```c
causa_intervention_t* iv = causa_intervention_new(arena, X, value);
causa_scm_t* intervened = causa_scm_do(scm, iv);
```

`causa_scm_do_multi` applies several interventions at once.

### Implementation

For a model whose equations are graph operations, the intervention is real
graph surgery. The SCM's graph is copied, and in the copy the operation that
produced the intervened variable is replaced by a copy of the constant. Every
consumer of that variable then reads the intervened value, so the change
propagates through the whole downstream subgraph.

If the intervened variable is produced by no operation at all, which is the
case for an observed exogenous variable, a copy operation is added instead of
one being replaced. The topological sort places it ahead of the consumers.

Values are carried across in the copy, so constants such as a structural
coefficient keep their meaning.

## Counterfactuals

A counterfactual query `Y_x(u)` asks what `Y` would have been had `X` been `x`,
given the noise actually observed. The computation has three steps, matching
Pearl's formulation:

1. **Abduction.** Infer `U` from the evidence, by minimising
   `|| SCM(U) - evidence ||^2` over the noise variables.
2. **Action.** Build the intervened SCM via `do(X = x)`.
3. **Prediction.** Run the intervened SCM with the inferred `U`.

`causa_counterfactual_compute` runs all three steps.
`causa_counterfactual_compute_with_grad` additionally builds a computation graph
by unrolling the abduction optimisation, so gradients can be taken through the
whole procedure. `causa_counterfactual_grad_wrt_intervention` returns
`d Y_x(u) / d x`.

### The abduction step

The abduction needs the model to be differentiable, because the noise is found
by minimising the reconstruction error.

`causa_scm_add_equation_op` adds a structural equation as an operation in
the SCM's graph, which makes the model differentiable: gradients flow from the
variables back to the exogenous noise.

`causa_scm_add_reconstruction_loss` then adds the squared reconstruction
error to that same graph. The abduction loop runs the graph forwards and
backwards and takes a step on the noise.

Three details matter for the loop to work:

- The gradient buffer of the loss is reset each iteration, because zeroing the
  graph clears the contents of every gradient buffer.
- The learning rate has to stay below `2 / lambda_max` of the loss Hessian.
  For a chain of length two the largest eigenvalue is about 21, so a rate of
  0.01 is comfortably inside the stable range.
- A variable with no equation of its own is pinned to its observed value
  before each forward pass, because the observation is the only information
  there is about it.

The reconstruction error falls along a curved valley, so the curvature along
the valley is much smaller than across it. Gradient descent is limited by the
larger curvature, which is why the iteration count has to be generous: the
error falls by a small factor per step, and a couple of thousand iterations
are needed to pin the noise down. A quasi-Newton method would need far fewer,
but it requires a second-order pass that the autodiff does not provide.

The observed value of every variable is supplied with
`causa_counterfactual_set_evidence`. The values are indexed by the SCM's
variable order, so a variable with no equation of its own is registered with
`causa_scm_add_observed_variable` to keep the two in step.

### Holding the noise fixed

`causa_scm_sample` draws fresh noise and then evaluates the model.
`causa_scm_evaluate` only evaluates, using the noise as it stands. A
counterfactual needs the second, because the abduction has already fixed the
noise and resampling would discard it.

### The intervened graph

The abduction adds the reconstruction loss to the model's own graph. Those
operations are not part of the model, so they are not copied into the
intervened graph: their inputs are the observed values, which are not model
tensors. The SCM keeps the number of model operations for exactly this
reason.

## Causal discovery

`causa_discover` implements a NOTEARS-style optimizer over an adjacency
matrix `W`:

```
minimize   L(W) + alpha * ||W||_1 + (rho/2) * h(W)^2 + beta * h(W)
where       h(W) = tr(exp(W . W)) - d
```

The `h(W)` term is the acyclicity constraint: it equals zero exactly when the
graph induced by `W` is acyclic. The augmented Lagrangian parameter `beta` is
updated each iteration as `beta += rho * h(W)`.

`L(W)` is the least-squares loss `|| X - XW ||^2 / (2N)`, computed through the
library's own autodiff so the optimizer uses the same backward kernels as the
rest of the stack.

### How the exponential is computed

`tr(exp(A))` is not available as a primitive, so the exponential is assembled
from graph operations:

1. `A = W . W` — elementwise square, a single multiply op.
2. `B = A / 2^8` — scaling for numerical stability.
3. `E = I + B + B^2/2! + ... + B^32/32!` — the Taylor series, each term a
   matmul and a scaled multiply.
4. `E = E @ E`, repeated eight times — scaling and squaring.

The trace is then the sum of the diagonal, and the whole expression is a graph,
so its gradient comes out of the same reverse-mode autodiff as everything else.
No analytic gradient formula for `tr(exp(A))` is needed, which avoids the
subtleties that come with non-symmetric `A`.

The diagonal is extracted with a dedicated `DIAG` op whose backward scatters
the gradient back onto the diagonal.

### Optimizer

Adam, with the learning rate, the sparsity weight `alpha`, and the iteration
count all configurable. `W` is updated in place, so the graph is built once and
reused across iterations.

The diagonal of `W` is held at zero after every step. A variable cannot be its
own cause, and leaving the diagonal free would let the fit explain a variable
by itself, which spends the acyclicity constraint on a self-loop that cannot
exist.

### The data the optimization runs on

The observations are centred, and then divided by one common factor: the mean of
the per-variable standard deviations.

Centring takes away the constant terms. Without it the fit is pulled away from
the true coefficients by an intercept that the model does not have: on data
where `x1 = 1.5 * x0 + 0.5`, the uncentred regression returns a coefficient of
about 1.26.

The common factor keeps the weights in a range the acyclicity constraint can
handle. The constraint compares the weights with each other, and `tr(exp(A))`
grows without bound as the entries of `A` grow, so data on a large scale would
make the constraint dominate the fit. One factor is used rather than one per
variable, because the relation between the variables is what tells the structure
apart. Standardising each variable separately would make the weights of every
pair nearly equal, and the optimizer would settle on a symmetric matrix that
says nothing about the structure.

### The coefficients are estimated separately

The optimization finds the structure, but the weights it leaves behind are not
the coefficients of the structural equations. The acyclicity constraint and the
sparsity penalty both pull them away from the fit.

So the coefficients are estimated again, by regressing each child on the
parents that were selected. The regression has an intercept, which is not part
of a structural equation — the intercept belongs to the noise — and a small
ridge term, which keeps the fit stable when the parents are close to collinear.
That happens whenever one parent is a descendant of another, which is the usual
case in a chain.

The regression is done on standardised data and the coefficients are put back
on the original scale afterwards. Doing the arithmetic on the raw data loses
the precision that decides whether a parent has an effect of its own when the
variables are large.

A parent whose effect turns out to be negligible is dropped and the fit is
repeated without it, so the coefficients that remain are not distorted by a
parent that does nothing.

### Orientation convention

The model is `X ~= X W`, so `(X W)[n][child] = sum_p X[n][parent] * W[parent][child]`.
A nonzero `W[parent][child]` therefore means `parent -> child`. The discovery
result stores `adjacency[parent][child]` and the parent list for each child
under that convention.

The sign of the weight is kept separately from its magnitude. The magnitude is
what the edge threshold is applied to, and it is compared across variables on
the scale of the standard deviations, so the threshold means the same thing for
every variable. The signed weight is the coefficient that ends up in the law: a
negative effect is as real a causal law as a positive one.

### Interventional data

When interventional samples are supplied, edges that the observational data
leaves ambiguous are oriented by comparing the mean of each variable under
intervention against its observational mean. A variable that responds to
`do(X)` has `X` as an ancestor.

### Verification

The constraint gradient is checked against central finite differences. Because
the forward pass runs a 32-term series and eight squarings in float32, the
computed `h` carries an absolute error of order `1e-4`, so the gradient is
compared as a vector using the relative L2 norm. The observed agreement is
better than 0.2%.

The discovery test generates data from a known chain `X0 -> X1 -> X2` and
checks that the recovered parent sets match.

## Memory integration

The laws that memory holds are put to use here: they become a model, and the
model can be run.

### Episodes as observations

An episodic slot holds one observation: the key carries the values of the
observed variables, and the value carries the outcome.
`causa_memory_collect` gathers the keys of every occupied slot into a
matrix, which is the layout the discovery routine consumes.

### Consolidation

`causa_memory_consolidate_causal` runs the whole bridge:

1. Gather the episodic observations into a matrix.
2. Run causal discovery over them.
3. Write the resulting parent sets and coefficients into semantic memory.
4. Prune the edges of a supplied SCM that the data does not support.

### Causal laws in semantic memory

A law is stored as a key and a value. The key is a role vector over the
variables: 2 marks the child the law describes, 1 marks each of its parents,
and 0 marks the rest. The value holds the coefficient of every variable in the
child's equation.

The encoding is self-describing, so a law can be found without a separate
index. `causa_memory_find_causal_laws` scans semantic memory for the laws
recorded for a given child, and `causa_memory_count_causal_laws` counts
them.

`causa_memory_causal_consolidate` writes the laws from an already computed
discovery result, which is useful when the structure was learned elsewhere.

### Counterfactuals in memory

`causa_memory_store_counterfactual` places a counterfactual result into
episodic memory, where it can later be retrieved by
`causa_memory_retrieve_counterfactuals`. This is how "what would have
happened if I had done the other thing" becomes part of the system's
experience rather than something recomputed each time.

## Reasoning over memory

Storing what was learned is only half of it. The memory has to be able to use
it, and there are two ways it does.

### Predicting what was not observed

`causa_memory_predict` takes a partial observation and fills in the rest. A
NaN in the observation marks a variable that was not observed.

The prediction comes from two places, and both are needed:

- The causal laws in semantic memory give the structure. An unobserved
  variable is predicted from its parents, using the coefficients the law
  states. The variables are taken in an order where a parent always comes
  before its child, so a variable whose parent is also unobserved is filled in
  from the one below it.
- The episodes in episodic memory give the offset. A law is linear, so it has
  no place for a constant term, and the constant that is really there ends up
  in the noise. The residual of an episode — what happened, minus what the law
  predicts — measures exactly that.

The episodes that are used are the ones nearest to the observation, measured
by the distance between their observed coordinates. A cosine would not pick
them out: with one observed variable it is either 1 or -1, so every episode on
the same side of it would count the same however far away it was, and the
residual would be an average over half the episodes rather than over the ones
that resemble the observation. Their residuals are averaged with weights that
fall off with the distance.

The law provides the structure and the episodes provide the offset. On data
where `x1 = 1.5 * x0 + 0.5`, the law alone gives 1.2 for `x1` when `x0` is
0.8, and the episodes lift it to 1.7.

A variable with no law is predicted by the mean of the episodes, which is the
best that can be said about it.

### Building a model out of what was learned

`causa_memory_build_scm` turns the laws into a structural causal model. Each
variable that has a law becomes an equation over its parents, with an
exogenous noise term of its own; a variable without a law is an exogenous
input.

The equations are added to the SCM's own graph as real operations, so the
model is differentiable and the abduction step can infer the parts of it that
were not observed. The variables are registered in index order, which is the
order the observed values have to be given in.

That closes the loop: the episodes become laws, the laws become a model, and
the model can be intervened on and reasoned about like any other.

## Procedural memory

The three levels below working memory each answer a different question, and
the fourth one — procedural memory — was the one that was missing.

| Level | Holds | Answers |
|-------|-------|---------|
| Working | what is happening now | what just happened |
| Episodic | what happened | what has been seen |
| Semantic | the laws | what causes what |
| Procedural | the procedures | what to do |

A procedure is a policy: a trigger and an action. The trigger is the state
the procedure applies to, and the action is what to do there.

`causa_memory_learn_procedures` groups the episodes into procedures. An
episode joins the nearest procedure when it falls within a radius of that
procedure's centre; otherwise it starts a procedure of its own. The centre
and the action are running means, so the procedure settles on the average of
the episodes it stands for.

The grouping is done with the straight-line distance between states, so the
radius is in the units of the state. The caller knows the scale of its own
data, so it can pick a radius that means something.

`causa_memory_find_procedures` retrieves the procedures that apply to a
state, the nearest first. The distance is the same one the grouping used, so
a procedure that is found is one that the learning would have joined.

A procedure that was stored by hand is stored the same way. The trigger and
the action are copied, and a procedure whose trigger is already there is
updated rather than stored again.

## Acting on what is known

The semantic level says what causes what. The procedural level says what to
do. The two together say how to reach a wanted outcome, which is what
`causa_memory_plan_intervention` works out.

The state says where things stand. The goal says what is wanted: it holds
the wanted value of the variable to be changed, and NaN for the rest, in the
same way the observation marks what was not observed.

The plan is worked out in three steps:

1. Where the goal variable stands now. The state is used as an observation
   with the goal variable left out, so the variable is predicted from what is
   known rather than read off the state.
2. How far it has to move. That is the wanted value minus where it stands.
3. What moves it. The law of the goal variable says which variables cause it
   and how much, so the change needed on a parent is the change needed on the
   goal, divided by the coefficient of that parent.

The strongest cause is the one that is acted on, because it moves the goal the
most for the least change. A goal variable with no law is exogenous, so it is
set directly.

The plan is a change to each variable. Applying it to the state and running
the model gives the goal, which is what the test checks.

Two things are left out on purpose. The plan acts on one variable, so a goal
that needs several causes at once is only partly reached. And it uses the
model's prediction of where the goal variable stands, which is the law plus
what the episodes say the law leaves out; the goal is reached to the extent
that the model is right.

## Serialization

`causa_write_scm` / `causa_read_scm` persist the variable table, parent sets,
noise assignment, and intervention flags. Variable and noise tensors are written
in a preceding tensor section and referenced by index.

A file is not trusted. Every count in it decides how much memory is set aside
or how many items are read, so each one is checked against a bound before it is
believed, and the sizes in the file are compared against the shapes that are
read alongside them. A file that says a tensor is larger than the shape it
gives is refused rather than copied into the smaller buffer.

The format version is 3. Version 2 wrote the dimensions of a memory as
floats, which have no room for a dimension above two to the twenty fourth.
Version 3 writes the whole of a model, including the operations of its graph,
so that a model loaded from a file can still be run.

## Python bindings

The causal layer is exposed to Python through the same CPython C API as the
tensor layer, with no external dependency.

| Type | Purpose |
|------|---------|
| `causa.Memory` | Hierarchical memory, collection, consolidation |
| `causa.SCM` | Variables, equations, interventions |
| `causa.Counterfactual` | A counterfactual query and its result |
| `causa.Discovery` | The result of causal discovery |
| `causa.discover` | Module-level entry point for discovery |

An equation in Python is written as

```python
scm.add_equation(y, [x], u, coefficient=1.5)
```

which builds the graph operations `u1 = coefficient * x` and `y = u1 + u`. The
noise is registered as an exogenous variable at the same time, which is what
lets the abduction infer it. The gradient requirement propagates from the
inputs to the target, so the gradient reaches the noise without the caller
having to mark the intermediate variables.

A counterfactual is constructed from four tensors and given the observed value
of every variable:

```python
cf = causa.Counterfactual(query=y, intervention_var=x,
                            intervention_val=value, evidence=observed_y)
cf.set_evidence([observed_x, observed_y])
result = cf.compute(scm)
```

### The tape

The Python binding keeps its own tape of operations, so that a tensor can be
built from other tensors without a graph.

The tape is shared by the whole module, so each tensor records where the tape
of its own graph begins — the earliest of the graphs its inputs belong to. A
backward pass finds the entries it applies by following the inputs back from
the tensor, rather than by counting backwards. Counting backwards would apply
the operations of other graphs that happened to be built in between, and would
clear the gradients of tensors that had nothing to do with this pass.

The gradients of everything the graph reaches are cleared before the walk, so
that calling `backward()` twice gives the same answer twice rather than twice
the answer. The seed is put in place after the clearing, or it would be
cleared too.

The operations are recorded with their own gradient. A negation used to be
recorded as a product, and the product's backward reads a second input that
was never written.

The tape holds a reference to every tensor that took part in an operation, and
it is only released when the module goes away. A program that builds a very
large number of graphs should be aware of that: the arena of every tensor that
takes part in an operation stays alive as long as the tape does.
