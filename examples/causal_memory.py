#!/usr/bin/env python3
"""A worked example of the causal memory system.

The library is one system: episodes become causal laws, the laws become a
model, and the model can be used to predict, to intervene, and to work out
what to do.

Run it with:

    python examples/causal_memory.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import causa


def make_generator(seed):
    """A generator of observations from a chain X0 -> X1 -> X2.

    The relations carry an offset, which a linear law cannot express. The
    law gives the slope and the episodes give the offset.
    """
    state = {"seed": seed}

    def rnd():
        state["seed"] = (state["seed"] * 1103515245 + 12345) & 0x7FFFFFFF
        return ((state["seed"] >> 16) & 0x7FFF) / 16384.0 - 1.0

    def observe():
        x0 = rnd()
        x1 = 1.5 * x0 + 0.5 + 0.25 * rnd()
        x2 = 2.0 * x1 + 0.25 + 0.25 * rnd()
        return x0, x1, x2

    return observe


def main():
    print("=" * 62)
    print("Causal memory")
    print("=" * 62)

    # ---- 1. Experience -----------------------------------------------------
    #
    # The episodes are all the system has at the start. Each one holds what
    # was observed and what came of it.
    memory = causa.Memory(key_dim=3, value_dim=3)
    observe = make_generator(20240917)

    for _ in range(400):
        x0, x1, x2 = observe()
        memory.write([x0, x1, x2], [x1, x2, 0.0], level="episodic")

    print(f"\nepisodes recorded: {memory.size('episodic')}")

    # ---- 2. Learning --------------------------------------------------------
    #
    # Consolidation runs causal discovery over the episodes and writes the
    # laws it finds into semantic memory.
    laws = memory.consolidate_causal(alpha=0.05, threshold=0.25, iterations=300)
    print(f"causal laws learned: {laws}")

    for child in range(3):
        found = memory.find_causal_laws(child)
        for law in found:
            coefficients = law.tolist()
            parents = [i for i, c in enumerate(coefficients) if abs(c) > 1e-3]
            if parents:
                terms = ", ".join(f"x{p} * {coefficients[p]:.3f}" for p in parents)
                print(f"  x{child} = {terms}")

    # ---- 3. Prediction -----------------------------------------------------
    #
    # Given a partial observation, the memory fills in what was not seen.
    # The law gives the structure and the episodes give the offset.
    nan = float("nan")
    partial = causa.tensor([0.8, nan, nan])
    prediction = memory.predict(partial)
    values = prediction.tolist()
    print(f"\nprediction from x0 = 0.8")
    print(f"  x1 = {values[1]:.3f}   (expected {1.5 * 0.8 + 0.5:.3f})")
    print(f"  x2 = {values[2]:.3f}   (expected {2 * (1.5 * 0.8 + 0.5) + 0.25:.3f})")

    # ---- 4. A model --------------------------------------------------------
    #
    # The laws become a structural causal model, which can be intervened on.
    model = memory.build_scm(3)
    print(f"\nmodel built from the laws")

    # ---- 5. Planning -------------------------------------------------------
    #
    # The laws say what moves the goal and by how much, so the intervention
    # that reaches a wanted value follows from them.
    plan, count = memory.plan_intervention(
        causa.tensor([1.0, nan, nan]),    # what is known
        causa.tensor([nan, nan, 10.0]))   # what is wanted
    changes = plan.tolist()
    print(f"\nplan to reach x2 = 10:")
    for i, change in enumerate(changes):
        if change:
            print(f"  change x{i} by {change:+.3f}")
    print(f"  (the plan acts on {count} variable)")

    # ---- 6. Procedures -----------------------------------------------------
    #
    # The procedural level holds what to do where. The episodes are grouped
    # into procedures, and the one that applies to a state is the nearest.
    procedures = memory.learn_procedures(radius=0.5)
    print(f"\nprocedures learned: {procedures}")

    found = memory.find_procedures(causa.tensor([0.5, 1.25, 2.75]))
    if found:
        trigger, action = found[0]
        print(f"nearest procedure: {[round(v, 3) for v in trigger.tolist()]}"
              f" -> {[round(v, 3) for v in action.tolist()]}")

    print()


if __name__ == "__main__":
    main()
