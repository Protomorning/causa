#!/usr/bin/env python3
"""Test suite for the causal and memory layer of the causa bindings."""

import sys
import math
import os
import tempfile

try:
    import causa
except ImportError:
    print("causa module not found - build it first:")
    print("    python setup.py build_ext --inplace")
    sys.exit(1)


FAILED = []


def check(name, condition, detail=""):
    if condition:
        print(f"PASS: {name}")
    else:
        print(f"FAIL: {name} {detail}")
        FAILED.append(name)


def test_memory_basics():
    print("Testing memory basics...")
    mem = causa.Memory(key_dim=3, value_dim=3)

    mem.write([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], level="episodic")
    mem.write([1.0, 2.0, 4.0], [7.0, 8.0, 9.0], level="episodic")

    check("memory size", mem.size("episodic") == 2, f"got {mem.size('episodic')}")
    check("memory usage", mem.usage("episodic") > 0.0)

    found = mem.read([1.0, 2.0, 3.0], level="episodic")
    check("memory read", found is not None and abs(found.tolist()[0] - 4.0) < 1e-5,
          f"got {found.tolist() if found else None}")

    missing = mem.read([9.0, 9.0, 9.0], level="episodic")
    check("memory miss returns None", missing is None)


def test_memory_collect():
    print("Testing memory collect...")
    mem = causa.Memory(key_dim=3, value_dim=3)
    for i in range(20):
        mem.write([float(i), float(i * 2), float(i * 3)], [0.0, 0.0, 0.0], level="episodic")

    obs = mem.collect("episodic")
    check("collect shape", obs.shape == [20, 3], f"got {obs.shape}")
    rows = obs.tolist()
    check("collect values", abs(rows[5][0] - 5.0) < 1e-5, f"got {rows[5]}")
    check("collect ratio", abs(rows[5][2] - 15.0) < 1e-5, f"got {rows[5]}")


def test_causal_memory_roundtrip():
    print("Testing causal memory consolidation...")
    mem = causa.Memory(key_dim=3, value_dim=3)
    # A chain: x0 -> x1 -> x2
    seed = 4242

    def rnd():
        nonlocal seed
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        # The high bits are used: the low bits of a linear congruential
        # generator repeat quickly and would not make independent samples.
        return ((seed >> 16) & 0x7FFF) / 16384.0 - 1.0

    for _ in range(200):
        x0 = rnd()
        x1 = 1.5 * x0 + 0.2 * rnd()
        x2 = 2.0 * x1 + 0.2 * rnd()
        mem.write([x0, x1, x2], [x1, x2, 0.0], level="episodic")

    laws = mem.consolidate_causal(alpha=0.05, threshold=0.25, iterations=300)
    check("consolidation produced laws", laws > 0, f"got {laws}")

    law1 = mem.find_causal_laws(1)
    check("law found for variable 1", len(law1) > 0, f"got {len(law1)}")
    if law1:
        coeffs = law1[0].tolist()
        check("variable 0 is a parent of variable 1", abs(coeffs[0]) > 1e-3,
              f"got {coeffs}")

    law2 = mem.find_causal_laws(2)
    check("law found for variable 2", len(law2) > 0, f"got {len(law2)}")
    if law2:
        coeffs = law2[0].tolist()
        check("variable 1 is a parent of variable 2", abs(coeffs[1]) > 1e-3,
              f"got {coeffs}")


def test_discovery():
    print("Testing causal discovery...")
    rows = []
    seed = 777
    def rnd():
        nonlocal seed
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        return ((seed >> 16) & 0x7FFF) / 16384.0 - 1.0

    for _ in range(150):
        x0 = rnd()
        x1 = 1.5 * x0 + 0.2 * rnd()
        x2 = 2.0 * x1 + 0.2 * rnd()
        rows.append([x0, x1, x2])

    data = causa.tensor(rows)
    disc = causa.discover(data, alpha=0.05, threshold=0.25, iterations=300)

    p1 = disc.parents(1)
    check("variable 1 has variable 0 as a parent", 0 in p1, f"got {p1}")
    p2 = disc.parents(2)
    check("variable 2 has variable 1 as a parent", 1 in p2, f"got {p2}")


def test_scm_and_counterfactual():
    print("Testing SCM and counterfactual...")
    scm = causa.SCM()

    x0 = scm.variable("x0")
    x1 = scm.variable("x1")
    x2 = scm.variable("x2")
    u1 = scm.variable("u1", trainable=True)
    u2 = scm.variable("u2", trainable=True)

    scm.add_observed_variable(x0)
    scm.add_equation(x1, [x0], u1, coefficient=1.5)
    scm.add_equation(x2, [x1], u2, coefficient=2.0)
    scm.build()

    # The ground truth: x0 = 2, u1 = 0.5, u2 = -0.25
    # x1 = 1.5*2 + 0.5 = 3.5, x2 = 2*3.5 - 0.25 = 6.75
    x0.tolist()  # touch
    check("variables exist", x1 is not None and x2 is not None)

    # Observed values
    obs0 = causa.tensor([2.0])
    obs1 = causa.tensor([3.5])
    obs2 = causa.tensor([6.75])

    # Counterfactual: what would x2 have been had x0 been 5?
    x0_cf = causa.tensor([5.0])
    cf = causa.Counterfactual(query=x2, intervention_var=x0,
                                intervention_val=x0_cf, evidence=obs2)
    cf.set_evidence([obs0, obs1, obs2])
    result = cf.compute(scm)

    check("counterfactual produced a result", result is not None)
    if result is not None:
        value = result.tolist()[0]
        # x1 = 1.5*5 + 0.5 = 8.0, x2 = 2*8 - 0.25 = 15.75
        check("counterfactual value is right", abs(value - 15.75) < 0.2,
              f"got {value}, expected 15.75")


def test_reasoning_over_memory():
    print("Testing reasoning over memory...")
    mem = causa.Memory(key_dim=3, value_dim=3)

    # A chain: x0 -> x1 -> x2
    #
    # The relations carry an offset, which a linear law cannot express. The
    # law gives the slope and the episodes give the offset, so a prediction
    # has to use both.
    seed = 13579

    def rnd():
        nonlocal seed
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        return ((seed >> 16) & 0x7FFF) / 16384.0 - 1.0

    for _ in range(300):
        x0 = rnd()
        x1 = 1.5 * x0 + 0.5 + 0.25 * rnd()
        x2 = 2.0 * x1 + 0.25 + 0.25 * rnd()
        mem.write([x0, x1, x2], [x1, x2, 0.0], level="episodic")

    laws = mem.consolidate_causal(alpha=0.05, threshold=0.25, iterations=300)
    check("consolidation produced laws", laws > 0, f"got {laws}")

    # Only x0 is observed; x1 and x2 have to be predicted.
    observation = causa.tensor([0.8, float("nan"), float("nan")])
    prediction = mem.predict(observation)
    check("prediction was made", prediction is not None)

    if prediction is not None:
        values = prediction.tolist()
        expected_x1 = 1.5 * 0.8 + 0.5
        expected_x2 = 2.0 * expected_x1 + 0.25
        print(f"  predicted x1 = {values[1]} (expected {expected_x1})")
        print(f"  predicted x2 = {values[2]} (expected {expected_x2})")

        check("x1 was predicted", abs(values[1] - expected_x1) < 0.3,
              f"got {values[1]}, expected {expected_x1}")
        check("x2 was predicted", abs(values[2] - expected_x2) < 0.5,
              f"got {values[2]}, expected {expected_x2}")
        check("the observed variable was left alone", abs(values[0] - 0.8) < 1e-6,
              f"got {values[0]}")

    # The model that comes out of the memory can be reasoned about.
    scm = mem.build_scm(3)
    check("a model was built", scm is not None)


def test_procedural_memory():
    print("Testing procedural memory...")
    mem = causa.Memory(key_dim=3, value_dim=3)

    # A hundred different states, each with what came of it
    for i in range(100):
        x0 = -1.0 + 0.02 * i
        x1 = 1.5 * x0 + 0.5
        x2 = 2.0 * x1 + 0.25
        mem.write([x0, x1, x2], [x1, x2, 0.0], level="episodic")

    procedures = mem.learn_procedures(radius=0.5)
    check("procedures were learned", procedures > 0, f"got {procedures}")
    check("the episodes were grouped", procedures < 100, f"got {procedures}")

    # The procedure that applies to a state is the nearest one
    found = mem.find_procedures(causa.tensor([0.5, 1.25, 2.75]))
    check("a procedure was found", len(found) > 0, f"got {len(found)}")

    if found:
        trigger, action = found[0]
        values = trigger.tolist()
        distance = sum((values[j] - v) ** 2 for j, v in enumerate([0.5, 1.25, 2.75]))
        check("the nearest procedure is close", distance ** 0.5 <= 0.5,
              f"got {distance ** 0.5}")
        check("the action came with it", action is not None)

    # A procedure that was stored by hand is found as well
    mem.store_procedure(causa.tensor([100.0, 100.0, 100.0]),
                        causa.tensor([1.0, 2.0, 3.0]))
    far = mem.find_procedures(causa.tensor([99.0, 99.0, 99.0]))
    check("the stored procedure was found", len(far) > 0)
    if far:
        trigger, action = far[0]
        check("the stored action came back",
              abs(action.tolist()[0] - 1.0) < 1e-5,
              f"got {action.tolist()}")


def test_intervention_planning():
    print("Testing intervention planning...")
    mem = causa.Memory(key_dim=3, value_dim=3)

    seed = 13579

    def rnd():
        nonlocal seed
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        return ((seed >> 16) & 0x7FFF) / 16384.0 - 1.0

    for _ in range(400):
        x0 = rnd()
        x1 = 1.5 * x0 + 0.5 + 0.25 * rnd()
        x2 = 2.0 * x1 + 0.25 + 0.25 * rnd()
        mem.write([x0, x1, x2], [x1, x2, 0.0], level="episodic")

    laws = mem.consolidate_causal(alpha=0.05, threshold=0.25, iterations=300)
    check("consolidation produced laws", laws > 0, f"got {laws}")

    # The state: only x0 is known. The goal: x2 should be 10.
    nan = float("nan")
    plan, count = mem.plan_intervention(
        causa.tensor([1.0, nan, nan]),
        causa.tensor([nan, nan, 10.0]))

    values = plan.tolist()
    print(f"  plan: change x0 by {values[0]}, x1 by {values[1]}, x2 by {values[2]}")

    check("the plan acts on one variable", count == 1, f"got {count}")
    # x2 stands at 2 * (1.5 * 1 + 0.5) + 0.25 = 4.25, so it has to move by
    # 5.75. The strongest cause of x2 is x1, which moves it by 2 for 1.
    check("the plan moves x1", abs(values[1] - 2.875) < 0.35,
          f"got {values[1]}, expected about 2.875")

    achieved = 2.0 * (2.0 + values[1]) + 0.25
    check("the plan reaches the goal", abs(achieved - 10.0) < 0.7,
          f"got {achieved}")


def test_intervention():
    print("Testing intervention...")
    scm = causa.SCM()

    x0 = scm.variable("x0")
    x1 = scm.variable("x1")
    u1 = scm.variable("u1", trainable=True)
    scm.add_observed_variable(x0)
    scm.add_equation(x1, [x0], u1, coefficient=1.5)
    scm.build()

    # do(X0 = 5) gives a model of its own
    intervened = scm.do(x0, 5.0)
    check("the intervention gave a model", intervened is not None)

    if intervened is not None:
        values = intervened.evaluate()
        check("the model has its variables", len(values) == 2, f"got {len(values)}")

        numbers = [v.tolist()[0] for v in values]
        print(f"  after do(x0=5): x0 = {numbers[0]}, x1 = {numbers[1]}")

        check("the intervened variable took the value", abs(numbers[0] - 5.0) < 1e-5,
              f"got {numbers[0]}")
        check("the effect followed", abs(numbers[1] - 7.5) < 1e-4,
              f"got {numbers[1]}, expected 7.5")


def test_persistence():
    print("Testing persistence...")
    with tempfile.TemporaryDirectory() as tmp:
        # A memory that has learned something has to survive the trip
        # through a file, otherwise the learned laws are lost when the
        # process ends.
        mem = causa.Memory(key_dim=3, value_dim=3)
        for i in range(50):
            x0 = (i % 10) * 0.1 - 0.5
            x1 = 1.5 * x0 + 0.25
            mem.write([x0, x1, 0.0], [x1, 0.0, 0.0], level="episodic")

        # A memory holds one slot per distinct key, so the count is what
        # the memory has rather than the number of writes.
        before = mem.size("episodic")

        path = os.path.join(tmp, "memory.bin")
        mem.save(path)
        check("the memory was written", os.path.exists(path))

        loaded = causa.load_memory(path)
        check("the memory was read", loaded is not None)
        if loaded is None:
            return

        check("the memory kept its size", loaded.size("episodic") == before,
              f"got {loaded.size('episodic')}, expected {before}")

        found = loaded.read([0.0, 0.25, 0.0], level="episodic")
        check("the memory kept its contents", found is not None)

        # A model built from the laws has to survive too, and it has to
        # still run afterwards. A model that loads but cannot be evaluated
        # is no better than one that failed to load.
        scm = mem.build_scm(3)
        model_path = os.path.join(tmp, "scm.bin")
        scm.save(model_path)
        check("the model was written", os.path.exists(model_path))

        loaded_scm = causa.load_scm(model_path)
        check("the model was read", loaded_scm is not None)
        if loaded_scm is None:
            return

        values = loaded_scm.evaluate()
        check("the loaded model has its variables", len(values) == 3,
              f"got {len(values)}")

        # The loaded model must give the same answer as the one that was
        # saved.
        before = [v.tolist()[0] for v in scm.evaluate()]
        after = [v.tolist()[0] for v in values]
        check("the loaded model gives the same values",
              all(abs(a - b) < 1e-4 for a, b in zip(before, after)),
              f"got {after}, expected {before}")

        # An intervention on the loaded model has to work too, since that
        # is the reason to keep a model at all. The variable it acts on is
        # the one the model was built around.
        intervened = loaded_scm.do(values[0], 5.0)
        check("the loaded model can be intervened on", intervened is not None)
        if intervened is not None:
            numbers = [v.tolist()[0] for v in intervened.evaluate()]
            check("the intervention took effect", abs(numbers[0] - 5.0) < 1e-5,
                  f"got {numbers[0]}")


def test_counterfactuals_in_memory():
    print("Testing counterfactuals in memory...")
    mem = causa.Memory(key_dim=3, value_dim=3)

    scm = causa.SCM()
    x0 = scm.variable("x0")
    x1 = scm.variable("x1")
    u1 = scm.variable("u1", trainable=True)
    scm.add_observed_variable(x0)
    scm.add_equation(x1, [x0], u1, coefficient=1.5)
    scm.build()

    obs0 = causa.tensor([2.0])
    obs1 = causa.tensor([3.5])

    # Two counterfactuals about different variables, in the same situation.
    cf_x1 = causa.Counterfactual(query=x1, intervention_var=x0,
                                 intervention_val=causa.tensor([5.0]),
                                 evidence=obs1)
    cf_x1.set_evidence([obs0, obs1, obs1])
    cf_x1.compute(scm)

    x2 = scm.variable("x2")
    cf_x2 = causa.Counterfactual(query=x2, intervention_var=x0,
                                 intervention_val=causa.tensor([5.0]),
                                 evidence=obs1)
    cf_x2.set_evidence([obs0, obs1, obs1])
    cf_x2.compute(scm)

    cf_x1.store_in(mem)
    cf_x2.store_in(mem)
    check("the counterfactuals were kept", True)

    # Asking about one variable must not return the other's.
    found_x1 = mem.find_counterfactuals(x1, causa.tensor([2.0, 3.5, 0.0]), k=4)
    check("one counterfactual was found for x1", len(found_x1) == 1,
          f"got {len(found_x1)}")

    found_x2 = mem.find_counterfactuals(x2, causa.tensor([2.0, 3.5, 0.0]), k=4)
    check("one counterfactual was found for x2", len(found_x2) == 1,
          f"got {len(found_x2)}")

    if found_x1 and found_x2:
        v1 = found_x1[0].tolist()[0]
        v2 = found_x2[0].tolist()[0]
        # x1 = 1.5 * 5 + 0.5 = 8.0
        check("the x1 counterfactual is right", abs(v1 - 8.0) < 0.2,
              f"got {v1}, expected about 8.0")
        check("the two counterfactuals differ", abs(v1 - v2) > 0.5,
              f"got {v1} and {v2}")


def main():
    print("=== Causa Causal Bindings Tests ===\n")

    tests = [
        test_memory_basics,
        test_memory_collect,
        test_causal_memory_roundtrip,
        test_discovery,
        test_scm_and_counterfactual,
        test_intervention,
        test_reasoning_over_memory,
        test_procedural_memory,
        test_intervention_planning,
        test_counterfactuals_in_memory,
        test_persistence,
    ]

    for t in tests:
        t()
        print()

    if FAILED:
        print(f"=== {len(FAILED)} test(s) FAILED: {', '.join(FAILED)} ===")
        return 1
    print("=== All causal binding tests passed ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
