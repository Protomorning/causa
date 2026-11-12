#!/usr/bin/env python3
"""Test suite for the causa Python bindings."""

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


def test_version():
    v = causa.version()
    check("version", v == "0.1.0", f"got {v}")


def test_zeros():
    t = causa.zeros([2, 3])
    check("zeros shape", t.shape == [2, 3], f"got {t.shape}")
    check("zeros numel", t.numel == 6)
    check("zeros dtype", t.dtype == "f32", f"got {t.dtype}")
    check("zeros device", t.device == "cpu")
    data = t.tolist()
    check("zeros values", all(v == 0.0 for row in data for v in row), f"got {data}")


def test_full():
    t = causa.full([2, 2], 3.5)
    data = t.tolist()
    check("full values", all(v == 3.5 for row in data for v in row), f"got {data}")


def test_randn():
    t = causa.randn([100])
    data = t.tolist()
    mean = sum(data) / len(data)
    check("randn roughly zero mean", abs(mean) < 0.5, f"mean={mean}")
    check("randn not constant", len(set(data)) > 10)


def test_from_data():
    t = causa.tensor([[1.0, 2.0], [3.0, 4.0]])
    check("tensor shape", t.shape == [2, 2], f"got {t.shape}")
    check("tensor values", t.tolist() == [[1.0, 2.0], [3.0, 4.0]], f"got {t.tolist()}")

    t1 = causa.tensor([1.0, 2.0, 3.0])
    check("1D tensor", t1.shape == [3], f"got {t1.shape}")


def test_arithmetic():
    a = causa.tensor([1.0, 2.0, 3.0])
    b = causa.tensor([4.0, 5.0, 6.0])

    check("add", (a + b).tolist() == [5.0, 7.0, 9.0], f"got {(a + b).tolist()}")
    check("sub", (b - a).tolist() == [3.0, 3.0, 3.0], f"got {(b - a).tolist()}")
    check("mul", (a * b).tolist() == [4.0, 10.0, 18.0], f"got {(a * b).tolist()}")
    check("div", (b / a).tolist() == [4.0, 2.5, 2.0], f"got {(b / a).tolist()}")


def test_matmul():
    a = causa.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = causa.tensor([[5.0, 6.0], [7.0, 8.0]])
    c = a @ b
    expected = [[19.0, 22.0], [43.0, 50.0]]
    check("matmul", c.tolist() == expected, f"got {c.tolist()}")


def test_indexing():
    t = causa.tensor([10.0, 20.0, 30.0, 40.0])
    check("index 0", t[0] == 10.0)
    check("index -1", t[-1] == 40.0)
    check("slice", t[1:3] == [20.0, 30.0], f"got {t[1:3]}")

    t[0] = 99.0
    check("setitem", t[0] == 99.0, f"got {t[0]}")


def test_activations():
    t = causa.tensor([-1.0, 0.0, 1.0, 2.0])
    check("relu", t.relu().tolist() == [0.0, 0.0, 1.0, 2.0], f"got {t.relu().tolist()}")
    check("sigmoid", abs(t.sigmoid().tolist()[2] - 0.7310586) < 1e-5)
    check("tanh", abs(t.tanh().tolist()[3] - 0.9640276) < 1e-5)

    g = t.gelu().tolist()
    check("gelu positive", g[3] > 1.9 and g[3] < 2.0, f"got {g[3]}")


def test_reductions():
    t = causa.tensor([1.0, 2.0, 3.0, 4.0])
    check("sum", abs(t.sum().tolist()[0] - 10.0) < 1e-5, f"got {t.sum().tolist()}")
    check("mean", abs(t.mean().tolist()[0] - 2.5) < 1e-5, f"got {t.mean().tolist()}")
    check("max", abs(t.max().tolist()[0] - 4.0) < 1e-5, f"got {t.max().tolist()}")
    check("min", abs(t.min().tolist()[0] - 1.0) < 1e-5, f"got {t.min().tolist()}")


def test_shapes():
    t = causa.zeros([2, 3, 4])
    check("3D shape", t.shape == [2, 3, 4], f"got {t.shape}")
    check("3D numel", t.numel == 24)

    tr = causa.tensor([[1.0, 2.0], [3.0, 4.0]]).transpose(0, 1)
    check("transpose shape", tr.shape == [2, 2], f"got {tr.shape}")

    r = causa.zeros([2, 3]).reshape([3, 2])
    check("reshape", r.shape == [3, 2], f"got {r.shape}")


def test_requires_grad():
    t = causa.tensor([1.0, 2.0], requires_grad=True)
    check("requires_grad", t.requires_grad is True)

    t2 = causa.tensor([1.0, 2.0])
    check("default no grad", t2.requires_grad is False)


def test_backward():
    a = causa.tensor([3.0, 4.0], requires_grad=True)
    b = (a * a).sum()
    g = b.backward()
    check("backward returns grad", g is not None)
    if g is not None:
        grad_data = g.tolist()
        check("grad of x^2 sum", abs(grad_data[0] - 6.0) < 1e-4, f"got {grad_data}")
        check("grad of x^2 sum (2)", abs(grad_data[1] - 8.0) < 1e-4, f"got {grad_data}")


def test_serialization():
    t = causa.tensor([[1.5, 2.5], [3.5, 4.5]], requires_grad=True)
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        path = f.name
    try:
        causa.save(path, t)
        t2 = causa.load(path)
        check("save/load shape", t2.shape == t.shape, f"got {t2.shape}")
        check("save/load values", t2.tolist() == t.tolist(), f"got {t2.tolist()}")
        check("save/load requires_grad", t2.requires_grad is True)
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_repr():
    t = causa.zeros([2, 3])
    r = repr(t)
    check("repr contains shape", "2, 3" in r, f"got {r}")


def test_unary_gradients():
    """The unary operations have to have the right gradients.

    They used to be recorded on the tape as a product, which read a second
    input that was never written.
    """
    cases = [
        ("neg", lambda a: -a, [-1.0, -1.0]),
        ("abs", lambda a: (a * a).abs(), [6.0, 8.0]),
        ("sqrt", lambda a: (a * a).sqrt(), [1.0, 1.0]),
        ("exp", lambda a: (a * 0).exp(), [0.0, 0.0]),
    ]
    for name, f, expected in cases:
        a = causa.tensor([3.0, 4.0], requires_grad=True)
        out = f(a)
        out.backward()
        got = a.grad.tolist() if a.grad is not None else None
        check(f"the gradient of {name}", got == expected, f"got {got}")


def test_gradient_of_two_paths():
    """A gradient that arrives by two paths is the sum of them."""
    a = causa.tensor([3.0, 4.0], requires_grad=True)
    out = (a * a).sum() + a.sum()
    out.backward()
    got = a.grad.tolist() if a.grad is not None else None
    check("both paths contribute", got == [7.0, 9.0], f"got {got}")

    b = causa.tensor([3.0, 4.0], requires_grad=True)
    out2 = ((b + b) * (b + b)).sum()
    out2.backward()
    got2 = b.grad.tolist() if b.grad is not None else None
    check("a chain of operations", got2 == [24.0, 32.0], f"got {got2}")


def test_max_gradient():
    """Only the largest element gets the gradient of a maximum."""
    x = causa.tensor([1.0, 5.0, 2.0], requires_grad=True)
    m = x.max()
    m.backward()
    got = x.grad.tolist() if x.grad is not None else None
    check("the gradient of max", got == [0.0, 1.0, 0.0], f"got {got}")


def test_scalar_operands():
    """A tensor can be worked with a plain number."""
    a = causa.tensor([1.0, 2.0], requires_grad=True)
    b = a * 3.0
    check("multiplying by a number", b.tolist() == [3.0, 6.0], f"got {b.tolist()}")

    b.backward()
    got = a.grad.tolist() if a.grad is not None else None
    check("the gradient of a scaled tensor", got == [3.0, 3.0], f"got {got}")

    c = causa.tensor([4.0, 9.0])
    check("dividing by a number", (c / 2.0).tolist() == [2.0, 4.5],
          f"got {(c / 2.0).tolist()}")


def test_backward_is_repeatable():
    """A backward pass has to give the same answer every time.

    The gradients used to be added to what was there before, so a second
    call gave twice the answer. And the tape was shared by every graph, so
    one graph's backward pass disturbed another's.
    """
    a = causa.tensor([3.0, 4.0], requires_grad=True)
    b = (a * a).sum()
    first = b.backward().tolist()
    second = b.backward().tolist()

    check("the first gradient is right", first == [6.0, 8.0], f"got {first}")
    check("the second gradient is the same", second == [6.0, 8.0], f"got {second}")

    x = causa.tensor([3.0, 4.0], requires_grad=True)
    y = causa.tensor([5.0, 6.0], requires_grad=True)
    graph_a = (x * x).sum().backward().tolist()
    graph_b = (y * y).sum().backward().tolist()

    check("the first graph is undisturbed", graph_a == [6.0, 8.0], f"got {graph_a}")
    check("the second graph is right", graph_b == [10.0, 12.0], f"got {graph_b}")


def test_backward_seed():
    """The seed of a backward pass has to be used."""
    a = causa.tensor([3.0, 4.0], requires_grad=True)
    b = (a * a).sum()
    g = b.backward(grad=causa.tensor([10.0])).tolist()
    check("the seed scales the gradient", g == [60.0, 80.0], f"got {g}")


def test_input_is_checked():
    """Bad input has to be refused rather than read past."""
    try:
        t = causa.tensor([1.0, 2.0, [3.0, 4.0]])
        check("a list that mixes numbers and sequences is refused", False,
              f"got shape {t.shape}")
    except ValueError:
        check("a list that mixes numbers and sequences is refused", True)

    try:
        scm = causa.SCM()
        x0 = scm.variable("x0")
        u = scm.variable("u", trainable=True)
        for _ in range(70):
            scm.add_equation(x0, [x0], u, coefficient=1.0)
        check("a model with too many variables is refused", False, "no error")
    except ValueError:
        check("a model with too many variables is refused", True)


def test_leaf_backward_gives_no_tensor():
    """The backward pass on a leaf has nothing to give back.

    A leaf has no gradient buffer until a backward pass makes one, so the
    result of asking for it is a tensor with no data behind it. Using it as
    the right-hand side of an operation used to read through that missing
    pointer and stop the interpreter.
    """
    a = causa.zeros([2])
    g = a.backward()
    check("a leaf gives no gradient tensor", g is None or g.tolist() is None,
          f"got {g}")

    raised = False
    try:
        x = causa.zeros([2]) + g
    except (ValueError, TypeError):
        raised = True
    check("a tensor with no data behind it is refused", raised,
          "the operation went through")


def test_other_element_types_are_read_correctly():
    """A tensor of another element type has to be read and written as one.

    The bindings used to read every buffer as floats. A tensor of one byte
    per element was then read four bytes at a time, so it showed nonsense
    and a write to one element ran into the next one.
    """
    import struct

    def u32(v):
        return struct.pack("<I", v)

    def i64(v):
        return struct.pack("<q", v)

    def u64(v):
        return struct.pack("<Q", v)

    def u8(v):
        return struct.pack("<B", v)

    # A tensor of four bytes: the element type is one byte per element.
    record = u32(7) + u32(0) + u32(1)      # dtype, device, layout
    record += i64(1) + i64(4) + i64(1)     # ndim, shape, strides
    record += i64(0) + u8(0)               # offset, requires_grad
    record += u32(0)                       # the name
    record += u64(4) + bytes([1, 2, 3, 4]) # the data
    record += u8(0)                       # no gradient

    blob = u32(0x52534E54) + u32(3) + u32(1) + u32(1) + record

    path = os.path.join(tempfile.mkdtemp(prefix="causa_dtype_"), "int8.bin")
    with open(path, "wb") as f:
        f.write(blob)

    t = causa.load(path)
    check("a tensor of bytes holds the values it was given",
          t.tolist() == [1.0, 2.0, 3.0, 4.0], f"got {t.tolist()}")

    # A write to one element must not run into the next one.
    t[0] = 9.0
    check("a write to one element leaves the others alone",
          t.tolist() == [9.0, 2.0, 3.0, 4.0], f"got {t.tolist()}")


def test_transpose_gradient():
    """The gradient of a transpose is the transpose of the gradient.

    The transpose gives a view, and a view is not laid out in order, so a
    gradient that was copied over in one go would be read from the wrong
    places. The mistake only shows on a matrix that is not square.
    """
    a = causa.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    a.requires_grad = True
    b = a.transpose(0, 1)
    w = causa.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    loss = (b * w).sum()
    g = loss.backward()

    check("the gradient of a transpose is the transpose of the gradient",
          g.tolist() == [[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]], f"got {g.tolist()}")


def test_memory_capacity_is_bounded():
    """A capacity decides how much memory is set aside, so it is checked.

    A level has one slot struct for each of its slots, so a capacity of a
    billion would ask for a billion of them before anything was stored.
    """
    raised = False
    try:
        causa.Memory(capacity=[10 ** 9, 128, 1024, 1024])
    except (ValueError, RuntimeError):
        raised = True
    check("a capacity that is too large is refused", raised,
          "the memory was made")

    raised = False
    try:
        causa.Memory(capacity=[-1])
    except (ValueError, RuntimeError):
        raised = True
    check("a negative capacity is refused", raised, "the memory was made")


def main():
    print("=== Causa Python Bindings Tests ===\n")

    tests = [
        test_version,
        test_zeros,
        test_full,
        test_randn,
        test_from_data,
        test_arithmetic,
        test_matmul,
        test_indexing,
        test_activations,
        test_reductions,
        test_shapes,
        test_requires_grad,
        test_backward,
        test_unary_gradients,
        test_gradient_of_two_paths,
        test_max_gradient,
        test_scalar_operands,
        test_backward_is_repeatable,
        test_backward_seed,
        test_input_is_checked,
        test_serialization,
        test_repr,
        test_leaf_backward_gives_no_tensor,
        test_other_element_types_are_read_correctly,
        test_transpose_gradient,
        test_memory_capacity_is_bounded,
    ]

    for t in tests:
        t()

    print()
    if FAILED:
        print(f"=== {len(FAILED)} test(s) FAILED: {', '.join(FAILED)} ===")
        return 1
    print("=== All Python binding tests passed ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
