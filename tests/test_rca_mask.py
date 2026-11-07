"""
Unit tests for compute_advantage_mask on synthetic advantages
Run manually:
    python -m pytest tests/test_rca_mask.py -v
or just:
    python tests/test_rca_mask.py
"""
import sys
from pathlib import Path


import numpy as np
from reflectrl.training.rca import compute_advantage_mask, credit_uniform_mask


def test_uniform_credit_reduces_to_vanilla_advantage():
    """The key sanity property: c~_i = 1/k for all i
    -> every covered token just gets scalar_advantage unchanged. This is
    what makes the uniform-mask ablation equivalent to vanilla/Dr. GRPO."""
    modules = ["a", "b", "c"]
    credit = credit_uniform_mask(modules)
    spans = {"a": (0, 10), "b": (10, 25), "c": (25, 40)}
    mask = compute_advantage_mask(spans, credit, total_length=40, scalar_advantage=2.5)
    assert np.allclose(mask, 2.5), f"expected uniform 2.5 everywhere covered, got {mask}"


def test_skewed_credit_scales_proportionally():
    """A module credited 2x another should get 2x the per-token advantage
    magnitude (same sign as scalar_advantage)."""
    modules = ["a", "b"]
    credit = {"a": 2.0 / 3.0, "b": 1.0 / 3.0}  # a is 2x b
    spans = {"a": (0, 5), "b": (5, 10)}
    mask = compute_advantage_mask(spans, credit, total_length=10, scalar_advantage=3.0)
    a_val = mask[0]
    b_val = mask[5]
    assert np.allclose(mask[0:5], a_val)
    assert np.allclose(mask[5:10], b_val)
    assert abs(a_val - 2 * b_val) < 1e-9, f"expected a's per-token advantage to be 2x b's, got a={a_val} b={b_val}"
    # k=2, so a_val = scalar_advantage * k * credit['a'] = 3.0 * 2 * (2/3) = 4.0
    assert abs(a_val - 4.0) < 1e-9, f"expected a_val=4.0, got {a_val}"
    assert abs(b_val - 2.0) < 1e-9, f"expected b_val=2.0, got {b_val}"


def test_uncovered_tokens_get_zero():
    """Tokens outside any declared module span (prompt/template tokens, or
    gaps between spans) must get 0 -- they shouldn't carry gradient signal
    from this mask."""
    modules = ["a"]
    credit = {"a": 1.0}
    spans = {"a": (5, 8)}  # only tokens 5,6,7 covered, out of 20
    mask = compute_advantage_mask(spans, credit, total_length=20, scalar_advantage=10.0)
    assert np.allclose(mask[0:5], 0.0)
    assert np.allclose(mask[8:20], 0.0)
    assert np.allclose(mask[5:8], 10.0 * 1 * 1.0)  # k=1, credit=1.0 -> scalar_advantage unchanged


def test_zero_credit_module_gets_zero_advantage():
    """A module with 0 credit should get 0 advantage on its tokens, even
    though other modules in the same distribution have nonzero credit."""
    modules = ["a", "b"]
    credit = {"a": 1.0, "b": 0.0}
    spans = {"a": (0, 5), "b": (5, 10)}
    mask = compute_advantage_mask(spans, credit, total_length=10, scalar_advantage=5.0)
    assert np.allclose(mask[5:10], 0.0), "zero-credit module should get zero advantage"
    assert np.allclose(mask[0:5], 10.0), "k=2, credit=1.0 -> 5.0*2*1.0=10.0"


def test_out_of_bounds_span_raises():
    modules = ["a"]
    credit = {"a": 1.0}
    spans = {"a": (0, 50)}  # exceeds total_length
    try:
        compute_advantage_mask(spans, credit, total_length=10, scalar_advantage=1.0)
        raise AssertionError("expected ValueError for out-of-bounds span")
    except ValueError:
        pass


def test_advantage_sign_preserved():
    """A negative scalar_advantage (failed rollout) should produce negative
    per-token advantages, not get accidentally flipped by the masking math."""
    modules = ["a", "b"]
    credit = credit_uniform_mask(modules)
    spans = {"a": (0, 5), "b": (5, 10)}
    mask = compute_advantage_mask(spans, credit, total_length=10, scalar_advantage=-1.5)
    assert np.all(mask <= 0), f"expected all non-positive with negative scalar_advantage, got {mask}"
    assert np.allclose(mask, -1.5)


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"PASS: {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"FAIL: {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} passed")
    if passed != len(tests):
        sys.exit(1)
