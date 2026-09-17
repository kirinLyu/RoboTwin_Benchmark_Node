import random

import numpy as np
import pytest

from adapters.robotwin_adapter import _instruction_rng


def template_selection(seed):
    with _instruction_rng(seed):
        templates = list(range(40))
        random.shuffle(templates)
        return random.choice(templates), int(np.random.choice(templates))


def test_instruction_replays_even_after_unrelated_global_random_draws():
    first = template_selection(100001)
    random.seed(99)
    np.random.seed(1234)
    random.random()
    np.random.rand(100)
    assert template_selection(100001) == first


def test_instruction_rng_preserves_simulator_rng_even_on_error():
    python_before, numpy_before = random.getstate(), np.random.get_state()
    with pytest.raises(RuntimeError):
        with _instruction_rng(100001):
            random.random()
            np.random.rand()
            raise RuntimeError('template error')
    assert random.getstate() == python_before
    actual = np.random.get_state()
    assert actual[0] == numpy_before[0]
    np.testing.assert_array_equal(actual[1], numpy_before[1])
    assert actual[2:] == numpy_before[2:]
