from dataclasses import replace
from types import SimpleNamespace

from adapters.robotwin_adapter import RobotwinAdapter
from config import RobotwinConfig


def initialized_adapter():
    adapter = RobotwinAdapter.__new__(RobotwinAdapter)
    adapter.cfg = RobotwinConfig(seed=0)
    adapter._seed_cursor = {('adjust_bottle', 0): 100002}
    adapter._scene_seed_ledger = [{'scene_seed': 100001}]
    adapter._standing = ('old batch',)
    adapter._last_scene_seed = 100001
    adapter._last_seed_retries = 1
    calls = []
    adapter._sampler = SimpleNamespace(reset=lambda: calls.append('reset'))
    return adapter, calls


def test_new_session_replays_requested_seed_and_has_its_own_ledger():
    adapter, calls = initialized_adapter()
    adapter.cfg = replace(adapter.cfg)
    assert adapter._advance_cursor('adjust_bottle', 0) == 100000
    assert adapter._scene_seed_ledger == []
    assert adapter._standing is None
    assert adapter._last_scene_seed == 100001  # old scene still needs closing
    assert adapter._last_seed_retries == 0
    assert calls == ['reset']


def test_changed_session_seed_uses_the_new_seed_domain():
    adapter, _ = initialized_adapter()
    adapter.cfg = replace(adapter.cfg, seed=1)
    assert adapter._advance_cursor('adjust_bottle', 0) == 200000
    assert adapter._advance_cursor('adjust_bottle', 0) == 200001


def test_same_config_within_a_batch_preserves_the_seed_cursor():
    adapter, calls = initialized_adapter()
    adapter.cfg = adapter.cfg
    assert adapter._advance_cursor('adjust_bottle', 0) == 100002
    assert adapter._scene_seed_ledger == [{'scene_seed': 100001}]
    assert calls == []
