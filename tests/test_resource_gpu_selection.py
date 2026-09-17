from types import SimpleNamespace

import pytest

from adapters import robotwin_adapter


@pytest.mark.parametrize("visible, expected", [(None, "0"), ("2,3", "2"), ("GPU-test", "GPU-test")])
def test_resource_measurement_targets_the_policy_gpu(monkeypatch, visible, expected):
    if visible is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=0, stdout="17, 256\n")

    monkeypatch.setattr(robotwin_adapter.subprocess, "run", run)
    sample = robotwin_adapter._ResourceSampler._pick_gpu_sampler()
    assert sample() == (17.0, 256.0)
    assert f"--id={expected}" in calls[-1]


@pytest.mark.parametrize("visible", ["", "-1"])
def test_disabled_cuda_does_not_measure_an_unrelated_gpu(monkeypatch, visible):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    def forbidden(*args, **kwargs):
        pytest.fail("GPU query must not run when CUDA is disabled")
    monkeypatch.setattr(robotwin_adapter.subprocess, "run", forbidden)
    assert robotwin_adapter._ResourceSampler._pick_gpu_sampler()() == (None, None)
