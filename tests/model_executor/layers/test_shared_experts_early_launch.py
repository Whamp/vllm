# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterator
from contextlib import AbstractContextManager, ExitStack, contextmanager, nullcontext
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import pytest
from hypothesis import given
from hypothesis import strategies as st

import vllm.envs as envs
from vllm.model_executor.layers.fused_moe.runner import (
    shared_experts as shared_module,
)
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
    SharedExperts,
    SharedExpertsOrder,
)


@pytest.fixture
def should_do_global_cleanup_after_test() -> bool:
    return False


class _FakePlatform:
    def __init__(self, *, cuda: bool = True, rocm: bool = False) -> None:
        self._cuda = cuda
        self._rocm = rocm

    def is_cuda(self) -> bool:
        return self._cuda

    def is_rocm(self) -> bool:
        return self._rocm


class _FakeStream:
    def __init__(self, name: str, timeline: list[tuple[object, ...]]) -> None:
        self.name = name
        self.timeline = timeline

    def wait_stream(self, other: "_FakeStream") -> None:
        self.timeline.append(("stream_wait", self.name, other.name))


class _FakeStreamContext:
    def __init__(self, stream: _FakeStream) -> None:
        self.stream = stream

    def __enter__(self) -> None:
        self.stream.timeline.append(("stream_enter", self.stream.name))

    def __exit__(self, *_: object) -> None:
        self.stream.timeline.append(("stream_exit", self.stream.name))


class _FakeEvent:
    def __init__(
        self,
        event_id: int,
        timeline: list[tuple[object, ...]],
    ) -> None:
        self.event_id = event_id
        self.timeline = timeline

    def record(self, stream: _FakeStream) -> None:
        self.timeline.append(("event_record", self.event_id, stream.name))

    def wait(self, stream: _FakeStream) -> None:
        self.timeline.append(("event_wait", self.event_id, stream.name))


class _FakeTensor:
    shape = (1, 8)

    def __init__(self, name: str, timeline: list[tuple[object, ...]]) -> None:
        self.name = name
        self.timeline = timeline

    def record_stream(self, stream: _FakeStream) -> None:
        self.timeline.append(("tensor_record_stream", self.name, stream.name))


class _FakeOutput:
    def __init__(
        self,
        value: tuple[object, ...],
        timeline: list[tuple[object, ...]],
    ) -> None:
        self.value = value
        self.timeline = timeline

    def record_stream(self, stream: _FakeStream) -> None:
        self.timeline.append(("output_record_stream", self.value, stream.name))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _FakeOutput):
            return self.value == other.value
        return self.value == other

    def __repr__(self) -> str:
        return repr(self.value)


class _FakeLayer:
    def __init__(self, timeline: list[tuple[object, ...]]) -> None:
        self.timeline = timeline
        self.call_count = 0

    def __call__(self, hidden_states: _FakeTensor) -> object:
        self.call_count += 1
        output = _FakeOutput(
            ("output", self.call_count, hidden_states.name),
            self.timeline,
        )
        self.timeline.append(("layer", hidden_states.name, output))
        return output


class _SharedExpertsRuntime:
    def __init__(
        self,
        *,
        early_launch: bool,
        enable_dbo: bool = True,
        platform: _FakePlatform | None = None,
        mk_overlap: bool = False,
    ) -> None:
        self.timeline: list[tuple[object, ...]] = []
        self.current_slot = 0
        self.main_stream = _FakeStream("main", self.timeline)
        self.aux_stream = _FakeStream("aux", self.timeline)
        self.layer = _FakeLayer(self.timeline)
        self._event_count = 0
        patchers = [
            patch.object(
                shared_module.envs,
                "VLLM_CUDA_SHARED_EXPERTS_EARLY_LAUNCH",
                early_launch,
                create=True,
            ),
            patch.object(
                shared_module.envs,
                "VLLM_DISABLE_SHARED_EXPERTS_STREAM",
                False,
            ),
            patch.object(shared_module.envs, "VLLM_PLE_CPU_OFFLOAD", False),
            patch.object(
                shared_module.envs,
                "VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD",
                256,
            ),
            patch.object(
                shared_module,
                "current_platform",
                platform if platform is not None else _FakePlatform(),
            ),
            patch.object(shared_module, "aux_stream", lambda: self.aux_stream),
            patch.object(shared_module, "current_stream", lambda: self.main_stream),
            patch.object(
                shared_module,
                "dbo_current_ubatch_id",
                lambda: self.current_slot,
            ),
            patch.object(shared_module.torch.cuda, "Event", self._make_event),
            patch.object(
                shared_module.torch.cuda,
                "stream",
                lambda stream: _FakeStreamContext(stream),
            ),
        ]
        with ExitStack() as patch_stack:
            for patcher in patchers:
                patch_stack.enter_context(cast(AbstractContextManager[object], patcher))

            parallel_config = SimpleNamespace(
                enable_eplb=False,
                all2all_backend=None,
                use_fi_nvl_two_sided_kernels=False,
            )
            moe_config = SimpleNamespace(moe_parallel_config=parallel_config)
            self.shared_experts = SharedExperts(
                layer=self.layer,
                moe_config=moe_config,
                enable_dbo=enable_dbo,
                mk_can_overlap_shared_experts=lambda: mk_overlap,
            )
            self._patch_stack = patch_stack.pop_all()

    def _make_event(self) -> _FakeEvent:
        event = _FakeEvent(self._event_count, self.timeline)
        self._event_count += 1
        return event

    def close(self) -> None:
        self._patch_stack.close()


@contextmanager
def _shared_experts_runtime(
    *,
    early_launch: bool,
    enable_dbo: bool = True,
    platform: _FakePlatform | None = None,
    mk_overlap: bool = False,
) -> Iterator[_SharedExpertsRuntime]:
    runtime = _SharedExpertsRuntime(
        early_launch=early_launch,
        enable_dbo=enable_dbo,
        platform=platform,
        mk_overlap=mk_overlap,
    )
    try:
        yield runtime
    finally:
        runtime.close()


def test_shared_experts_early_launch_defaults_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VLLM_CUDA_SHARED_EXPERTS_EARLY_LAUNCH", raising=False)

    assert envs.VLLM_CUDA_SHARED_EXPERTS_EARLY_LAUNCH is False


def test_shared_experts_early_launch_records_explicit_event_order() -> None:
    with _shared_experts_runtime(early_launch=True, enable_dbo=False) as runtime:
        runtime.timeline.clear()
        hidden_states = _FakeTensor("decode", runtime.timeline)

        assert runtime.shared_experts.maybe_forward_async(hidden_states)
        runtime.shared_experts.wait()
        output = runtime.shared_experts.output

        assert output == ("output", 1, "decode")
        assert runtime.timeline == [
            ("tensor_record_stream", "decode", "aux"),
            ("event_record", 0, "main"),
            ("stream_enter", "aux"),
            ("event_wait", 0, "aux"),
            ("layer", "decode", ("output", 1, "decode")),
            ("event_record", 2, "aux"),
            ("stream_exit", "aux"),
            ("event_wait", 2, "main"),
            ("output_record_stream", ("output", 1, "decode"), "main"),
        ]


def test_shared_experts_early_launch_does_not_allocate_events_on_rocm() -> None:
    with _shared_experts_runtime(
        early_launch=True,
        enable_dbo=False,
        platform=_FakePlatform(cuda=False, rocm=True),
    ) as runtime:
        hidden_states = _FakeTensor("decode", runtime.timeline)

        assert runtime._event_count == 0
        assert not runtime.shared_experts.maybe_forward_async(hidden_states)
        assert runtime.timeline == []


def test_shared_experts_early_launch_defers_to_modular_kernel_owner() -> None:
    with _shared_experts_runtime(
        early_launch=True,
        enable_dbo=False,
        mk_overlap=True,
    ) as runtime:
        runtime.timeline.clear()
        hidden_states = _FakeTensor("decode", runtime.timeline)

        assert not runtime.shared_experts.maybe_forward_async(hidden_states)
        runtime.shared_experts(
            hidden_states,
            SharedExpertsOrder.MK_INTERNAL_OVERLAPPED,
        )
        assert runtime.shared_experts.output == ("output", 1, "decode")

        assert runtime.layer.call_count == 1
        assert not any(event[0] == "event_record" for event in runtime.timeline)


def test_shared_experts_early_launch_off_preserves_legacy_path() -> None:
    with _shared_experts_runtime(early_launch=False, enable_dbo=False) as runtime:
        hidden_states = _FakeTensor("decode", runtime.timeline)

        assert not runtime.shared_experts.maybe_forward_async(hidden_states)
        assert runtime._event_count == 0
        assert runtime.timeline == []

        runtime.shared_experts.maybe_sync_shared_experts_stream(hidden_states)
        runtime.shared_experts(
            hidden_states,
            SharedExpertsOrder.MULTI_STREAM_OVERLAPPED,
        )
        output = runtime.shared_experts.output

        assert output == ("output", 1, "decode")
        assert runtime.timeline == [
            ("tensor_record_stream", "decode", "aux"),
            ("stream_wait", "aux", "main"),
            ("stream_enter", "aux"),
            ("layer", "decode", ("output", 1, "decode")),
            ("stream_exit", "aux"),
            ("stream_wait", "main", "aux"),
        ]


def test_shared_experts_early_launch_clears_slot_after_host_error() -> None:
    with _shared_experts_runtime(early_launch=True, enable_dbo=False) as runtime:

        def fail_layer(_: object) -> object:
            raise RuntimeError("shared expert test failure")

        runtime.shared_experts._layer = fail_layer
        hidden_states = _FakeTensor("decode", runtime.timeline)

        with pytest.raises(RuntimeError, match="shared expert test failure"):
            runtime.shared_experts.maybe_forward_async(hidden_states)

        assert runtime.shared_experts._output == [None, None]
        assert runtime.shared_experts._async_in_flight == [False, False]
        assert ("tensor_record_stream", "decode", "aux") in runtime.timeline
        assert ("event_record", 2, "aux") in runtime.timeline
        assert ("event_wait", 2, "main") in runtime.timeline

        runtime.shared_experts._layer = runtime.layer
        assert runtime.shared_experts.maybe_forward_async(hidden_states)
        runtime.shared_experts.wait()
        assert runtime.shared_experts.output == ("output", 1, "decode")


def test_shared_experts_early_launch_preserves_layer_error_when_cleanup_fails() -> None:
    with _shared_experts_runtime(early_launch=True, enable_dbo=False) as runtime:

        def fail_layer(_: object) -> object:
            raise ValueError("original shared-layer failure")

        def fail_wait(_: object) -> None:
            raise RuntimeError("cleanup wait failure")

        runtime.shared_experts._layer = fail_layer
        runtime.shared_experts._output_ready_event[0].wait = fail_wait

        with pytest.raises(ValueError, match="original shared-layer failure"):
            runtime.shared_experts.maybe_forward_async(
                _FakeTensor("decode", runtime.timeline)
            )

        assert runtime.shared_experts._output == [None, None]
        assert runtime.shared_experts._async_in_flight == [False, False]


def test_shared_experts_early_launch_rejects_idle_output_consumption() -> None:
    with (
        _shared_experts_runtime(early_launch=True, enable_dbo=False) as runtime,
        pytest.raises(RuntimeError, match="slot 0 has no output"),
    ):
        _ = runtime.shared_experts.output


def test_shared_experts_early_launch_discards_in_flight_output() -> None:
    with _shared_experts_runtime(early_launch=True, enable_dbo=False) as runtime:
        hidden_states = _FakeTensor("decode", runtime.timeline)
        assert runtime.shared_experts.maybe_forward_async(hidden_states)

        runtime.shared_experts.wait_and_discard_async_output()

        assert runtime.shared_experts._output == [None, None]
        assert runtime.shared_experts._async_in_flight == [False, False]
        assert runtime.shared_experts.maybe_forward_async(hidden_states)
        runtime.shared_experts.wait()
        assert runtime.shared_experts.output == ("output", 2, "decode")


def test_shared_experts_discard_clears_slot_when_event_wait_raises() -> None:
    with _shared_experts_runtime(early_launch=True, enable_dbo=False) as runtime:
        hidden_states = _FakeTensor("decode", runtime.timeline)
        assert runtime.shared_experts.maybe_forward_async(hidden_states)

        def fail_wait(_: object) -> None:
            raise RuntimeError("event wait test failure")

        runtime.shared_experts._output_ready_event[0].wait = fail_wait
        with pytest.raises(RuntimeError, match="event wait test failure"):
            runtime.shared_experts.wait_and_discard_async_output()

        assert runtime.shared_experts._output == [None, None]
        assert runtime.shared_experts._async_in_flight == [False, False]


def test_moe_runner_launches_shared_experts_before_gate_and_dispatch() -> None:
    timeline: list[object] = []

    class RoutedExperts:
        def _ensure_moe_quant_config_init(self) -> None:
            timeline.append("quant_init")

    class AsyncSharedExperts:
        def maybe_forward_async(self, hidden_states: object) -> bool:
            timeline.append(("shared_start", hidden_states))
            return True

    hidden_states = object()
    router_logits = object()
    shared_input = object()
    routed_output = object()
    combined_output = object()

    def run_gate(_: object) -> tuple[object, None]:
        timeline.append("gate")
        return router_logits, None

    def record_legacy_sync(_: object) -> None:
        timeline.append("legacy_sync")

    def dispatch(hidden: object, logits: object) -> tuple[object, object]:
        timeline.append("dispatch")
        return hidden, logits

    def apply_quant_method(**kwargs: object) -> tuple[object, object]:
        timeline.append(("quant", kwargs.get("shared_experts_overlapping", "missing")))
        return object(), routed_output

    def combine(_: object, __: object) -> object:
        timeline.append("combine")
        return combined_output

    runner = SimpleNamespace(
        routed_experts=RoutedExperts(),
        _shared_experts=AsyncSharedExperts(),
        gate=run_gate,
        _fse_fuse_gate=False,
        _maybe_sync_shared_experts_stream=record_legacy_sync,
        _sequence_parallel_context=nullcontext,
        _maybe_dispatch=dispatch,
        _apply_quant_method=apply_quant_method,
        _maybe_combine=combine,
    )

    result = MoERunner._forward_impl(
        runner,
        hidden_states,
        router_logits,
        shared_input,
    )

    assert result is combined_output
    assert timeline == [
        "quant_init",
        ("shared_start", shared_input),
        "gate",
        "dispatch",
        ("quant", True),
        "combine",
    ]


def test_moe_runner_preserves_legacy_order_when_early_launch_is_inapplicable() -> None:
    timeline: list[object] = []

    class RoutedExperts:
        def _ensure_moe_quant_config_init(self) -> None:
            timeline.append("quant_init")

    class SynchronousSharedExperts:
        def maybe_forward_async(self, _: object) -> bool:
            timeline.append("early_probe")
            return False

    def run_gate(_: object) -> tuple[object, None]:
        timeline.append("gate")
        return object(), None

    def record_legacy_sync(_: object) -> None:
        timeline.append("legacy_sync")

    def dispatch(hidden: object, logits: object) -> tuple[object, object]:
        timeline.append("dispatch")
        return hidden, logits

    def apply_quant_method(**kwargs: object) -> tuple[object, object]:
        timeline.append(("legacy_launch", kwargs["shared_experts_overlapping"]))
        return object(), object()

    def combine(_: object, __: object) -> object:
        timeline.append("combine")
        return object()

    runner = SimpleNamespace(
        routed_experts=RoutedExperts(),
        _shared_experts=SynchronousSharedExperts(),
        gate=run_gate,
        _fse_fuse_gate=False,
        _maybe_sync_shared_experts_stream=record_legacy_sync,
        _sequence_parallel_context=nullcontext,
        _maybe_dispatch=dispatch,
        _apply_quant_method=apply_quant_method,
        _maybe_combine=combine,
    )

    MoERunner._forward_impl(runner, object(), object(), object())

    assert timeline == [
        "quant_init",
        "early_probe",
        "legacy_sync",
        "gate",
        "dispatch",
        ("legacy_launch", False),
        "combine",
    ]


def test_moe_runner_preserves_gate_error_when_cleanup_also_raises() -> None:
    class RoutedExperts:
        def _ensure_moe_quant_config_init(self) -> None:
            pass

    class FailingAsyncSharedExperts:
        def maybe_forward_async(self, _: object) -> bool:
            return True

        def wait_and_discard_async_output(self) -> None:
            raise RuntimeError("cleanup test failure")

    def fail_gate(_: object) -> tuple[object, None]:
        raise ValueError("original gate failure")

    runner = SimpleNamespace(
        routed_experts=RoutedExperts(),
        _shared_experts=FailingAsyncSharedExperts(),
        gate=fail_gate,
        _fse_fuse_gate=False,
        _maybe_sync_shared_experts_stream=lambda _: None,
    )

    with pytest.raises(ValueError, match="original gate failure"):
        MoERunner._forward_impl(runner, object(), object(), object())


def test_moe_runner_discards_early_output_after_gate_error() -> None:
    timeline: list[str] = []

    class RoutedExperts:
        def _ensure_moe_quant_config_init(self) -> None:
            timeline.append("quant_init")

    class AsyncSharedExperts:
        def maybe_forward_async(self, _: object) -> bool:
            timeline.append("shared_start")
            return True

        def wait_and_discard_async_output(self) -> None:
            timeline.append("shared_discard")

    def fail_gate(_: object) -> tuple[object, None]:
        timeline.append("gate")
        raise RuntimeError("gate test failure")

    runner = SimpleNamespace(
        routed_experts=RoutedExperts(),
        _shared_experts=AsyncSharedExperts(),
        gate=fail_gate,
        _fse_fuse_gate=False,
        _maybe_sync_shared_experts_stream=lambda _: timeline.append("legacy_sync"),
    )

    with pytest.raises(RuntimeError, match="gate test failure"):
        MoERunner._forward_impl(runner, object(), object(), object())

    assert timeline == ["quant_init", "shared_start", "gate", "shared_discard"]


@given(
    operations=st.lists(
        st.tuples(
            st.sampled_from(("start", "wait", "consume", "discard")),
            st.integers(min_value=0, max_value=1),
        ),
        min_size=1,
        max_size=100,
    )
)
def test_shared_experts_dbo_lifecycle_matches_two_slot_state_machine(
    operations: list[tuple[str, int]],
) -> None:
    """Generated schedules preserve each DBO slot's async output lifecycle."""
    with _shared_experts_runtime(early_launch=True) as runtime:
        states = ["idle", "idle"]
        expected_outputs: list[object | None] = [None, None]

        for operation, slot in operations:
            runtime.current_slot = slot
            shared_experts = runtime.shared_experts

            if operation == "start":
                if states[slot] != "idle":
                    with pytest.raises(RuntimeError, match="slot .* is occupied"):
                        shared_experts.maybe_forward_async(
                            _FakeTensor(f"slot-{slot}", runtime.timeline)
                        )
                else:
                    assert shared_experts.maybe_forward_async(
                        _FakeTensor(f"slot-{slot}", runtime.timeline)
                    )
                    expected_outputs[slot] = shared_experts._output[slot]
                    states[slot] = "in_flight"
            elif operation == "wait":
                if states[slot] != "in_flight":
                    with pytest.raises(RuntimeError, match="slot .* is not in flight"):
                        shared_experts.wait()
                else:
                    shared_experts.wait()
                    states[slot] = "ready"
            elif operation == "consume":
                if states[slot] == "in_flight":
                    with pytest.raises(RuntimeError, match="slot .* was not waited"):
                        _ = shared_experts.output
                elif states[slot] == "idle":
                    with pytest.raises(RuntimeError, match="slot .* has no output"):
                        _ = shared_experts.output
                else:
                    assert shared_experts.output is expected_outputs[slot]
                    expected_outputs[slot] = None
                    states[slot] = "idle"
            else:
                shared_experts.wait_and_discard_async_output()
                expected_outputs[slot] = None
                states[slot] = "idle"

            for observed_slot, state in enumerate(states):
                assert (shared_experts._output[observed_slot] is None) == (
                    state == "idle"
                )
                assert shared_experts._async_in_flight[observed_slot] == (
                    state == "in_flight"
                )
