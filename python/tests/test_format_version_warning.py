"""Loader advisories reach Python as warnings, not as stderr noise.

The loader does not refuse a model whose schema version it does not recognise.
It first lets the model speak for itself: a model carrying verification cases
states what its producer computed, and reproducing those outputs demonstrates
this build executes it correctly — better evidence than a version string. Only
when there is nothing to verify does the version become the sole evidence, and
then a mismatch is an advisory.

The C++ tests pin that decision. These pin the plumbing that carries it into
Python, which matters because the alternative — a library writing to stderr —
cannot be filtered, captured, or escalated.
"""

from __future__ import annotations

import warnings

import pytest

import omle_runtime as omr

from conftest import make_model_bytes


def _stump_ir():
    """conftest's 2-feature stump, as IR so metadata can be re-stamped."""
    from omle.proto.convert import get_pb2, proto_to_ir

    msg = get_pb2().OMLEModel()
    msg.ParseFromString(make_model_bytes(2, [(0, 0.5, 2.0, -2.0)]))
    return proto_to_ir(msg)


def _with_version(version: str) -> bytes:
    """A working 2-feature model, re-stamped with *version* and no verification."""
    from omle.proto.convert import ir_to_bytes

    model = _stump_ir()
    model.metadata.format_version = version
    model.verification = None
    return ir_to_bytes(model)


def _load_capturing(data: bytes):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = omr.Model.load_bytes(data)
    return model, [str(c.message) for c in caught
                   if issubclass(c.category, RuntimeWarning)
                   and "omle-runtime" in str(c.message)]


def test_supported_version_loads_without_warning():
    model, msgs = _load_capturing(_with_version("0.1.0"))
    assert model.num_inputs == 1
    assert msgs == []


def test_absent_version_loads_without_warning():
    # Descriptive metadata; a model without it is not suspect.
    model, msgs = _load_capturing(_with_version(""))
    assert model.num_inputs == 1
    assert msgs == []


@pytest.mark.parametrize("version", ["0.2.0", "1.0.0", "9.9.9"])
def test_unsupported_version_warns_but_loads(version):
    model, msgs = _load_capturing(_with_version(version))
    # Loads: the runtime reports, it does not decide for the caller.
    assert model.num_inputs == 1
    assert len(msgs) == 1, msgs
    assert version in msgs[0]
    assert "verification" in msgs[0], (
        "the advisory should say why the version had to be trusted"
    )


def test_the_advisory_is_a_warning_and_so_can_be_escalated():
    """Routed through the warnings module, so -W error works on it.

    This is the property that a write to stderr would not have: a deployment
    that wants a version mismatch to be fatal can make it fatal without the
    library offering an option for it.
    """
    data = _with_version("0.2.0")
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(RuntimeWarning, match="0.2.0"):
            omr.Model.load_bytes(data)


def test_a_verified_model_is_not_second_guessed():
    """With verification present and passing, the version stops mattering."""
    import omle
    from omle.ir.verification import (
        ModelVerification,
        NumericTolerance,
        TensorRef,
        VerificationCase,
    )
    from omle.proto.convert import ir_to_bytes

    TensorEntry, Tensor = omle.TensorEntry, omle.Tensor
    TensorType, DataType = omle.TensorType, omle.DataType

    model = _stump_ir()
    model.metadata.format_version = "0.2.0"

    # X = [[1.0, 0.0]] → feat0 = 1.0 ≥ 0.5 → right leaf, −2.0
    model.tensor_entries = list(model.tensor_entries or []) + [
        TensorEntry(id="v_X", dense=Tensor(
            name="X",
            type=TensorType(dtype=DataType.FLOAT32, shape=[1, 2]),
            float32_data=[1.0, 0.0])),
        TensorEntry(id="v_score", dense=Tensor(
            name="score",
            type=TensorType(dtype=DataType.FLOAT32, shape=[1]),
            float32_data=[-2.0])),
    ]
    model.verification = ModelVerification(
        cases=[VerificationCase(inputs=[TensorRef(id="v_X")],
                                expected_outputs=[TensorRef(id="v_score")])],
        tolerance=NumericTolerance(atol=omle.Scalar(float_value=1e-5),
                                   rtol=omle.Scalar(float_value=1e-5)),
    )

    loaded, msgs = _load_capturing(ir_to_bytes(model))
    assert loaded.num_inputs == 1
    assert msgs == [], (
        "a model that reproduces its recorded outputs should load silently, "
        f"whatever version it declares: {msgs}"
    )
