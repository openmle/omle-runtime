import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))  # omleruntime package

# Use the omle Python IR for fixture generation to stay in sync with the
# current proto format (NameRef for NodeInput.name, Tensor sub-messages for
# split_threshold / leaf_value).
_OMLE_SRC = HERE.parent.parent.parent / "omle" / "src"
if _OMLE_SRC.exists():
    sys.path.insert(0, str(_OMLE_SRC))

import pytest


def _make_stump(feat, threshold, lv, rv):
    from omle.ir.bodies import Tree
    from omle.ir.tensor import Tensor
    from omle.ir.types import TensorValue
    from omle.ir.enums import TreeNodeKind, TreeSplitOp
    return Tree(
        num_nodes=3,
        node_kind=[TreeNodeKind.BRANCH, TreeNodeKind.LEAF, TreeNodeKind.LEAF],
        split_feature=[feat, 0, 0],
        split_threshold=TensorValue.of_tensor(
            Tensor(float32_data=[threshold, 0.0, 0.0])),
        split_op=[TreeSplitOp.LESS_THAN,
                  TreeSplitOp.SPLIT_OP_UNSPECIFIED,
                  TreeSplitOp.SPLIT_OP_UNSPECIFIED],
        children_index=[1, 2],
        children_offset=[0, -1, -1],
        children_count=[2, 0, 0],
        default_child=[2, 0, 0],
        leaf_value=TensorValue.of_tensor(
            Tensor(float32_data=[0.0, lv, rv])),
    )


def make_model_bytes(n_features, stumps):
    """Build a TreeEnsemble proto model with given stumps.

    stumps: list of (feature_index, threshold, left_value, right_value)
    Uses a single matrix input named "X" with shape [-1, n_features].
    """
    from omle.ir.model import OMLEModel
    from omle.ir.interface import InputSpec, OutputSpec
    from omle.ir.node import Node, NodeInput, NodeOutput
    from omle.ir.types import NameRef, TensorType
    from omle.ir.bodies import TreeEnsemble
    from omle.ir.enums import TreeAggregation, PostTransform, DataType
    from omle.proto.convert import ir_to_bytes

    te = TreeEnsemble(
        aggregation=TreeAggregation.SUM,
        post_transform=PostTransform.IDENTITY,
        trees=[_make_stump(feat, thr, lv, rv) for feat, thr, lv, rv in stumps],
    )

    node = Node(
        domain="omle.ml",
        op="TreeEnsemble",
        inputs=[NodeInput(name=NameRef("X"))],
        outputs=[NodeOutput(name="score")],
        tree_ensemble=te,
    )

    model = OMLEModel(
        inputs=[InputSpec(name="X", type=TensorType(dtype=DataType.FLOAT32,
                                                    shape=[-1, n_features]))],
        outputs=[OutputSpec(name="score")],
        nodes=[node],
    )

    return ir_to_bytes(model)


@pytest.fixture(scope="session")
def model_bytes_2f():
    """2-feature model: feat0 < 0.5 → +2.0, else −2.0."""
    return make_model_bytes(2, [(0, 0.5, 2.0, -2.0)])


@pytest.fixture(scope="session")
def model(model_bytes_2f):
    import omleruntime as omr
    return omr.load_bytes(model_bytes_2f)


def make_label_encoder_model_bytes(input_name, labels):
    """Build a single-input LabelEncoder model.

    The input tensor is named `input_name` with dtype STRING, shape [-1, 1].
    Labels are encoded in order (first label → 0, second → 1, …).
    """
    from omle.ir.model import OMLEModel
    from omle.ir.interface import InputSpec, OutputSpec
    from omle.ir.node import Node, NodeInput, NodeOutput, Attribute
    from omle.ir.types import NameRef, TensorType
    from omle.ir.enums import DataType
    from omle.proto.convert import ir_to_bytes
    import omle as _omle

    labels_list = list(labels)
    labels_entry = _omle.TensorEntry(
        id="le_labels",
        dense=_omle.Tensor(
            name="le_labels",
            type=TensorType(dtype=DataType.STRING, shape=[len(labels_list)]),
            string_data=labels_list,
        ),
    )
    offsets_entry = _omle.TensorEntry(
        id="le_offsets",
        dense=_omle.Tensor(
            name="le_offsets",
            type=TensorType(dtype=DataType.INT64, shape=[2]),
            int64_data=[0, len(labels_list)],
        ),
    )

    node = Node(
        domain="omle.feature",
        op="LabelEncoder",
        inputs=[NodeInput(name=NameRef(input_name))],
        outputs=[NodeOutput(name="y_pred")],
        attributes=[
            Attribute(name="labels",        tensor_ref=_omle.TensorRef(id="le_labels")),
            Attribute(name="label_offsets", tensor_ref=_omle.TensorRef(id="le_offsets")),
        ],
    )

    model = OMLEModel(
        inputs=[InputSpec(name=input_name,
                          type=TensorType(dtype=DataType.STRING, shape=[-1, 1]))],
        outputs=[OutputSpec(name="y_pred")],
        nodes=[node],
        tensor_entries=[labels_entry, offsets_entry],
    )

    return ir_to_bytes(model)


@pytest.fixture(scope="session")
def label_encoder_model_bytes():
    """LabelEncoder model: input 'category' (string) → encoded index."""
    return make_label_encoder_model_bytes("category", ["A", "B", "C"])


@pytest.fixture(scope="session")
def label_encoder_model(label_encoder_model_bytes):
    import omleruntime as omr
    return omr.load_bytes(label_encoder_model_bytes)


def make_mixed_model_bytes(string_input, float_input, labels):
    """Build a 2-input model with one string input (LabelEncoder) and one float input.

    The float input is declared but not wired to any output; the model
    output is solely the encoded label from the string column.
    """
    from omle.ir.model import OMLEModel
    from omle.ir.interface import InputSpec, OutputSpec
    from omle.ir.node import Node, NodeInput, NodeOutput, Attribute
    from omle.ir.types import NameRef, TensorType
    from omle.ir.enums import DataType
    from omle.proto.convert import ir_to_bytes
    import omle as _omle

    labels_list = list(labels)
    labels_entry = _omle.TensorEntry(
        id="le_labels",
        dense=_omle.Tensor(
            name="le_labels",
            type=TensorType(dtype=DataType.STRING, shape=[len(labels_list)]),
            string_data=labels_list,
        ),
    )
    offsets_entry = _omle.TensorEntry(
        id="le_offsets",
        dense=_omle.Tensor(
            name="le_offsets",
            type=TensorType(dtype=DataType.INT64, shape=[2]),
            int64_data=[0, len(labels_list)],
        ),
    )

    node = Node(
        domain="omle.feature",
        op="LabelEncoder",
        inputs=[NodeInput(name=NameRef(string_input))],
        outputs=[NodeOutput(name="y_pred")],
        attributes=[
            Attribute(name="labels",        tensor_ref=_omle.TensorRef(id="le_labels")),
            Attribute(name="label_offsets", tensor_ref=_omle.TensorRef(id="le_offsets")),
        ],
    )

    model = OMLEModel(
        inputs=[
            InputSpec(name=float_input,
                      type=TensorType(dtype=DataType.FLOAT32, shape=[-1, 1])),
            InputSpec(name=string_input,
                      type=TensorType(dtype=DataType.STRING, shape=[-1, 1])),
        ],
        outputs=[OutputSpec(name="y_pred")],
        nodes=[node],
        tensor_entries=[labels_entry, offsets_entry],
    )

    return ir_to_bytes(model)


@pytest.fixture(scope="session")
def mixed_model_bytes():
    """Mixed model: float 'age' + string 'category' → encoded label."""
    return make_mixed_model_bytes("category", "age", ["low", "med", "high"])


@pytest.fixture(scope="session")
def mixed_model(mixed_model_bytes):
    import omleruntime as omr
    return omr.load_bytes(mixed_model_bytes)
