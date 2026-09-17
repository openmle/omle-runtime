"""Generate protobuf test fixture models for the Spark / Java integration tests.

Run from the repo root (omle-runtime/):
    python spark/scripts/gen_test_fixtures.py

Writes:
    spark/src/test/resources/test_model_3class.omle
    java/src/test/resources/test_model_2f.omle
    spark/src/test/resources/test_model_2f.omle

test_model_3class.omle — 3-class TreeEnsemble (2 features, tree_group=[0,1,2], SOFTMAX):
    feat0 < 0.5  →  scores=[+2, 0, -1]  →  softmax  →  class 0 wins
    feat0 >= 0.5 →  scores=[-1, 0, +2]  →  softmax  →  class 2 wins
    Expected probabilities (softmax of [2,0,-1]):
        p = exp([2,0,-1]) / (e^2 + 1 + e^-1)
          ≈ [0.8438, 0.1142, 0.0420]  (feat0 < 0.5)
          ≈ [0.0420, 0.1142, 0.8438]  (feat0 >= 0.5)

test_model_2f.omle — 2-feature single-input (X, shape [-1,2]) TreeEnsemble stump (SUM, IDENTITY):
    feat0 < 0.5  → score = +2.0
    feat0 >= 0.5 → score = -2.0
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Use the omle Python IR from the sibling repo for all fixture generation.
# This ensures the proto uses the current NameRef / Tensor sub-message format.
_OMLE_SRC = REPO_ROOT.parent / "omle" / "src"
if _OMLE_SRC.exists():
    sys.path.insert(0, str(_OMLE_SRC))


def _stump(feat: int, threshold: float, left_val: float, right_val: float):
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
            Tensor(float32_data=[0.0, left_val, right_val])),
    )


def _const_tree(value: float):
    from omle.ir.bodies import Tree
    from omle.ir.tensor import Tensor
    from omle.ir.types import TensorValue
    from omle.ir.enums import TreeNodeKind, TreeSplitOp
    return Tree(
        num_nodes=1,
        node_kind=[TreeNodeKind.LEAF],
        split_feature=[0],
        split_threshold=TensorValue.of_tensor(Tensor(float32_data=[0.0])),
        split_op=[TreeSplitOp.SPLIT_OP_UNSPECIFIED],
        children_offset=[0],
        children_count=[0],
        default_child=[0],
        leaf_value=TensorValue.of_tensor(Tensor(float32_data=[value])),
    )


def make_3class_model_bytes() -> bytes:
    """
    3-class TreeEnsemble with a single matrix input X (shape [-1, 2]).
    tree_group = [0, 1, 2] → SUM per class → SOFTMAX.

    Class trees:
      class 0: feat0 < 0.5 → +2.0, else -1.0
      class 1: constant  0.0
      class 2: feat0 < 0.5 → -1.0, else +2.0
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
        post_transform=PostTransform.SOFTMAX,
        tree_group=[0, 1, 2],
        trees=[
            _stump(0, 0.5,  2.0, -1.0),
            _const_tree(0.0),
            _stump(0, 0.5, -1.0,  2.0),
        ],
    )

    node = Node(
        domain="omle.ml",
        op="TreeEnsemble",
        inputs=[NodeInput(name=NameRef("X"))],
        outputs=[NodeOutput(name="y_prob")],
        tree_ensemble=te,
    )

    model = OMLEModel(
        inputs=[InputSpec(name="X", type=TensorType(dtype=DataType.FLOAT32, shape=[-1, 2]))],
        outputs=[OutputSpec(name="y_prob")],
        nodes=[node],
    )

    return ir_to_bytes(model)


def make_2f_model_bytes() -> bytes:
    """
    2-feature single-matrix-input TreeEnsemble stump.
    Input: X (shape [-1, 2]).  Output: score (SUM, IDENTITY).

    Stump: feat0 < 0.5 → score = +2.0, else score = -2.0
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
        trees=[_stump(0, 0.5, 2.0, -2.0)],
    )

    node = Node(
        domain="omle.ml",
        op="TreeEnsemble",
        inputs=[NodeInput(name=NameRef("X"))],
        outputs=[NodeOutput(name="score")],
        tree_ensemble=te,
    )

    model = OMLEModel(
        inputs=[InputSpec(name="X", type=TensorType(dtype=DataType.FLOAT32, shape=[-1, 2]))],
        outputs=[OutputSpec(name="score")],
        nodes=[node],
    )

    return ir_to_bytes(model)


if __name__ == "__main__":
    spark_resources = REPO_ROOT / "spark" / "src" / "test" / "resources"
    java_resources  = REPO_ROOT / "java"  / "src" / "test" / "resources"
    spark_resources.mkdir(parents=True, exist_ok=True)
    java_resources.mkdir(parents=True, exist_ok=True)

    data3 = make_3class_model_bytes()
    out3 = spark_resources / "test_model_3class.omle"
    out3.write_bytes(data3)
    print(f"Wrote {len(data3)} bytes to {out3}")

    data2f = make_2f_model_bytes()
    for out_dir in (spark_resources, java_resources):
        out_path = out_dir / "test_model_2f.omle"
        out_path.write_bytes(data2f)
        print(f"Wrote {len(data2f)} bytes to {out_path}")
