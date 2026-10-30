#!/usr/bin/env python3
"""Generate synthetic model .omle files for benchmarking."""

import sys, os, random, struct, math, argparse

sys.path.insert(0, "/tmp/omle_proto_py")
import omle_pb2 as P

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def float32_bytes(values):
    return struct.pack(f"{len(values)}f", *values)

def make_tensor(name, data, shape):
    t = P.Tensor()
    t.name = name
    t.type.dtype = P.FLOAT32
    for d in shape:
        t.type.shape.append(d)
    t.raw_data = float32_bytes(data)
    return t

def make_input(name):
    inp = P.InputSpec()
    inp.name = name
    inp.type.dtype = P.FLOAT32
    inp.type.shape.extend([-1, 1])
    return inp

def make_output(name, cols=1):
    out = P.OutputSpec()
    out.name = name
    out.role = P.PREDICTION
    out.type.dtype = P.FLOAT32
    out.type.shape.extend([-1, cols])
    return out

# ---------------------------------------------------------------------------
# Tree ensemble (binary stumps, depth-1)
# ---------------------------------------------------------------------------

def make_stump(n_features, rng):
    """Return a depth-1 tree that splits on a random feature."""
    t = P.Tree()
    t.num_nodes = 3
    feat = rng.randrange(n_features)
    thresh = rng.uniform(0.2, 0.8)
    left_val  = rng.uniform(-1, 1)
    right_val = rng.uniform(-1, 1)

    t.node_kind.extend([P.Tree.BRANCH, P.Tree.LEAF, P.Tree.LEAF])
    t.split_feature.extend([feat, 0, 0])
    t.split_threshold.extend([thresh, 0.0, 0.0])
    t.split_op.extend([P.Tree.LESS_THAN, P.Tree.LESS_THAN, P.Tree.LESS_THAN])
    # children: node 0 → [1, 2]
    t.children_offset.extend([0, -1, -1])
    t.children_count.extend([2, 0, 0])
    t.children_index.extend([1, 2])
    t.default_child.extend([2, -1, -1])
    t.leaf_value.extend([0.0, left_val, right_val])
    t.leaf_vector_size = 1
    return t


def gen_tree_ensemble(n_features, n_trees, outpath, rng):
    m = P.OMLEModel()
    m.metadata.name = f"bench_trees_{n_trees}x{n_features}"
    m.metadata.format_version = "1.0.0"

    for i in range(n_features):
        m.inputs.append(make_input(f"f{i}"))
    m.outputs.append(make_output("score"))

    node = m.nodes.add()
    node.name = "ensemble"
    for i in range(n_features):
        ni = node.inputs.add(); ni.name = f"f{i}"
    no = node.outputs.add(); no.name = "score"

    te = node.tree_ensemble
    for _ in range(n_trees):
        te.trees.append(make_stump(n_features, rng))
    te.aggregation = P.TreeEnsemble.SUM
    te.post_transform = P.IDENTITY

    with open(outpath, "wb") as f:
        f.write(m.SerializeToString())
    print(f"  wrote {outpath}  ({os.path.getsize(outpath)//1024} KB)")


# ---------------------------------------------------------------------------
# Linear model
# ---------------------------------------------------------------------------

def gen_linear(n_features, n_outputs, outpath, rng):
    m = P.OMLEModel()
    m.metadata.name = f"bench_linear_{n_features}x{n_outputs}"
    m.metadata.format_version = "1.0.0"

    for i in range(n_features):
        m.inputs.append(make_input(f"f{i}"))
    for j in range(n_outputs):
        m.outputs.append(make_output(f"y{j}"))

    # coefficients: [n_outputs, n_features]
    coef = [rng.gauss(0, 0.1) for _ in range(n_outputs * n_features)]
    bias = [rng.gauss(0, 0.1) for _ in range(n_outputs)]

    m.constants.append(make_tensor("W", coef, [n_outputs, n_features]))
    m.constants.append(make_tensor("b", bias, [n_outputs]))

    node = m.nodes.add()
    node.name = "linear"
    for i in range(n_features):
        ni = node.inputs.add(); ni.name = f"f{i}"
    for j in range(n_outputs):
        no = node.outputs.add(); no.name = f"y{j}"

    lin = node.linear
    lin.coefficients.name = "W"
    lin.intercept.name    = "b"
    lin.post_transform    = P.IDENTITY

    with open(outpath, "wb") as f:
        f.write(m.SerializeToString())
    print(f"  wrote {outpath}  ({os.path.getsize(outpath)//1024} KB)")


# ---------------------------------------------------------------------------
# Scorecard (one characteristic per feature)
# ---------------------------------------------------------------------------

def gen_scorecard(n_features, outpath, rng):
    m = P.OMLEModel()
    m.metadata.name = f"bench_scorecard_{n_features}f"
    m.metadata.format_version = "1.0.0"

    for i in range(n_features):
        m.inputs.append(make_input(f"f{i}"))
    m.outputs.append(make_output("score"))

    node = m.nodes.add()
    node.name = "scorecard"
    for i in range(n_features):
        ni = node.inputs.add(); ni.name = f"f{i}"
    no = node.outputs.add(); no.name = "score"

    sc = node.scorecard
    sc.baseline_score = 0.0

    for i in range(n_features):
        ch = sc.characteristics.add()
        ch.name = f"f{i}"
        # Three buckets: < 0.33 → pts[0], < 0.66 → pts[1], else → pts[2]
        pts = [rng.uniform(-20, 20) for _ in range(3)]
        thresholds = [0.33, 0.66]
        for k, thr in enumerate(thresholds):
            attr = ch.attributes.add()
            attr.partial_score = pts[k]
            pred = attr.predicate
            sp = pred.simple
            sp.column = f"f{i}"
            sp.op = P.SimplePredicate.LESS_THAN
            sp.value.double_value = thr
        # else (true predicate)
        attr = ch.attributes.add()
        attr.partial_score = pts[2]
        pred = attr.predicate
        pred.true_predicate.CopyFrom(P.TruePredicate())

    with open(outpath, "wb") as f:
        f.write(m.SerializeToString())
    print(f"  wrote {outpath}  ({os.path.getsize(outpath)//1024} KB)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate benchmark model .omle files")
    parser.add_argument("--outdir", default="/tmp/omle_bench", help="Output directory")
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rng = random.Random(args.seed)

    print("Generating tree ensemble models...")
    gen_tree_ensemble(10,   100, f"{args.outdir}/trees_100x10.omle",   rng)
    gen_tree_ensemble(10,   500, f"{args.outdir}/trees_500x10.omle",   rng)
    gen_tree_ensemble(20,  1000, f"{args.outdir}/trees_1000x20.omle",  rng)
    gen_tree_ensemble(50,  5000, f"{args.outdir}/trees_5000x50.omle",  rng)

    print("\nGenerating linear models...")
    gen_linear(100,   1, f"{args.outdir}/linear_100f_1o.omle",   rng)
    gen_linear(100,  10, f"{args.outdir}/linear_100f_10o.omle",  rng)
    gen_linear(1000,  1, f"{args.outdir}/linear_1000f_1o.omle",  rng)

    print("\nGenerating scorecard models...")
    gen_scorecard( 20, f"{args.outdir}/scorecard_20f.omle",  rng)
    gen_scorecard(100, f"{args.outdir}/scorecard_100f.omle", rng)

    print("\nDone.")
