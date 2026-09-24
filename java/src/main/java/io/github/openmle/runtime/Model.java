package io.github.openmle.runtime;

import com.sun.jna.Memory;
import com.sun.jna.NativeLong;
import com.sun.jna.Pointer;
import com.sun.jna.StringArray;
import com.sun.jna.ptr.IntByReference;
import com.sun.jna.ptr.PointerByReference;
import io.github.openmle.runtime.internal.Check;
import io.github.openmle.runtime.internal.Lib;
import io.github.openmle.runtime.internal.LibHolder;

import java.util.Arrays;

/**
 * An immutable, thread-safe model handle.
 *
 * <p>Load via {@link #loadFile} or {@link #loadBytes}, then call
 * {@link #predict} directly (thread-safe, uses the model's thread pool) or
 * create a per-thread {@link Session} via {@link #createSession()}.
 *
 * <p>Must be closed when no longer needed:
 * <pre>{@code
 *   try (Model model = Model.loadFile("model.omle")) {
 *       float[] scores = model.predict(X);
 *   }
 * }</pre>
 */
public final class Model implements AutoCloseable {

    private final Lib lib = LibHolder.LIB;
    private Pointer ptr;

    private final int          numInputs;
    private final int          numOutputs;
    private final InputSpec[]  inputs;
    private final OutputSpec[] outputs;

    private Model(Pointer ptr) {
        this.ptr        = ptr;
        this.numInputs  = lib.omle_model_num_inputs(ptr);
        this.numOutputs = lib.omle_model_num_outputs(ptr);
        this.inputs     = readInputSpecs(ptr, numInputs);
        this.outputs    = readOutputSpecs(ptr, numOutputs);
    }

    // ---------------------------------------------------------------------------
    // Factory methods
    // ---------------------------------------------------------------------------

    /**
     * Load a model from a protobuf binary file.
     *
     * @param path      file-system path to the .omle file
     * @param nThreads  worker threads for {@link #predict} (0 = auto, 1 = default)
     */
    public static Model loadFile(String path, int nThreads) {
        Lib lib = LibHolder.LIB;
        Lib.LoadOptions opts = new Lib.LoadOptions();
        opts.n_threads         = nThreads;
        opts.min_parallel_rows = 64;
        PointerByReference ref = new PointerByReference();
        Check.ok(lib.omle_model_load_file(path, opts, ref));
        return new Model(ref.getValue());
    }

    /** Load a model from a file, single-threaded. */
    public static Model loadFile(String path) {
        return loadFile(path, 1);
    }

    /**
     * Load a model from a byte array (e.g. fetched from a database).
     *
     * @param data     serialised protobuf bytes
     * @param nThreads worker threads for {@link #predict}
     */
    public static Model loadBytes(byte[] data, int nThreads) {
        if (data == null || data.length == 0)
            throw new IllegalArgumentException("data must be non-empty");
        Lib lib = LibHolder.LIB;
        Lib.LoadOptions opts = new Lib.LoadOptions();
        opts.n_threads         = nThreads;
        opts.min_parallel_rows = 64;
        PointerByReference ref = new PointerByReference();
        Check.ok(lib.omle_model_load_memory(data, new NativeLong(data.length), opts, ref));
        return new Model(ref.getValue());
    }

    /** Load a model from bytes, single-threaded. */
    public static Model loadBytes(byte[] data) {
        return loadBytes(data, 1);
    }

    // ---------------------------------------------------------------------------
    // Schema
    // ---------------------------------------------------------------------------

    public int numInputs()  { return inputs.length; }
    public int numOutputs() { return outputs.length; }

    public InputSpec  inputSpec(int idx)  { return inputs[idx]; }
    public OutputSpec outputSpec(int idx) { return outputs[idx]; }

    public InputSpec[]  inputs()  { return inputs.clone(); }
    public OutputSpec[] outputs() { return outputs.clone(); }

    // ---------------------------------------------------------------------------
    // Inference
    // ---------------------------------------------------------------------------

    /**
     * Run batch inference (thread-safe).
     *
     * <p>Sends the entire matrix as a single wide tensor; the runtime splits
     * columns into individual named input slots automatically.
     *
     * @param X     row-major matrix, shape [nSamples][nFeatures]
     * @return      flat row-major output, length = nSamples * numOutputs
     */
    public float[] predict(float[][] X) {
        int nRows = X.length;
        int nCols = X[0].length;
        return predict(Tensor.flatten(X, nRows, nCols), nRows, nCols);
    }

    /**
     * Run batch inference from a pre-flattened array (avoids an extra copy).
     *
     * @param X     flat row-major values, length = nRows * nCols
     * @param nRows number of samples
     * @param nCols number of features
     * @return      flat row-major output, length = nRows * numOutputs
     */
    public float[] predict(float[] X, int nRows, int nCols) {
        try (Tensor input = Tensor.ofFlat(X, nRows, nCols)) {
            return runPredict(input);
        }
    }

    /**
     * Run batch inference from per-column arrays (supports STRING and float columns).
     *
     * @param colNames column names matching the model's declared input names
     * @param colTypes per-column type: {@link io.github.openmle.runtime.internal.Lib#COL_FLOAT32} or
     *                 {@link io.github.openmle.runtime.internal.Lib#COL_STRING}
     * @param cols     per-column data: {@code float[]} for float columns,
     *                 {@code String[]} for string columns
     * @param nRows    number of rows
     * @return flat row-major output, length = nRows * numOutputs
     */
    /**
     * Column type codes for {@link #predictColumns}.
     *
     * <p>These mirror the {@code OMLE_COL_*} constants in {@code c_api.h} and
     * are exposed here so callers need not reach into the internal package —
     * or, worse, hard-code the numbers. {@code COL_STRING} was 1 until FLOAT64
     * support was added, which took that value and moved STRING to 2. Nothing
     * validates a stale code: a string column sent as 1 has its {@code char*}
     * array read as {@code double*}, so every row decodes to category index 0
     * and the model predicts as though each row held the first category, with
     * no error raised.
     */
    public static final int COL_FLOAT32 = Lib.COL_FLOAT32;
    public static final int COL_FLOAT64 = Lib.COL_FLOAT64;
    public static final int COL_STRING  = Lib.COL_STRING;

    public float[] predictColumns(String[] colNames, int[] colTypes, Object[] cols, int nRows) {
        int n = colNames.length;
        Pointer[]     colData = new Pointer[n];
        Memory[]      mems    = new Memory[n];
        StringArray[] sas     = new StringArray[n];
        for (int i = 0; i < n; i++) {
            if (colTypes[i] == Lib.COL_FLOAT32) {
                float[] fa = (float[]) cols[i];
                Memory m = new Memory((long) nRows * Float.BYTES);
                m.write(0, fa, 0, nRows);
                mems[i]    = m;
                colData[i] = m;
            } else if (colTypes[i] == Lib.COL_FLOAT64) {
                double[] da = (double[]) cols[i];
                Memory m = new Memory((long) nRows * Double.BYTES);
                m.write(0, da, 0, nRows);
                mems[i]    = m;
                colData[i] = m;
            } else if (colTypes[i] == Lib.COL_STRING) {
                StringArray sa = new StringArray((String[]) cols[i]);
                sas[i]     = sa;
                colData[i] = sa;
            } else {
                // Not an else-fallthrough any more. While COL_STRING was the
                // only non-float type, "anything else is a string" was safe;
                // adding COL_FLOAT64 made that assumption silently wrong for
                // a caller still using the old numbering, so an unknown code
                // now says so instead of casting a double[] to String[].
                throw new IllegalArgumentException(
                    "unknown column type " + colTypes[i] + " for column '"
                    + colNames[i] + "'; expected COL_FLOAT32 ("
                    + Lib.COL_FLOAT32 + "), COL_FLOAT64 (" + Lib.COL_FLOAT64
                    + ") or COL_STRING (" + Lib.COL_STRING + ")");
            }
        }

        IntByReference     nOut     = new IntByReference();
        PointerByReference outNames = new PointerByReference();
        PointerByReference outTens  = new PointerByReference();
        Check.ok(lib.omle_model_predict_columns(
            ptr, nRows, n, colNames, colTypes, colData, 0, null, nOut, outNames, outTens));

        // Prevent GC of native buffers before the call returns.
        java.lang.ref.Reference.reachabilityFence(mems);
        java.lang.ref.Reference.reachabilityFence(sas);

        int count = nOut.getValue();
        try {
            Pointer tensArray = outTens.getValue();
            Pointer first     = tensArray.getPointerArray(0, count)[0];
            int rows    = lib.omle_tensor_n_rows(first);
            int outCols = lib.omle_tensor_n_cols(first);
            int dtype   = lib.omle_tensor_dtype(first);
            int numel   = rows * outCols;
            if (dtype == 12) {
                double[] d = lib.omle_tensor_data(first).getDoubleArray(0, numel);
                float[]  f = new float[numel];
                for (int i = 0; i < numel; i++) f[i] = (float) d[i];
                return f;
            }
            return lib.omle_tensor_data(first).getFloatArray(0, numel);
        } finally {
            lib.omle_free_strings(outNames.getValue(), count);
            lib.omle_free_tensors(outTens.getValue(),  count);
        }
    }

    /** Create a per-thread {@link Session} backed by this model. */
    public Session createSession() {
        PointerByReference ref = new PointerByReference();
        Check.ok(lib.omle_session_create(ptr, ref));
        return new Session(ref.getValue(), this);
    }

    // ---------------------------------------------------------------------------
    // AutoCloseable
    // ---------------------------------------------------------------------------

    @Override
    public void close() {
        if (ptr != null) {
            lib.omle_free_model(ptr);
            ptr = null;
        }
    }

    @Override
    public String toString() {
        return "Model{numInputs=" + numInputs + ", numOutputs=" + numOutputs + '}';
    }

    // ---------------------------------------------------------------------------
    // Internal helpers
    // ---------------------------------------------------------------------------

    /** Pointer exposed to Session — null-safe because we check ptr above. */
    Pointer ptr() { return ptr; }

    float[] runPredict(Tensor input) {
        int n = inputs.length;
        if (n == 1) {
            // Single input: pass the tensor as-is under the declared name.
            return callPredict(new String[] { inputs[0].name() },
                               new Pointer[] { input.ptr() });
        }

        // Multiple inputs: split the flat tensor column-wise by declared input widths.
        int nRows  = input.nRows();
        int nTotal = input.nCols();
        float[] flat = input.toFlatArray();

        int[] widths = new int[n];
        for (int i = 0; i < n; i++) {
            long[] s = inputs[i].shape();
            widths[i] = (s.length == 1) ? 1 : (int) s[1];
        }

        String[]  names   = new String[n];
        Pointer[] ptrs    = new Pointer[n];
        Tensor[]  slices  = new Tensor[n];
        try {
            int offset = 0;
            for (int i = 0; i < n; i++) {
                names[i] = inputs[i].name();
                int w = widths[i];
                float[] slice = new float[nRows * w];
                for (int r = 0; r < nRows; r++)
                    System.arraycopy(flat, r * nTotal + offset, slice, r * w, w);
                slices[i] = Tensor.ofFlat(slice, nRows, w);
                ptrs[i]   = slices[i].ptr();
                offset   += w;
            }
            return callPredict(names, ptrs);
        } finally {
            for (Tensor t : slices) if (t != null) t.close();
        }
    }

    private float[] callPredict(String[] names, Pointer[] tensors) {
        IntByReference     nOut      = new IntByReference();
        PointerByReference outNames  = new PointerByReference();
        PointerByReference outTens   = new PointerByReference();

        Check.ok(lib.omle_model_predict(
            ptr, names.length, names, tensors, 0, null, nOut, outNames, outTens));

        int count = nOut.getValue();
        try {
            Pointer tensArray = outTens.getValue();
            Pointer first     = tensArray.getPointerArray(0, count)[0];
            int rows  = lib.omle_tensor_n_rows(first);
            int cols  = lib.omle_tensor_n_cols(first);
            int dtype = lib.omle_tensor_dtype(first);
            int numel = rows * cols;
            // OMLE_DTYPE_FLOAT64 == 12; downcast to float32 for the Java caller.
            if (dtype == 12) {
                double[] d = lib.omle_tensor_data(first).getDoubleArray(0, numel);
                float[]  f = new float[numel];
                for (int i = 0; i < numel; i++) f[i] = (float) d[i];
                return f;
            }
            return lib.omle_tensor_data(first).getFloatArray(0, numel);
        } finally {
            lib.omle_free_strings(outNames.getValue(), count);
            lib.omle_free_tensors(outTens.getValue(),  count);
        }
    }

    private InputSpec[] readInputSpecs(Pointer model, int n) {
        // The C library may over-report n for matrix inputs (it counts columns).
        // Stop at the first error, mirroring Python's behaviour.
        java.util.List<InputSpec> list = new java.util.ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            Lib.TensorSpec s = new Lib.TensorSpec();
            if (lib.omle_model_input_spec(model, i, s) != 0) break;
            list.add(new InputSpec(s.name.getString(0), DataType.fromCode(s.dtype), readShape(s)));
        }
        return list.toArray(new InputSpec[0]);
    }

    private OutputSpec[] readOutputSpecs(Pointer model, int n) {
        java.util.List<OutputSpec> list = new java.util.ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            Lib.TensorSpec s = new Lib.TensorSpec();
            if (lib.omle_model_output_spec(model, i, s) != 0) break;
            list.add(new OutputSpec(s.name.getString(0), DataType.fromCode(s.dtype), readShape(s)));
        }
        return list.toArray(new OutputSpec[0]);
    }

    private static long[] readShape(Lib.TensorSpec s) {
        long[] shape = new long[s.shape_rank];
        for (int i = 0; i < s.shape_rank; i++)
            shape[i] = s.shape.getLong((long) i * Long.BYTES);
        return shape;
    }
}
