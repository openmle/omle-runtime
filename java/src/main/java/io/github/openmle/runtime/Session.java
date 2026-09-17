package io.github.openmle.runtime;

import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;
import io.github.openmle.runtime.internal.Check;
import io.github.openmle.runtime.internal.Lib;
import io.github.openmle.runtime.internal.LibHolder;

/**
 * Per-thread execution context — <strong>not</strong> thread-safe.
 *
 * <p>Reuses internal buffers across calls, so creating one session per thread
 * and reusing it is more efficient than calling {@link Model#predict} in a
 * tight loop.
 *
 * <p>Obtain via {@link Model#createSession()}:
 * <pre>{@code
 *   try (Session session = model.createSession()) {
 *       for (float[][] batch : batches) {
 *           float[] scores = session.predict(batch);
 *       }
 *   }
 * }</pre>
 *
 * <p>Tensors returned by {@link #run} are borrowed — they are invalidated by
 * the next {@link #run} call and must not be closed by the caller.
 */
public final class Session implements AutoCloseable {

    private final Lib   lib = LibHolder.LIB;
    private final Model model;
    private Pointer ptr;

    Session(Pointer ptr, Model model) {
        this.ptr   = ptr;
        this.model = model;
    }

    // ---------------------------------------------------------------------------
    // Inference
    // ---------------------------------------------------------------------------

    /**
     * Run inference on a 2-D row-major matrix.
     *
     * @param X     shape [nSamples][nFeatures]
     * @return      flat row-major output, length = nSamples * model.numOutputs()
     */
    public float[] predict(float[][] X) {
        int nRows = X.length;
        int nCols = X[0].length;
        return predict(Tensor.flatten(X, nRows, nCols), nRows, nCols);
    }

    /**
     * Run inference from a pre-flattened array.
     *
     * @param X     flat row-major float array
     * @param nRows number of samples
     * @param nCols number of features
     * @return      flat row-major output, length = nRows * model.numOutputs()
     */
    public float[] predict(float[] X, int nRows, int nCols) {
        try (Tensor input = Tensor.ofFlat(X, nRows, nCols)) {
            return runSession(input);
        }
    }

    /**
     * Run inference and return a {@link Tensor}.
     *
     * <p>The returned Tensor is <em>borrowed</em> — it is owned by this session
     * and valid only until the next call to {@code run}.  Do not call
     * {@link Tensor#close()} on it.
     *
     * @param X shape [nSamples][nFeatures]
     */
    public Tensor run(float[][] X) {
        int nRows = X.length;
        int nCols = X[0].length;
        try (Tensor input = Tensor.ofFlat(Tensor.flatten(X, nRows, nCols), nRows, nCols)) {
            return runSessionTensor(input);
        }
    }

    public int numOutputs() {
        return lib.omle_session_num_outputs(ptr);
    }

    // ---------------------------------------------------------------------------
    // AutoCloseable
    // ---------------------------------------------------------------------------

    @Override
    public void close() {
        if (ptr != null) {
            lib.omle_free_session(ptr);
            ptr = null;
        }
    }

    @Override
    public String toString() {
        return "Session{numOutputs=" + numOutputs() + '}';
    }

    // ---------------------------------------------------------------------------
    // Internal
    // ---------------------------------------------------------------------------

    private float[] runSession(Tensor input) {
        lib.omle_session_clear_inputs(ptr);
        Check.ok(lib.omle_session_bind_input(ptr, model.inputSpec(0).name(), input.ptr()));
        Check.ok(lib.omle_session_run(ptr));

        String outputName = model.outputSpec(0).name();
        PointerByReference ref = new PointerByReference();
        Check.ok(lib.omle_session_get_output(ptr, outputName, ref));
        Pointer outPtr = ref.getValue();
        int rows = lib.omle_tensor_n_rows(outPtr);
        int cols = lib.omle_tensor_n_cols(outPtr);
        return lib.omle_tensor_data(outPtr).getFloatArray(0, rows * cols);
    }

    private Tensor runSessionTensor(Tensor input) {
        lib.omle_session_clear_inputs(ptr);
        Check.ok(lib.omle_session_bind_input(ptr, model.inputSpec(0).name(), input.ptr()));
        Check.ok(lib.omle_session_run(ptr));

        String outputName = model.outputSpec(0).name();
        PointerByReference ref = new PointerByReference();
        Check.ok(lib.omle_session_get_output(ptr, outputName, ref));
        // Borrowed — the session owns this handle.
        return new Tensor(ref.getValue(), false);
    }
}
