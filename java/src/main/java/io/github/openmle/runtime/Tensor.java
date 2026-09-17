package io.github.openmle.runtime;

import com.sun.jna.Pointer;
import io.github.openmle.runtime.internal.LibHolder;
import io.github.openmle.runtime.OMLEException;

/**
 * A float32 tensor backed by a native C handle.
 *
 * <p>Tensors obtained directly via {@link #of(float[][])} own their handle and
 * must be closed (or used in a try-with-resources block).  Tensors returned by
 * {@link Session#run} are <em>borrowed</em> — they are valid only until the
 * next {@link Session#run} call on the same session and must <strong>not</strong>
 * be closed by the caller.
 */
public final class Tensor implements AutoCloseable {

    // Non-null only for owned handles.
    private Pointer ptr;
    private final boolean owned;

    private final int nRows;
    private final int nCols;

    /** Create a Tensor from a native handle, optionally taking ownership. */
    Tensor(Pointer ptr, boolean owned) {
        this.ptr   = ptr;
        this.owned = owned;
        this.nRows = LibHolder.LIB.omle_tensor_n_rows(ptr);
        this.nCols = LibHolder.LIB.omle_tensor_n_cols(ptr);
    }

    /**
     * Create an owned Tensor from a 2-D float array.
     *
     * @param data row-major matrix, shape [nRows][nCols]
     */
    public static Tensor of(float[][] data) {
        if (data == null || data.length == 0)
            throw new IllegalArgumentException("data must be non-empty");
        int nRows = data.length;
        int nCols = data[0].length;
        float[] flat = flatten(data, nRows, nCols);
        return ofFlat(flat, nRows, nCols);
    }

    /**
     * Create an owned Tensor from a flat row-major float array.
     *
     * @param data  row-major values, length must equal nRows * nCols
     * @param nRows number of rows
     * @param nCols number of columns
     */
    public static Tensor ofFlat(float[] data, int nRows, int nCols) {
        Pointer p = LibHolder.LIB.omle_tensor_create_f32(nRows, nCols, data);
        if (p == null) {
            String msg = LibHolder.LIB.omle_last_error();
            throw new OMLEException(1, msg != null ? msg : "omle_tensor_create_f32 failed");
        }
        return new Tensor(p, true);
    }

    public int nRows() { return nRows; }
    public int nCols() { return nCols; }
    public int numel() { return nRows * nCols; }

    /**
     * Copy the tensor data into a new 2-D float array.
     */
    public float[][] toArray() {
        float[] flat = toFlatArray();
        float[][] result = new float[nRows][nCols];
        for (int r = 0; r < nRows; r++)
            System.arraycopy(flat, r * nCols, result[r], 0, nCols);
        return result;
    }

    /**
     * Copy the tensor data into a flat row-major float array.
     */
    public float[] toFlatArray() {
        ensureOpen();
        Pointer dataPtr = LibHolder.LIB.omle_tensor_data(ptr);
        if (dataPtr == null)
            throw new IllegalStateException("tensor has no dense data (sparse tensors not supported here)");
        return dataPtr.getFloatArray(0, nRows * nCols);
    }

    /** Get a single element by row/column (works for all tensor kinds). */
    public double get(int row, int col) {
        ensureOpen();
        return LibHolder.LIB.omle_tensor_get(ptr, row, col);
    }

    /** The raw native handle — for internal use only. */
    Pointer ptr() {
        ensureOpen();
        return ptr;
    }

    @Override
    public void close() {
        if (owned && ptr != null) {
            LibHolder.LIB.omle_free_tensor(ptr);
            ptr = null;
        }
    }

    @Override
    public String toString() {
        return "Tensor{rows=" + nRows + ", cols=" + nCols + '}';
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    static float[] flatten(float[][] data, int nRows, int nCols) {
        float[] flat = new float[nRows * nCols];
        for (int r = 0; r < nRows; r++) {
            if (data[r].length != nCols)
                throw new IllegalArgumentException(
                    "row " + r + " has " + data[r].length + " columns, expected " + nCols);
            System.arraycopy(data[r], 0, flat, r * nCols, nCols);
        }
        return flat;
    }

    private void ensureOpen() {
        if (ptr == null) throw new IllegalStateException("Tensor has been closed");
    }
}
