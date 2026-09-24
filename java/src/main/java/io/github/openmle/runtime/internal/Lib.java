package io.github.openmle.runtime.internal;

import com.sun.jna.*;
import com.sun.jna.ptr.*;

/**
 * JNA direct mapping of the omle C API.
 * Package-private — use the public API classes instead.
 */
public interface Lib extends Library {

    // ---------------------------------------------------------------------------
    // Structs
    // ---------------------------------------------------------------------------

    @Structure.FieldOrder({"n_threads", "min_parallel_rows"})
    final class LoadOptions extends Structure {
        public int n_threads         = 1;
        public int min_parallel_rows = 64;
    }

    @Structure.FieldOrder({"name", "dtype", "shape", "shape_rank"})
    final class TensorSpec extends Structure {
        public Pointer name;        // const char* — borrowed from model
        public int     dtype;
        public Pointer shape;       // const int64_t* — borrowed from model
        public int     shape_rank;
    }

    // ---------------------------------------------------------------------------
    // Error
    // ---------------------------------------------------------------------------

    String omle_last_error();

    // ---------------------------------------------------------------------------
    // Model lifecycle
    // ---------------------------------------------------------------------------

    int  omle_model_load_file(String path, LoadOptions opts, PointerByReference outModel);
    int  omle_model_load_memory(byte[] data, NativeLong size, LoadOptions opts, PointerByReference outModel);
    void omle_free_model(Pointer model);

    // ---------------------------------------------------------------------------
    // Model introspection
    // ---------------------------------------------------------------------------

    int omle_model_num_inputs(Pointer model);
    int omle_model_num_outputs(Pointer model);
    int omle_model_input_spec(Pointer model, int idx, TensorSpec spec);
    int omle_model_output_spec(Pointer model, int idx, TensorSpec spec);

    // ---------------------------------------------------------------------------
    // Stateless prediction
    //
    // out_names / out_tensors are caller-allocated arrays returned by the C
    // library; caller must free with omle_free_strings / omle_free_tensors.
    // ---------------------------------------------------------------------------

    int omle_model_predict(
        Pointer model,
        int nInputs,
        String[] inputNames,
        Pointer[] inputTensors,
        int nReqOutputs,
        String[] reqOutputNames,
        IntByReference nOut,
        PointerByReference outNames,
        PointerByReference outTensors
    );

    // ---------------------------------------------------------------------------
    // Session
    // ---------------------------------------------------------------------------

    int  omle_session_create(Pointer model, PointerByReference outSession);
    void omle_free_session(Pointer session);
    int  omle_session_bind_input(Pointer session, String name, Pointer tensor);
    void omle_session_clear_inputs(Pointer session);
    int  omle_session_run(Pointer session);
    int  omle_session_get_output(Pointer session, String name, PointerByReference outTensor);
    int  omle_session_num_inputs(Pointer session);
    int  omle_session_num_outputs(Pointer session);

    // ---------------------------------------------------------------------------
    // Tensor construction
    // ---------------------------------------------------------------------------

    Pointer omle_tensor_create_f32(int nRows, int nCols, float[] data);
    void omle_free_tensor(Pointer tensor);

    // ---------------------------------------------------------------------------
    // Tensor accessors
    // ---------------------------------------------------------------------------

    int     omle_tensor_dtype(Pointer tensor);
    int     omle_tensor_n_rows(Pointer tensor);
    int     omle_tensor_n_cols(Pointer tensor);
    Pointer omle_tensor_data(Pointer tensor);
    double  omle_tensor_get(Pointer tensor, int row, int col);

    // ---------------------------------------------------------------------------
    // Cleanup for omle_model_predict outputs
    // ---------------------------------------------------------------------------

    void omle_free_strings(Pointer names, int count);
    void omle_free_tensors(Pointer tensors, int count);

    // ---------------------------------------------------------------------------
    // Bulk column prediction (supports STRING and float columns)
    //
    // col_types[c]: COL_FLOAT32 (0), COL_FLOAT64 (1) or COL_STRING (2).
    // col_data[c]:  for FLOAT32, pointer to float[n_rows];
    //               for FLOAT64, pointer to double[n_rows];
    //               for STRING,  pointer to const char*[n_rows].
    //
    // These mirror OMLE_COL_* in include/omle/c_api.h and must be kept in step
    // with it. COL_STRING was 1 until FLOAT64 support was added, which took
    // that value and pushed STRING to 2. Nothing detects a mismatch: a caller
    // still sending 1 for a string column has its char* array read as
    // double*, so the tensor fills with garbage, a LabelEncoder over it
    // returns index 0 for every row, and the model silently predicts as
    // though every row held the first category. Pass these constants rather
    // than integer literals.
    // ---------------------------------------------------------------------------

    int COL_FLOAT32 = 0;
    int COL_FLOAT64 = 1;
    int COL_STRING  = 2;

    int omle_model_predict_columns(
        Pointer model,
        int nRows,
        int nCols,
        String[] colNames,
        int[] colTypes,
        Pointer[] colData,
        int nFilter,
        String[] filterNames,
        IntByReference nOut,
        PointerByReference outNames,
        PointerByReference outTensors
    );
}
