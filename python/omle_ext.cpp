#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "omle/c_api.h"

namespace py = pybind11;
using F32Array = py::array_t<float, py::array::c_style | py::array::forcecast>;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

static inline void check(int rc) {
  if (rc != OMLE_OK) throw std::runtime_error(omle_last_error());
}

static py::array read_tensor(const omle_tensor_t* t) {
  int rows = omle_tensor_n_rows(t);
  int cols = omle_tensor_n_cols(t);
  const void* data = omle_tensor_data(t);
  switch (omle_tensor_dtype(t)) {
    case OMLE_DTYPE_FLOAT64:
      return py::array_t<double>({rows, cols},
                                 static_cast<const double*>(data));
    case OMLE_DTYPE_INT32:
      return py::array_t<int32_t>({rows, cols},
                                  static_cast<const int32_t*>(data));
    case OMLE_DTYPE_INT64:
      return py::array_t<int64_t>({rows, cols},
                                  static_cast<const int64_t*>(data));
    default:
      return py::array_t<float>({rows, cols}, static_cast<const float*>(data));
  }
}

static py::dict collect_outputs(int n_out, char** out_names,
                                omle_tensor_t** out_tensors) {
  py::dict result;
  for (int i = 0; i < n_out; ++i)
    result[out_names[i]] = read_tensor(out_tensors[i]);
  omle_free_strings(out_names, n_out);
  omle_free_tensors(out_tensors, n_out);
  return result;
}

// Build col_types + col_data_ptrs from list[ndarray | list[str]].
// str_bufs and f32_arrays are output params that keep data alive for the C
// call.
static void build_col_arrays(const py::list& col_data_py,
                             std::vector<int>& col_types,
                             std::vector<const void*>& col_data_ptrs,
                             std::vector<std::vector<const char*>>& str_bufs,
                             std::vector<F32Array>& f32_arrays) {
  int nc = (int)col_data_py.size();
  col_types.resize(nc);
  col_data_ptrs.resize(nc);
  str_bufs.resize(nc);

  for (int c = 0; c < nc; ++c) {
    py::handle item = col_data_py[c];
    if (py::isinstance<py::array>(item)) {
      py::array arr = item.cast<py::array>();
      if (arr.dtype().kind() == 'O') {
        // numpy object array (strings) — iterate PyObject* directly,
        // avoids the tolist() → Python list allocation path
        int nr = (int)arr.size();
        str_bufs[c].resize(nr);
        auto** raw = reinterpret_cast<PyObject**>(arr.mutable_data());
        for (int r = 0; r < nr; ++r) {
          PyObject* s = raw[r];
          const char* p =
              (s == nullptr || s == Py_None) ? "" : PyUnicode_AsUTF8(s);
          str_bufs[c][r] = p ? p : "";
        }
        col_types[c] = OMLE_COL_STRING;
        col_data_ptrs[c] = str_bufs[c].data();
      } else {
        f32_arrays.push_back(arr.cast<F32Array>());
        col_types[c] = OMLE_COL_FLOAT32;
        col_data_ptrs[c] = f32_arrays.back().data();
      }
    } else {
      // list[str] fallback — use PyUnicode_AsUTF8 for zero-copy string access
      py::list lst = item.cast<py::list>();
      int nr = (int)lst.size();
      str_bufs[c].resize(nr);
      for (int r = 0; r < nr; ++r) {
        PyObject* s = lst[r].ptr();
        const char* p = (s == Py_None) ? "" : PyUnicode_AsUTF8(s);
        str_bufs[c][r] = p ? p : "";
      }
      col_types[c] = OMLE_COL_STRING;
      col_data_ptrs[c] = str_bufs[c].data();
    }
  }
}

static int n_rows_from_col_data(const py::list& col_data_py) {
  if (col_data_py.empty()) return 0;
  py::handle first = col_data_py[0];
  if (py::isinstance<py::array>(first))
    return (int)first.cast<py::array>().shape(0);
  return (int)py::len(first.cast<py::list>());
}

// ---------------------------------------------------------------------------

PYBIND11_MODULE(omle_ext, m) {
  m.doc() = "OMLE pybind11 extension";

  // ------------------------------------------------------------------
  // Errors
  // ------------------------------------------------------------------
  m.def("last_error", []() -> std::string {
    const char* e = omle_last_error();
    return e ? std::string(e) : std::string();
  });

  // ------------------------------------------------------------------
  // Model lifecycle
  // ------------------------------------------------------------------
  m.def(
      "model_load_file",
      [](const char* path, int n_threads, int min_parallel_rows) -> uintptr_t {
        omle_load_options_t opts{n_threads, min_parallel_rows};
        omle_model_t* mdl = nullptr;
        check(omle_model_load_file(path, &opts, &mdl));
        return reinterpret_cast<uintptr_t>(mdl);
      },
      py::arg("path"), py::arg("n_threads") = 1,
      py::arg("min_parallel_rows") = 64);

  m.def(
      "model_load_memory",
      [](py::bytes data_py, int n_threads, int min_parallel_rows) -> uintptr_t {
        const char* buf = nullptr;
        Py_ssize_t sz = 0;
        if (PyBytes_AsStringAndSize(data_py.ptr(), const_cast<char**>(&buf),
                                    &sz) < 0)
          throw py::error_already_set();
        omle_load_options_t opts{n_threads, min_parallel_rows};
        omle_model_t* mdl = nullptr;
        check(
            omle_model_load_memory(buf, static_cast<size_t>(sz), &opts, &mdl));
        return reinterpret_cast<uintptr_t>(mdl);
      },
      py::arg("data"), py::arg("n_threads") = 1,
      py::arg("min_parallel_rows") = 64);

  m.def("free_model", [](uintptr_t ptr) {
    omle_free_model(reinterpret_cast<omle_model_t*>(ptr));
  });

  // ------------------------------------------------------------------
  // Model introspection
  // ------------------------------------------------------------------
  m.def("model_num_inputs", [](uintptr_t ptr) {
    return omle_model_num_inputs(reinterpret_cast<const omle_model_t*>(ptr));
  });

  m.def("model_num_outputs", [](uintptr_t ptr) {
    return omle_model_num_outputs(reinterpret_cast<const omle_model_t*>(ptr));
  });

  m.def("model_input_spec", [](uintptr_t ptr, int idx) -> py::tuple {
    omle_tensor_spec_t spec{};
    check(omle_model_input_spec(reinterpret_cast<const omle_model_t*>(ptr), idx,
                                &spec));
    std::vector<int64_t> shape(spec.shape, spec.shape + spec.shape_rank);
    return py::make_tuple(spec.name ? spec.name : "", (int)spec.dtype, shape);
  });

  m.def("model_output_spec", [](uintptr_t ptr, int idx) -> py::tuple {
    omle_tensor_spec_t spec{};
    check(omle_model_output_spec(reinterpret_cast<const omle_model_t*>(ptr),
                                 idx, &spec));
    std::vector<int64_t> shape(spec.shape, spec.shape + spec.shape_rank);
    return py::make_tuple(spec.name ? spec.name : "", (int)spec.dtype, shape,
                          (int)spec.role);
  });

  m.def("model_task_type", [](uintptr_t ptr) -> int {
    return (int)omle_model_task_type(
        reinterpret_cast<const omle_model_t*>(ptr));
  });

  // ------------------------------------------------------------------
  // Model predict
  // ------------------------------------------------------------------

  // dict[name → float32 ndarray] → dict[name → ndarray]
  m.def(
      "model_predict",
      [](uintptr_t model_ptr, py::dict inputs_py,
         py::list filter_names_py) -> py::dict {
        int n = (int)inputs_py.size();
        std::vector<std::string> name_strs;
        name_strs.reserve(n);
        std::vector<const char*> names;
        names.reserve(n);
        std::vector<omle_tensor_t*> tensors;
        tensors.reserve(n);
        std::vector<F32Array> f32_arrs;
        f32_arrs.reserve(n);

        for (auto [key, val] : inputs_py) {
          name_strs.push_back(key.cast<std::string>());
          names.push_back(name_strs.back().c_str());
          f32_arrs.push_back(val.cast<F32Array>());
          auto req = f32_arrs.back().request();
          int nr = (req.ndim >= 1) ? (int)req.shape[0] : 1;
          int nc = (req.ndim >= 2) ? (int)req.shape[1] : 1;
          auto* t = omle_tensor_create_f32(nr, nc,
                                           static_cast<const float*>(req.ptr));
          if (!t) {
            for (auto* tx : tensors) omle_free_tensor(tx);
            throw std::runtime_error(omle_last_error());
          }
          tensors.push_back(t);
        }

        int n_filter = (int)filter_names_py.size();
        std::vector<std::string> fstrs(n_filter);
        std::vector<const char*> fptrs(n_filter);
        for (int i = 0; i < n_filter; ++i) {
          fstrs[i] = filter_names_py[i].cast<std::string>();
          fptrs[i] = fstrs[i].c_str();
        }

        int n_out = 0;
        char** out_names = nullptr;
        omle_tensor_t** out_tens = nullptr;
        int rc = omle_model_predict(
            reinterpret_cast<const omle_model_t*>(model_ptr), n, names.data(),
            (omle_tensor_t* const*)tensors.data(), n_filter,
            n_filter > 0 ? fptrs.data() : nullptr, &n_out, &out_names,
            &out_tens);
        for (auto* t : tensors) omle_free_tensor(t);
        check(rc);
        return collect_outputs(n_out, out_names, out_tens);
      },
      py::arg("model_ptr"), py::arg("inputs"), py::arg("filter_names"));

  // list[str] col_names × list[ndarray|list[str]] col_data → dict[name →
  // ndarray]
  m.def(
      "model_predict_columns",
      [](uintptr_t model_ptr, py::list col_names_py, py::list col_data_py,
         py::list filter_names_py) -> py::dict {
        int n_cols = (int)col_names_py.size();
        std::vector<std::string> cstrs(n_cols);
        std::vector<const char*> cnames(n_cols);
        for (int c = 0; c < n_cols; ++c) {
          cstrs[c] = col_names_py[c].cast<std::string>();
          cnames[c] = cstrs[c].c_str();
        }
        std::vector<int> col_types;
        std::vector<const void*> col_data;
        std::vector<std::vector<const char*>> str_bufs;
        std::vector<F32Array> f32_arrs;
        build_col_arrays(col_data_py, col_types, col_data, str_bufs, f32_arrs);

        int n_rows = n_rows_from_col_data(col_data_py);
        int n_filter = (int)filter_names_py.size();
        std::vector<std::string> fstrs(n_filter);
        std::vector<const char*> fptrs(n_filter);
        for (int i = 0; i < n_filter; ++i) {
          fstrs[i] = filter_names_py[i].cast<std::string>();
          fptrs[i] = fstrs[i].c_str();
        }
        int n_out = 0;
        char** out_names = nullptr;
        omle_tensor_t** out_tens = nullptr;
        check(omle_model_predict_columns(
            reinterpret_cast<const omle_model_t*>(model_ptr), n_rows, n_cols,
            cnames.data(), col_types.data(), col_data.data(), n_filter,
            n_filter > 0 ? fptrs.data() : nullptr, &n_out, &out_names,
            &out_tens));
        return collect_outputs(n_out, out_names, out_tens);
      },
      py::arg("model_ptr"), py::arg("col_names"), py::arg("col_data"),
      py::arg("filter_names"));

  // ------------------------------------------------------------------
  // Session lifecycle
  // ------------------------------------------------------------------
  m.def("session_create", [](uintptr_t model_ptr) -> uintptr_t {
    omle_session_t* s = nullptr;
    check(omle_session_create(reinterpret_cast<const omle_model_t*>(model_ptr),
                              &s));
    return reinterpret_cast<uintptr_t>(s);
  });

  m.def("free_session", [](uintptr_t ptr) {
    omle_free_session(reinterpret_cast<omle_session_t*>(ptr));
  });

  m.def("session_clear_inputs", [](uintptr_t ptr) {
    omle_session_clear_inputs(reinterpret_cast<omle_session_t*>(ptr));
  });

  m.def("session_bind_input",
        [](uintptr_t sess, const char* name, uintptr_t tensor_ptr) {
          check(omle_session_bind_input(
              reinterpret_cast<omle_session_t*>(sess), name,
              reinterpret_cast<omle_tensor_t*>(tensor_ptr)));
        });

  m.def("session_run", [](uintptr_t ptr) {
    check(omle_session_run(reinterpret_cast<omle_session_t*>(ptr)));
  });

  // Returns a borrowed tensor ptr — valid until next session_run or
  // free_session.
  m.def("session_get_output", [](uintptr_t ptr, const char* name) -> uintptr_t {
    omle_tensor_t* t = nullptr;
    check(omle_session_get_output(reinterpret_cast<omle_session_t*>(ptr), name,
                                  &t));
    return reinterpret_cast<uintptr_t>(t);
  });

  m.def("session_num_inputs", [](uintptr_t ptr) {
    return omle_session_num_inputs(
        reinterpret_cast<const omle_session_t*>(ptr));
  });

  m.def("session_num_outputs", [](uintptr_t ptr) {
    return omle_session_num_outputs(
        reinterpret_cast<const omle_session_t*>(ptr));
  });

  // Bind column arrays as session inputs; call session_run() after.
  m.def(
      "session_bind_columns",
      [](uintptr_t sess_ptr, py::list col_names_py, py::list col_data_py) {
        int n_cols = (int)col_names_py.size();
        std::vector<std::string> cstrs(n_cols);
        std::vector<const char*> cnames(n_cols);
        for (int c = 0; c < n_cols; ++c) {
          cstrs[c] = col_names_py[c].cast<std::string>();
          cnames[c] = cstrs[c].c_str();
        }
        std::vector<int> col_types;
        std::vector<const void*> col_data;
        std::vector<std::vector<const char*>> str_bufs;
        std::vector<F32Array> f32_arrs;
        build_col_arrays(col_data_py, col_types, col_data, str_bufs, f32_arrs);
        int n_rows = n_rows_from_col_data(col_data_py);
        check(omle_session_bind_columns(
            reinterpret_cast<omle_session_t*>(sess_ptr), n_rows, n_cols,
            cnames.data(), col_types.data(), col_data.data()));
      },
      py::arg("session_ptr"), py::arg("col_names"), py::arg("col_data"));

  // ------------------------------------------------------------------
  // Session fast-path predict
  //   inp  : float32 C-contiguous (n_rows, n_cols) input array
  //   out  : float32 C-contiguous (n_rows, n_out_cols) output — written
  //   in-place
  // ------------------------------------------------------------------
  m.def(
      "session_predict_f32",
      [](uintptr_t session_ptr, const char* in_name, F32Array inp,
         const char* out_name, py::array_t<float, py::array::c_style> out) {
        auto ri = inp.request();
        auto ro = out.request();
        if (ri.ndim != 2) throw std::invalid_argument("input must be 2-D");
        if (ro.ndim != 2) throw std::invalid_argument("output must be 2-D");
        auto* sess = reinterpret_cast<omle_session_t*>(session_ptr);
        check(omle_session_predict_f32(
            sess, in_name, static_cast<int>(ri.shape[0]),
            static_cast<int>(ri.shape[1]), static_cast<const float*>(ri.ptr),
            out_name, static_cast<float*>(ro.ptr)));
      },
      py::arg("session_ptr"), py::arg("in_name"), py::arg("inp"),
      py::arg("out_name"), py::arg("out"),
      "Run one predict cycle: zero-copy input view, write output into caller "
      "buffer.");

  // Bind columns + run + write output in one shot (avoids 3 extra Python
  // round-trips).
  m.def(
      "session_predict_columns_f32",
      [](uintptr_t session_ptr, py::list col_names_py, py::list col_data_py,
         const char* out_name, py::array_t<float, py::array::c_style> out) {
        int n_cols = (int)col_names_py.size();
        std::vector<std::string> cstrs(n_cols);
        std::vector<const char*> cnames(n_cols);
        for (int c = 0; c < n_cols; ++c) {
          cstrs[c] = col_names_py[c].cast<std::string>();
          cnames[c] = cstrs[c].c_str();
        }
        std::vector<int> col_types;
        std::vector<const void*> col_data;
        std::vector<std::vector<const char*>> str_bufs;
        std::vector<F32Array> f32_arrs;
        build_col_arrays(col_data_py, col_types, col_data, str_bufs, f32_arrs);

        int n_rows = n_rows_from_col_data(col_data_py);
        auto ro = out.request();
        if (ro.ndim != 2) throw std::invalid_argument("output must be 2-D");
        check(omle_session_predict_columns_f32(
            reinterpret_cast<omle_session_t*>(session_ptr), n_rows, n_cols,
            cnames.data(), col_types.data(), col_data.data(), out_name,
            static_cast<float*>(ro.ptr)));
      },
      py::arg("session_ptr"), py::arg("col_names"), py::arg("col_data"),
      py::arg("out_name"), py::arg("out"),
      "Bind columns, run, and write output to caller buffer in one shot.");

  // Pre-register column schema once (avoids per-call name string building).
  m.def(
      "session_register_columns",
      [](uintptr_t session_ptr, py::list col_names_py, py::list col_types_py,
         const char* out_name) {
        int n = (int)col_names_py.size();
        std::vector<std::string> name_strs(n);
        std::vector<const char*> name_ptrs(n);
        std::vector<int> col_types(n);
        for (int i = 0; i < n; ++i) {
          name_strs[i] = col_names_py[i].cast<std::string>();
          name_ptrs[i] = name_strs[i].c_str();
          col_types[i] = col_types_py[i].cast<int>();
        }
        check(omle_session_register_columns(
            reinterpret_cast<omle_session_t*>(session_ptr), n, name_ptrs.data(),
            col_types.data(), out_name));
      },
      py::arg("session_ptr"), py::arg("col_names"), py::arg("col_types"),
      py::arg("out_name"),
      "Register column schema once so predict_registered_f32 needs no names "
      "per call.");

  // Predict using pre-registered schema: no name building per call.
  m.def(
      "session_predict_registered_f32",
      [](uintptr_t session_ptr, py::list col_data_py,
         py::array_t<float, py::array::c_style> out) {
        std::vector<int> col_types;
        std::vector<const void*> col_data;
        std::vector<std::vector<const char*>> str_bufs;
        std::vector<F32Array> f32_arrs;
        build_col_arrays(col_data_py, col_types, col_data, str_bufs, f32_arrs);

        int n_rows = n_rows_from_col_data(col_data_py);
        auto ro = out.request();
        if (ro.ndim != 2) throw std::invalid_argument("output must be 2-D");
        check(omle_session_predict_registered_f32(
            reinterpret_cast<omle_session_t*>(session_ptr), n_rows,
            col_data.data(), static_cast<float*>(ro.ptr)));
      },
      py::arg("session_ptr"), py::arg("col_data"), py::arg("out"),
      "Predict using pre-registered column schema (no per-call name "
      "overhead).");

  // ------------------------------------------------------------------
  // Tensor
  // ------------------------------------------------------------------
  m.def("tensor_create_f32",
        [](int n_rows, int n_cols, F32Array data) -> uintptr_t {
          auto req = data.request();
          auto* t = omle_tensor_create_f32(n_rows, n_cols,
                                           static_cast<const float*>(req.ptr));
          if (!t) throw std::runtime_error(omle_last_error());
          return reinterpret_cast<uintptr_t>(t);
        });

  m.def("tensor_create_strings",
        [](int n_rows, int n_cols, py::list strings_py) -> uintptr_t {
          int n = (int)strings_py.size();
          std::vector<const char*> ptrs(n);
          for (int i = 0; i < n; ++i) {
            PyObject* s = strings_py[i].ptr();
            const char* p = (s == Py_None) ? "" : PyUnicode_AsUTF8(s);
            ptrs[i] = p ? p : "";
          }
          auto* t = omle_tensor_create_strings(n_rows, n_cols, ptrs.data());
          if (!t) throw std::runtime_error(omle_last_error());
          return reinterpret_cast<uintptr_t>(t);
        });

  m.def("free_tensor", [](uintptr_t ptr) {
    omle_free_tensor(reinterpret_cast<omle_tensor_t*>(ptr));
  });

  m.def("tensor_dtype", [](uintptr_t ptr) {
    return (int)omle_tensor_dtype(reinterpret_cast<const omle_tensor_t*>(ptr));
  });

  m.def("tensor_n_rows", [](uintptr_t ptr) {
    return omle_tensor_n_rows(reinterpret_cast<const omle_tensor_t*>(ptr));
  });

  m.def("tensor_n_cols", [](uintptr_t ptr) {
    return omle_tensor_n_cols(reinterpret_cast<const omle_tensor_t*>(ptr));
  });

  // Read tensor data into a numpy array (copies data)
  m.def("tensor_read", [](uintptr_t ptr) -> py::array {
    return read_tensor(reinterpret_cast<const omle_tensor_t*>(ptr));
  });
}
