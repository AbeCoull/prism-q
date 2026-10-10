//! A bounded worker pool that simulation calls run inside instead of the
//! process-wide one.

use prism_q::ThreadPool;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::error::PyPrismResult;

/// A bounded pool of `num_threads` workers (0 takes the default width).
///
/// `install(fn, *args, **kwargs)` calls `fn` on one of the workers, and every
/// parallel kernel a PRISM-Q call inside it reaches runs on this pool rather than
/// the process-wide one sized by `RAYON_NUM_THREADS`.
#[pyclass(name = "ThreadPool", module = "prism_q", frozen)]
pub struct PyThreadPool {
    inner: ThreadPool,
}

#[pymethods]
impl PyThreadPool {
    #[new]
    #[pyo3(signature = (num_threads = 0))]
    fn new(num_threads: usize) -> PyPrismResult<Self> {
        Ok(Self {
            inner: ThreadPool::with_threads(num_threads)?,
        })
    }

    /// Worker count, with the default width resolved.
    #[getter]
    fn num_threads(&self) -> usize {
        self.inner.num_threads()
    }

    /// Call `fn(*args, **kwargs)` on a worker of this pool and return its
    /// result, blocking the calling thread until it does. `fn` runs on another
    /// OS thread, so thread-local state of the caller is not visible to it.
    #[pyo3(signature = (r#fn, *args, **kwargs))]
    fn install(
        &self,
        py: Python<'_>,
        r#fn: Py<PyAny>,
        args: Py<PyTuple>,
        kwargs: Option<Py<PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        py.detach(|| {
            self.inner.install(|| {
                Python::attach(|py| {
                    r#fn.call(py, args.bind(py), kwargs.as_ref().map(|k| k.bind(py)))
                })
            })
        })
    }

    fn __repr__(&self) -> String {
        format!("ThreadPool(num_threads={})", self.inner.num_threads())
    }
}
