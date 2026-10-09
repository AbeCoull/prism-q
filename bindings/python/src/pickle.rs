//! `__reduce__` plumbing shared by the picklable classes.

use pyo3::prelude::*;
use pyo3::types::PyBytes;

/// What `__reduce__` returns: a constructor and its one bytes argument.
pub(crate) type Reduced<'py> = (Bound<'py, PyAny>, (Bound<'py, PyBytes>,));

/// Reduce to the class's `_from_pickle` applied to `data`.
pub(crate) fn reduce<'py>(slf: &Bound<'py, PyAny>, data: Vec<u8>) -> PyResult<Reduced<'py>> {
    let ctor = slf.get_type().getattr("_from_pickle")?;
    Ok((ctor, (PyBytes::new(slf.py(), &data),)))
}

/// What `__reduce__` returns for an enum member: `getattr` and its arguments.
pub(crate) type ReducedMember<'py> = (Bound<'py, PyAny>, (Bound<'py, PyAny>, &'static str));

/// Reduce a unit enum member to `getattr(cls, name)`.
pub(crate) fn reduce_member<'py>(
    slf: &Bound<'py, PyAny>,
    name: &'static str,
) -> PyResult<ReducedMember<'py>> {
    let getattr = slf.py().import("builtins")?.getattr("getattr")?;
    Ok((getattr, (slf.get_type().into_any(), name)))
}
