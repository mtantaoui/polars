//! EXPERIMENT: element-wise float math through simdly's slice kernels.

use polars_arrow::array::PrimitiveArray;
use polars_core::prelude::*;

/// `f` applied to each chunk's values buffer, keeping its validity. Values
/// under nulls are computed too and stay hidden, as with `apply_values`.
pub fn simdly_apply<T: PolarsNumericType>(
    ca: &ChunkedArray<T>,
    f: impl Fn(&[T::Native]) -> Vec<T::Native>,
) -> ChunkedArray<T> {
    let chunks = ca
        .data_views()
        .zip(ca.iter_validities())
        .map(|(values, validity)| {
            PrimitiveArray::from_vec(f(values)).with_validity(validity.cloned())
        });
    ChunkedArray::from_chunk_iter(ca.name().clone(), chunks)
}
