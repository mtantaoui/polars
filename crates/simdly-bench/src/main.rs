//! EXPERIMENT (simdly): median time of each float math expression, in f32 and
//! f64, on 64K and 1M rows spread over each function's usual range, and the
//! largest distance in ULP from libm's f64 result rounded to the column's
//! type.
//!
//! ```bash
//! cargo run --release -p simdly-bench > before.txt
//! cargo run --release -p simdly-bench --features simdly > after.txt
//! ```
//!
//! Each query runs on the in-memory engine over a single-chunk column, so
//! the kernel is called once on the whole buffer. A column with nulls is
//! checked too: the output must keep them where they were.

use std::hint::black_box;
use std::time::{Duration, Instant};

use polars::prelude::*;
use polars_config::Engine;
use polars_core::query_result::QueryResult;

type Op = fn(Expr) -> Expr;

/// Name, input range, the expression, and the f64 reference.
const OPS: &[(&str, f64, f64, Op, fn(f64) -> f64)] = &[
    ("exp", -10.0, 10.0, |e| e.exp(), libm::exp),
    ("log1p", -0.5, 10.0, |e| e.log1p(), libm::log1p),
    ("erf", -5.0, 5.0, |e| e.erf(), libm::erf),
    ("cbrt", -100.0, 100.0, |e| e.cbrt(), libm::cbrt),
    ("sin", -10.0, 10.0, |e| e.sin(), libm::sin),
    ("cos", -10.0, 10.0, |e| e.cos(), libm::cos),
    ("tan", -1.5, 1.5, |e| e.tan(), libm::tan),
    ("arcsin", -1.0, 1.0, |e| e.arcsin(), libm::asin),
    ("arccos", -1.0, 1.0, |e| e.arccos(), libm::acos),
    ("arctan", -10.0, 10.0, |e| e.arctan(), libm::atan),
    ("sinh", -5.0, 5.0, |e| e.sinh(), libm::sinh),
    ("cosh", -5.0, 5.0, |e| e.cosh(), libm::cosh),
    ("tanh", -5.0, 5.0, |e| e.tanh(), libm::tanh),
    ("arcsinh", -10.0, 10.0, |e| e.arcsinh(), libm::asinh),
    ("arccosh", 1.0, 100.0, |e| e.arccosh(), libm::acosh),
    ("arctanh", -0.99, 0.99, |e| e.arctanh(), libm::atanh),
];

const SIZES: &[(&str, usize)] = &[("64k", 1 << 16), ("1m", 1 << 20)];

/// `size` values evenly spread over `[lo, hi]`, with every 7th one null when
/// `nulls`.
fn values(size: usize, lo: f64, hi: f64, nulls: bool) -> Vec<Option<f64>> {
    (0..size)
        .map(|i| (!nulls || i % 7 != 0).then(|| lo + (hi - lo) * i as f64 / size as f64))
        .collect()
}

/// A single-chunk frame with column `x` holding `values` as `dtype`.
fn frame(values: &[Option<f64>], dtype: &DataType) -> DataFrame {
    let s = Series::new("x".into(), values).cast(dtype).unwrap();
    let mut df = DataFrame::new_infer_height(vec![s.into()]).unwrap();
    df.rechunk_mut();
    df
}

fn run(df: &DataFrame, op: Op) -> Series {
    let out = df
        .clone()
        .lazy()
        .select([op(col("x"))])
        .collect_with_engine(Engine::InMemory)
        .unwrap();
    let QueryResult::Single(out) = out else {
        unreachable!()
    };
    out.columns()[0].as_materialized_series().clone()
}

/// Distance in ULP between `a` and `b` in their own type; 0 for two NaNs.
fn ulps(a: f64, b: f64, f32: bool) -> u64 {
    if a.is_nan() && b.is_nan() {
        return 0;
    }
    if a.is_nan() || b.is_nan() {
        return u64::MAX;
    }
    if f32 {
        let key = |x: f32| {
            let b = x.to_bits() as i64;
            if b < 0x8000_0000 { b } else { 0x8000_0000 - b }
        };
        key(a as f32).abs_diff(key(b as f32))
    } else {
        let key = |x: f64| {
            let b = x.to_bits() as i128;
            if b < 1 << 63 { b } else { (1 << 63) - b }
        };
        key(a).abs_diff(key(b)) as u64
    }
}

/// Largest ULP distance of `out` from `reference` applied to `input`.
fn max_ulps(input: &[Option<f64>], out: &Series, reference: fn(f64) -> f64, f32: bool) -> u64 {
    let out = out.cast(&DataType::Float64).unwrap();
    let out = out.f64().unwrap();
    let mut worst = 0;
    for (x, y) in input.iter().zip(out.iter()) {
        if let (Some(x), Some(y)) = (x, y) {
            // The input as the column holds it, then the reference rounded
            // to the column's type.
            let x = if f32 { *x as f32 as f64 } else { *x };
            let want = if f32 {
                reference(x) as f32 as f64
            } else {
                reference(x)
            };
            worst = worst.max(ulps(y, want, f32));
        }
    }
    worst
}

/// Median of `f`'s run times, over at least 10 runs and about 0.5 s.
fn median(mut f: impl FnMut()) -> Duration {
    for _ in 0..3 {
        f();
    }
    let mut times = Vec::new();
    let start = Instant::now();
    while times.len() < 10 || (start.elapsed() < Duration::from_millis(500) && times.len() < 1000) {
        let t = Instant::now();
        f();
        times.push(t.elapsed());
    }
    times.sort();
    times[times.len() / 2]
}

fn main() {
    println!("op\tcase\tmedian_ns\tmax_ulp");
    for &(name, lo, hi, op, reference) in OPS {
        for (dname, dtype) in [("f32", DataType::Float32), ("f64", DataType::Float64)] {
            let f32 = dname == "f32";

            let input = values(1000, lo, hi, true);
            let out = run(&frame(&input, &dtype), op);
            assert_eq!(out.dtype(), &dtype, "{name} {dname}: output type");
            let in_nulls: Vec<bool> = input.iter().map(Option::is_none).collect();
            let out_nulls: Vec<bool> = out.is_null().iter().map(|v| v.unwrap()).collect();
            assert_eq!(in_nulls, out_nulls, "{name} {dname}: nulls moved");

            for &(sname, size) in SIZES {
                let input = values(size, lo, hi, false);
                let df = frame(&input, &dtype);
                let worst = max_ulps(&input, &run(&df, op), reference, f32);
                let t = median(|| {
                    let _ = black_box(run(&df, op));
                });
                println!("{name}\t{dname}_{sname}\t{}\t{worst}", t.as_nanos());
            }
        }
    }
}
