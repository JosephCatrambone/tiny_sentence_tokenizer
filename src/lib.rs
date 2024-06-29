mod inference;

use ndarray::{array, concatenate, s, Array1, ArrayViewD, Axis};
use ort::{GraphOptimizationLevel, Session, inputs};
use pyo3::prelude::*;
use pyo3::anyhow;
use std::fmt::format;

const MODEL_MIN_LENGTH: usize = 24;

//#[pyclass(unsendable)]
#[pyclass]
struct SentenceSplitter {
	//#[pyo3(get)]
	model: Session,
}

#[pymethods]
impl SentenceSplitter {
	#[new]
	fn __new__() -> Self {
		let session = Session::builder().unwrap()
			.with_optimization_level(GraphOptimizationLevel::Level3).unwrap()
			.with_intra_threads(1).unwrap()
			//.commit_from_memory(include_bytes!(concat!(env!("OUT_DIR"), "/rust_resources/model.onnx")));
			.commit_from_memory(include_bytes!("../rust_resources/model.onnx")).unwrap();

		Self { model: session }
	}

	fn end_of_sentence(&self, sentence: &str) -> PyResult<bool> {
		Ok(inference::is_end_of_sentence(&self.model, sentence))
	}
}


/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
	Ok((a + b).to_string())
}

/// A Python module implemented in Rust.
#[pymodule]
fn tiny_sentence_tokenizer(m: &Bound<'_, PyModule>) -> PyResult<()> { // Must match the def in the cargo.toml.
	//m.add_function(wrap_pyfunction!(get_tokenizer, m)?)?;
	//m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
	m.add_class::<SentenceSplitter>()?;
	Ok(())
}
