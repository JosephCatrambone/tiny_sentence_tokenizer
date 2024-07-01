mod inference;

use ort::Session;
use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::inference::instance_model;

fn bulk_split(model: &Session, text: &str, min_probability: Option<f32>) -> Vec<String> {
	let mut sentences: Vec<String> = vec![];

	// Fill up 'unprocessed' with more and more of the characters from text.
	// When unprocessed finally reaches the end of a sentence, pull off that chunk and add it to 'sentences'.
	// At the end, take whatever is left and add it.
	let mut idx = 1;
	let mut unprocessed = text.to_string();
	while !unprocessed.is_empty() && idx < unprocessed.len() {
		let (head, _) = unprocessed.split_at(idx);
		let (p_not_eos, p_eos) = inference::get_eos_probabilities(model, head);
		let is_eos = if let Some(p) = min_probability {
			p_eos > p
		} else {
			p_eos > p_not_eos
		};
		if is_eos {
			sentences.push(unprocessed.drain(..idx).collect::<String>());
			idx = 0;
		} else {
			idx += 1;
		}
	}
	if !unprocessed.is_empty() {
		sentences.push(unprocessed);
	}

	sentences
}


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
		Self { model: instance_model() }
	}

	/// Returns true if the last character of the given string is the end of a sentence.
	fn end_of_sentence(&self, sentence: &str) -> PyResult<bool> {
		Ok(inference::is_end_of_sentence(&self.model, sentence))
	}

	/// Returns the probability (0-1 inclusive) that the last character of the given string is the end of a sentence.
	fn p_end_of_sentence(&self, sentence: &str) -> PyResult<f32> {
		let (_p_not_eos, p_eos) = inference::get_eos_probabilities(&self.model, sentence);
		Ok(p_eos)
	}

	/// Returns a list of sentences, given a blob of text `text`.
	///
	/// If `min_probability` is unspecified or `None`, will split a sentence when the probability
	/// a sentence is complete exceeds 50%.  (Actually, when the predicted 'eos' > 'not eos'.)
	#[pyo3(signature = (text, min_probability=None))]
	fn split_text(&self, text: &str, min_probability: Option<f32>) -> PyResult<PyObject> {
		let sentences = bulk_split(&self.model, text, min_probability);

		// Convert the vec<str> to Python:
		Python::with_gil(|py| {
			let res = PyList::new_bound(py, sentences);
			Ok(res.into())
		})
	}
}


/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
	Ok((a + b).to_string())
}

/// A Python module implemented in Rust.
#[pymodule]
//#[pyo3(name="my_lib_name")] // takes this fn name by default. If unspecified, must match the def in the cargo.toml.
fn tiny_sentence_tokenizer(m: &Bound<'_, PyModule>) -> PyResult<()> {
	//m.add_function(wrap_pyfunction!(get_tokenizer, m)?)?;
	//m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
	m.add_class::<SentenceSplitter>()?;
	Ok(())
}
