
use ort::{GraphOptimizationLevel, inputs, Session};


const MODEL_CONTEXT_SIZE: usize = 64;


pub fn instance_model() -> Session {
	Session::builder().unwrap()
		.with_optimization_level(GraphOptimizationLevel::Level3).unwrap()
		.with_intra_threads(1).unwrap()
		.commit_from_memory(include_bytes!("../trained_models/sentence_tokenizer_v11_64_prefix_64_suffix_256.onnx")).unwrap()
}


/// Returns the last n characters or the full string if len(s) < n.
fn last_n_characters(s: &str, n: usize) -> &str {
	if s.len() <= n { return s };
	let split_pos = s.char_indices().nth_back(n).unwrap().0;
	&s[split_pos..]
}

/// Returns the first n characters or the full string if len(s) < n.
fn first_n_characters(s: &str, n: usize) -> &str {
	if s.len() <= n { return s; }
	let split_pos = s.char_indices().nth(n).unwrap().0;
	&s[..split_pos]
}

pub fn prefix_to_tokens(s: &str) -> Vec<i64> {
	// Make sure we have AT LEAST n characers in our string, and left fill with spaces.
	//let padded_string = format!("{: >24}", s);
	let padded_string = format!("{: >MODEL_CONTEXT_SIZE$}", s);
	// Cut off the last n _characters_ which we will convert to bytes.
	let truncated_string = last_n_characters(&padded_string, MODEL_CONTEXT_SIZE);
	// Convert everything to bytes.
	let sbytes = truncated_string.as_bytes().iter().map(|b: &u8| { *b as i64 }).collect::<Vec::<i64>>();
	// Then cut off the bytes at 24.
	sbytes[sbytes.len().saturating_sub(MODEL_CONTEXT_SIZE)..].to_vec()
}

pub fn suffix_to_tokens(s: &str) -> Vec<i64> {
	let padded_string = format!("{: <MODEL_CONTEXT_SIZE$}", s);
	let truncated_string = first_n_characters(&padded_string, MODEL_CONTEXT_SIZE);
	let sbytes = truncated_string.as_bytes().iter().map(|b: &u8| { *b as i64 }).collect::<Vec::<i64>>();
	sbytes[..sbytes.len().saturating_sub(MODEL_CONTEXT_SIZE)].to_vec()
}

pub fn is_end_of_sentence(model: &Session, s: &str) -> bool {
	let (p_not_eos, p_eos) = get_eos_probabilities(model, s, None);
	p_eos > p_not_eos
}

/*
pub fn get_eos_probabilities_ndarray(model: &Session, s: &str) -> (f32, f32) {
	use ndarray::{Array1, ArrayViewD, Axis};

	let tokens = string_to_tokens(s);
	let tokens = Array1::from_iter(tokens.iter().cloned());
	//let outputs = model.run(ort::inputs!["input" => image]?)?;
	//let predictions = outputs["output"].try_extract_tensor::<f32>()?;
	//let array = tokens.view().insert_axis(Axis(0)).insert_axis(Axis(1));
	let array = tokens.view().insert_axis(Axis(0));
	let outputs = model.run(inputs![array].unwrap()).unwrap();
	//let generated_tokens: ArrayViewD<f32> = outputs["output"].try_extract_tensor().unwrap();
	let probabilities: Vec<(usize, f32)> = outputs["output"]
		.try_extract_tensor().unwrap()
		//.softmax(ndarray::Axis(1))
		.iter()
		.copied()
		.enumerate()
		.collect::<Vec<_>>();
	(probabilities[0].1, probabilities[1].1)
}
*/

pub fn get_eos_probabilities(model: &Session, s: &str, lookahead: Option<&str>) -> (f32, f32) {
	let prefix_tokens = prefix_to_tokens(s);
	let suffix_tokens = if let Some(suffix) = lookahead {
		suffix_to_tokens(suffix)
	} else {
		vec![b' ' as i64; MODEL_CONTEXT_SIZE]
	};
	// Raw tensor construction takes a tuple of (dimensions, data).
	// The model expects our input to have shape [B, _, S]
	let prefix_input = (vec![1, prefix_tokens.len() as i64], prefix_tokens.into_boxed_slice());
	let suffix_input = (vec![1, suffix_tokens.len() as i64], suffix_tokens.into_boxed_slice());
	let outputs = model.run(inputs![prefix_input, suffix_input].unwrap()).unwrap();
	let (_dims, probabilities): (Vec<i64>, &[f32]) = outputs["output"]
		.try_extract_raw_tensor::<f32>().unwrap();
	(probabilities[0], probabilities[1])
}
