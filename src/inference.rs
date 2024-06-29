use ndarray::{Array1, ArrayViewD, Axis};
use ort::{inputs, Session};


const MODEL_CONTEXT_SIZE: usize = 24;


/// Returns the last n characters or the full string if len(s) < n.
fn last_n_characters(s: &str, n: usize) -> &str {
	if s.len() <= n { return s };
	let split_pos = s.char_indices().nth_back(n).unwrap().0;
	&s[split_pos..]
}

pub fn string_to_tokens(s: &str) -> Vec<i64> {
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

pub fn is_end_of_sentence(model: &Session, s: &str) -> bool {
	let (p_not_eos, p_eos) = get_eos_probabilities(model, s);
	p_eos > p_not_eos
}

pub fn get_eos_probabilities(model: &Session, s: &str) -> (f32, f32) {
	let tokens = string_to_tokens(s);
	let mut tokens = Array1::from_iter(tokens.iter().cloned());
	//let outputs = model.run(ort::inputs!["input" => image]?)?;
	//let predictions = outputs["output"].try_extract_tensor::<f32>()?;
	//let array = tokens.view().insert_axis(Axis(0)).insert_axis(Axis(1));
	let array = tokens.view().insert_axis(Axis(0));
	let outputs = model.run(inputs![array].unwrap()).unwrap();
	//let generated_tokens: ArrayViewD<f32> = outputs["output"].try_extract_tensor().unwrap();
	let mut probabilities: Vec<(usize, f32)> = outputs["output"]
		.try_extract_tensor().unwrap()
		//.softmax(ndarray::Axis(1))
		.iter()
		.copied()
		.enumerate()
		.collect::<Vec<_>>();
	(probabilities[0].1, probabilities[1].1)
}
