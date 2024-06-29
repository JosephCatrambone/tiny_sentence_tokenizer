use anyhow::Result;
use ndarray::{array, concatenate, s, Array1, ArrayViewD, Axis};
use ort::{GraphOptimizationLevel, Session, inputs};
use std::fmt::format;

// use thiserror::Error; for libraries and AnyHow for binaries.

fn main() -> Result<()> {
	let session = Session::builder()?
		.with_optimization_level(GraphOptimizationLevel::Level3)?
		.with_intra_threads(1)?
		.commit_from_file("rust_resources/model.onnx")?;

	let tokens = tokenize("oh no");
	let mut tokens = Array1::from_iter(tokens.iter().cloned());

	//let outputs = model.run(ort::inputs!["input" => image]?)?;
	//let predictions = outputs["output"].try_extract_tensor::<f32>()?;
	//let array = tokens.view().insert_axis(Axis(0)).insert_axis(Axis(1));
	let array = tokens.view().insert_axis(Axis(0));
	let outputs = session.run(inputs![array]?)?;
	let generated_tokens: ArrayViewD<f32> = outputs["output"].try_extract_tensor()?;

	// Collect and sort logits
	/*
	let probabilities = &mut generated_tokens
		.slice(s![0, 0, -1, ..])
		.insert_axis(Axis(0))
		.to_owned()
		.iter()
		.cloned()
		.enumerate()
		.collect::<Vec<_>>();
	probabilities.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Less));
	*/
	dbg!(generated_tokens);

	Ok(())
}

fn tokenize(s: &str) -> Vec<i64> {
	let padded_string = format!("{: >24}", s);
	let sbytes = padded_string.as_bytes().iter().map(|b: &u8| { *b as i64 }).collect::<Vec::<i64>>();
	sbytes[sbytes.len().saturating_sub(24)..].to_vec()
}
