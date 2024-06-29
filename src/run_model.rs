mod inference;

use anyhow::Result;
use ort::{GraphOptimizationLevel, Session};
use std::io::{self, BufRead};


use inference::string_to_tokens;
use crate::inference::is_end_of_sentence;
// use thiserror::Error; for libraries and AnyHow for binaries.

fn main() -> Result<()> {
	let session = Session::builder()?
		.with_optimization_level(GraphOptimizationLevel::Level3)?
		.with_intra_threads(1)?
		//.commit_from_file("rust_resources/model.onnx")?;
		.commit_from_memory(include_bytes!("../rust_resources/model.onnx")).unwrap();



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

	let stdin = io::stdin();
	let mut linebuffer: String = String::new();
	for line in stdin.lock().lines() {
		let newline = line.unwrap();
		if newline.is_empty() { continue; }
		linebuffer.push_str(&newline);
		let mut found_break = true;
		while found_break {
			found_break = false;
			for i in 1..linebuffer.len() {
				let is_eos = {
					let (head, _) = linebuffer.split_at(i);
					let is_eos = is_end_of_sentence(&session, head);
					if is_eos {
						println!("{}", head);
					}
					is_eos
				};
				if is_eos {
					found_break = true;
					_ = linebuffer.drain(..i);
					break;
				}
			}
		}
	}

	Ok(())
}


