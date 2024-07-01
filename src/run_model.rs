mod inference;

use anyhow::Result;
use std::io::{self, BufRead};

use crate::inference::{instance_model, is_end_of_sentence};
// use thiserror::Error; for libraries and AnyHow for binaries.

fn main() -> Result<()> {
	let session = instance_model();

	let stdin = io::stdin();
	let mut linebuffer: String = String::new();
	for line in stdin.lock().lines() {
		let newline = line.unwrap();
		linebuffer.push_str(&newline);
		if linebuffer.is_empty() { continue; }
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


