#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| prism_q_fuzz::check_parse_tokens(data));
