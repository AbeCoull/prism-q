//! Environment knobs, parsed once per process. An invalid value warns on stderr and
//! yields the default: readers sit on infallible paths, so a typo must neither fail
//! the run nor pass silently.

/// A count knob: `default` when unset, and when the value does not parse or
/// falls below `min`, with a warning naming the variable.
pub(crate) fn usize_knob(var: &str, default: usize, min: usize) -> usize {
    parse_usize_knob(var, std::env::var(var).ok(), default, min)
}

/// A count override: `None` when unset, and when the value does not parse or
/// falls below `min`, with a warning naming the variable, so the caller's own
/// fallback (detection, a derived budget) applies.
pub(crate) fn usize_override(var: &str, min: usize) -> Option<usize> {
    parse_usize_override(var, std::env::var(var).ok(), min)
}

pub(crate) fn parse_usize_knob(
    var: &str,
    raw: Option<String>,
    default: usize,
    min: usize,
) -> usize {
    parse_usize_override(var, raw, min).unwrap_or(default)
}

pub(crate) fn parse_usize_override(var: &str, raw: Option<String>, min: usize) -> Option<usize> {
    let raw = raw?;
    match raw.trim().parse::<usize>() {
        Ok(n) if n >= min => Some(n),
        Ok(n) => {
            eprintln!("warning: {var}={n} is below the minimum of {min}; ignoring it.");
            None
        }
        Err(_) => {
            eprintln!("warning: {var}={raw:?} is not a count; ignoring it.");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The readers cache per process, so the parsers are exercised directly
    // rather than through the environment.
    #[test]
    fn a_count_knob_falls_back_on_anything_it_cannot_use() {
        let parse = |raw: &str| parse_usize_knob("PRISM_TEST", Some(raw.into()), 7, 1);
        assert_eq!(parse("4"), 4);
        assert_eq!(parse(" 4 "), 4);
        assert_eq!(parse("abc"), 7, "unparseable falls back");
        assert_eq!(parse("-1"), 7, "negative falls back");
        assert_eq!(parse("0"), 7, "below the minimum falls back");
        assert_eq!(parse(""), 7, "empty falls back");
        assert_eq!(parse_usize_knob("PRISM_TEST", None, 7, 1), 7);
    }

    #[test]
    fn a_count_override_is_absent_when_it_cannot_be_used() {
        let parse = |raw: &str| parse_usize_override("PRISM_TEST", Some(raw.into()), 0);
        assert_eq!(parse("20"), Some(20));
        assert_eq!(parse("0"), Some(0), "the minimum itself is accepted");
        assert_eq!(parse("abc"), None, "unparseable is absent, not zero");
        assert_eq!(parse("2e1"), None);
        assert_eq!(parse_usize_override("PRISM_TEST", None, 0), None);
    }
}
