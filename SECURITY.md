# Security Policy

## Supported Versions

Security fixes target the default branch. Published releases may receive fixes when a safe backport is practical.

## Reporting a Vulnerability

Report suspected vulnerabilities privately through GitHub Security Advisories:

https://github.com/AbeCoull/prism-q/security/advisories/new

Do not open a public issue for security reports.

Include as much detail as possible:

- Affected version, commit, or branch
- Operating system and Rust version
- Feature flags used
- Steps to reproduce
- Expected and actual behavior
- Impact assessment
- Proof of concept, crash log, or benchmark input if available
- Suggested fix if known

## Response Expectations

Maintainers aim to acknowledge reports within 7 days and provide an initial assessment within 14 days when enough information is available.

Fix timing depends on severity, exploitability, complexity, and release risk.

## Scope

Security-relevant reports include memory safety issues, undefined behavior, dependency vulnerabilities, unsafe handling of untrusted input, denial of service cases, and incorrect behavior that could affect downstream systems relying on simulator results.

General bugs, performance regressions, and correctness issues without a security impact should be reported as normal GitHub issues.

## Disclosure

Please allow time for a fix before public disclosure. Coordinated disclosure helps protect users while preserving a clear public record once a patch is available.
