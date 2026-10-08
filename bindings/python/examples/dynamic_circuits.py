"""Teleport a qubit with mid-circuit measurement and feed-forward from OpenQASM 3."""

import math

from prism_q import parse_qasm, simulate

theta = 1.2
teleport = parse_qasm(f"""
OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
bit[3] c;
ry({theta}) q[0];
h q[1];
cx q[1], q[2];
cx q[0], q[1];
h q[0];
c[0] = measure q[0];
c[1] = measure q[1];
if (c[1]) x q[2];
if (c[0]) z q[2];
c[2] = measure q[2];
""")
shots = simulate(teleport).seed(42).shots(100_000).shots
print(f"P(1) on q[2]: {shots[:, 2].mean():.4f}, expected {math.sin(theta / 2) ** 2:.4f}")

branch = parse_qasm("""
OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
bit[3] c;
h q[0];
c[0] = measure q[0];
reset q[0];
if (c[0]) {
  x q[1];
} else {
  h q[1];
}
c[1] = measure q[1];
c[2] = measure q[0];
""")
print(sorted(simulate(branch).seed(42).sample_counts(1000).counts().items()))
