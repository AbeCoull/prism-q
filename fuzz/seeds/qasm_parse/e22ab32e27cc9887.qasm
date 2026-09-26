OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
u3(pi / 2, 0, pi) q[0];
cx q[0], q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
if (c == 3) x q[0];
