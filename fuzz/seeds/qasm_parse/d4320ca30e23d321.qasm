OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
h q[0];
crx(pi / 3) q[0], q[1];
cry(pi / 4) q[0], q[1];
crz(pi / 5) q[0], q[1];
swap q[0], q[1];
