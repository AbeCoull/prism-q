OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
bit[2] c;
ry(0.7853981633974483) q[0];
cx q[0], q[1];
rz(1.5707963267948966) q[1];
c[0] = measure q[0];
c[1] = measure q[1];
