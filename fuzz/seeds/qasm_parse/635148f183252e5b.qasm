OPENQASM 3.0;
input float[64] t;
qubit[3] q;
h q[0];
rxyz(0.0) q[0], q[1], q[2];
rxx(t) q[0], q[1];
