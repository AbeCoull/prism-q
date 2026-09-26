OPENQASM 3.0;
qubit[2] q;
x q[0];
x q[1];
ctrl @ rz(pi) q[0], q[1];
