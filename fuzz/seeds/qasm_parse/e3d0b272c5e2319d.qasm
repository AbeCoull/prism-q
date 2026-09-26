OPENQASM 3.0;
qubit[2] q;
h q[0];
ctrl @ x q[0], q[1];
