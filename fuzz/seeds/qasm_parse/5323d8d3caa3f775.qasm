OPENQASM 3.0;
qubit[3] q;
h q[0];
h q[1];
ctrl @ ctrl @ x q[0], q[1], q[2];
