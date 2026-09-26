OPENQASM 3.0;
qubit[3] q;
x q[0];
ctrl @ ctrl @ x q[0], q[1], q[2];
