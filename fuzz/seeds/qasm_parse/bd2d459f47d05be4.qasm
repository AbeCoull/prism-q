OPENQASM 3.0;
qubit[2] q;
x q[0];
ctrl @ rx(pi/4) q[0], q[1];
inv @ ctrl @ rx(pi/4) q[0], q[1];
