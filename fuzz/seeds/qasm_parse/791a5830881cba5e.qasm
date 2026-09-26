OPENQASM 3.0;
qubit[2] q;
h q[0];
ctrl @ rx(pi/3) q[0], q[1];
