OPENQASM 3.0;
qubit[3] q;
h q[0];
h q[1];
ccx q[0], q[1], q[2];
