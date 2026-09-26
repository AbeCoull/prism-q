OPENQASM 3.0;
qubit[3] q;
x q[0];
x q[2];
cswap q[0], q[1], q[2];
