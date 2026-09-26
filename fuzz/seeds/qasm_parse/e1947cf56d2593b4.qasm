OPENQASM 3.0;
qubit[2] q;
bit[1] c;
c[0] = measure q[0];
if (c[0]) x q[1];
