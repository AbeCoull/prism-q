OPENQASM 3.0;
qubit[3] q;
bit[3] c;
x q[0];
measure q[0] -> c[0];
measure q[1] -> c[1];
if (c[0] ^ c[1]) { x q[2]; }
