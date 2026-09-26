OPENQASM 3.0;
qubit[1] q;
bit[3] c;
if (c[0] ^ c[2]) { x q[0]; }
