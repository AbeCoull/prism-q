OPENQASM 3.0;
qubit[1] q;
bit[1] c;
if (c[0]) { measure q[0] -> c[0]; } else { x q[0]; }
