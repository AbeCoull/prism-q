OPENQASM 3.0;
qubit[4] q;
bit[2] c;
x q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
if (c[0]) { x q[2]; } else if (c[1]) { x q[3]; } else { x q[2]; x q[3]; }
