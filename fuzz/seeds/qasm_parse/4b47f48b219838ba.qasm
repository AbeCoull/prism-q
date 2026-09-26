OPENQASM 3.0;
qubit[2] q;
bit[1] c;
gate pair a, b { x a; cx a, b; }
if (c[0]) pair q[0], q[1];
