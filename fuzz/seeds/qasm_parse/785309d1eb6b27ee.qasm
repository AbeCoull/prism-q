OPENQASM 3.0;
qubit[2] q;
gate mygate a, b { cx a, b; }
mygate q[0], q[1];
