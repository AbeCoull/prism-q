OPENQASM 3.0;
qubit[1] q;
gate mygate a, b { cx a, b; }
mygate q[0];
