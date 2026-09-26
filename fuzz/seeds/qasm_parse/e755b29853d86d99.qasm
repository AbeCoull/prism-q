OPENQASM 3.0;
qubit[1] q;
gate mg a { inv @ t a; s a; }
h q[0];
mg q[0];
inv @ mg q[0];
