OPENQASM 3.0;
qubit[1] q;
gate st a { s a; t a; }
h q[0];
pow(2) @ st q[0];
