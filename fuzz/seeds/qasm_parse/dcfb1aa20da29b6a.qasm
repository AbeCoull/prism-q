OPENQASM 3.0;
qubit[1] q;
gate st a { s a; t a; }
h q[0];
st q[0];
pow(-1) @ st q[0];
