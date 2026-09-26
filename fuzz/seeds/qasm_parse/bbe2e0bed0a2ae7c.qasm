OPENQASM 3.0;
qubit[2] q;
gate st a { s a; t a; }
h q[0];
h q[1];
st q;
inv @ st q;
