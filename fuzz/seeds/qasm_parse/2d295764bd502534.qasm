OPENQASM 3.0;
qubit[2] q;
gate myh a { h a; }
gate mybell a, b { myh a; cx a, b; }
mybell q[0], q[1];
