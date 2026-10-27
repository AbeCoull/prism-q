OPENQASM 3.0;
qubit[1] q;
gate g a { x a; h a; }
pow(1000) @ g q[0];
