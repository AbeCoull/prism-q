OPENQASM 3.0;
qubit[2] q;
def g(qubit a) { h a; }
ctrl @ g(q[0]);
