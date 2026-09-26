OPENQASM 3.0;
qubit[2] q;
def bell(qubit a, qubit b) {
    h a;
    cx a, b;
}
bell(q[0]);
