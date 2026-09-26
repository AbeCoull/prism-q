OPENQASM 3.0;
qubit[1] q;
def st(qubit a) {
    s a;
    t a;
}
h q[0];
st(q[0]);
inv @ st(q[0]);
