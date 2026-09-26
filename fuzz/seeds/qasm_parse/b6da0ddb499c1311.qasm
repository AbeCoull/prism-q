OPENQASM 3.0;
qubit[3] q;
def at_idx(int i, qubit a) {
    rx(i * pi / 2) a;
}
at_idx(2, q[1]);
