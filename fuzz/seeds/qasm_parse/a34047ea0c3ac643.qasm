OPENQASM 3.0;
qubit[3] q;
def all_h(qubit a, qubit b, qubit c) {
    for int i in [0:2] {
        rx(i * pi / 2) a;
    }
    h b;
    h c;
}
all_h(q[0], q[1], q[2]);
