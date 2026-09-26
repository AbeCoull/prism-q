OPENQASM 3.0;
qubit[1] q;
def loop(qubit a) {
    loop(a);
}
loop(q[0]);
