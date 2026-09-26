OPENQASM 3.0;
qubit[4] q;
def step(qubit a) {
    h a;
}
for int i in [0:3] {
    step(q[i]);
}
