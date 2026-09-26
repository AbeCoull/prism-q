OPENQASM 3.0;
qubit[4] q;
for int i in [0:1] {
    for int j in [0:1] {
        cx q[i], q[j+2];
    }
}
