OPENQASM 3.0;
qubit[4] q;
for int i in [0:3] {
    h q[i];
}
