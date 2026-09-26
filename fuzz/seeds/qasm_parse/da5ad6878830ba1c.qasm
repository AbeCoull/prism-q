OPENQASM 3.0;
qubit[4] q;
for int i in [3:-1:0] {
    x q[i];
}
