OPENQASM 3.0;
qubit[5] q;
for int i in {0, 2, 4} {
    x q[i];
}
