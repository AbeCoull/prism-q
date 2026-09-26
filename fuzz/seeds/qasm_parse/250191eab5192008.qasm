OPENQASM 3.0;
qubit[8] q;
for int i in [0:0x3] {
    x q[i];
}
