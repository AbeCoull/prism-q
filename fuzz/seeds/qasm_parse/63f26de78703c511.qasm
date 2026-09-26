OPENQASM 3.0;
qubit[3] q;
for int i in [0:2] {
    rx(i * pi / 4) q[i];
}
