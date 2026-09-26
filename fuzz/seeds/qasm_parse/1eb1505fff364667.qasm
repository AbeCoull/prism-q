OPENQASM 3.0;
qubit[2] q;
gate bell a, b {
    h a;
    cx a, b;
}
bell q[0], q[1];
