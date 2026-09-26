OPENQASM 3.0;
qubit[2] q;
gate myrxx(theta) a, b {
    rx(theta/2) a;
    cx a, b;
    rx(-theta/2) a;
    cx a, b;
}
myrxx(pi) q[0], q[1];
