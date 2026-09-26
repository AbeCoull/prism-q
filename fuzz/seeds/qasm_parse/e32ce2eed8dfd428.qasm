OPENQASM 3.0;
qubit[1] q;
gate myrz(theta) a {
    rz(theta) a;
}
myrz(pi/4) q[0];
