OPENQASM 3.0;
qubit[1] q;
gate myu3(theta, phi, lambda) a {
    rz(lambda) a;
    ry(theta) a;
    rz(phi) a;
}
myu3(pi/2, pi/4, pi/8) q[0];
