OPENQASM 3.0;
qubit[1] q;
gate half_rot(theta) a {
    rz(theta + pi) a;
}
half_rot(pi/4) q[0];
