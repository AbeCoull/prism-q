OPENQASM 3.0;
qubit[1] q;
gate myrot(theta) a {
    rx(theta) a;
}
myrot(pi) q[0];
