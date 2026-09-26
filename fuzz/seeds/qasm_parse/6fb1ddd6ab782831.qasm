OPENQASM 3.0;
qubit[2] q;
xx_plus_yy(0.6, 0.3) q[0], q[1];
xx_minus_yy(0.9, -0.4) q[0], q[1];
rzz(0.31) q[0], q[1];
