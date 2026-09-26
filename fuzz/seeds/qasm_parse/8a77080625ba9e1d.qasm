OPENQASM 3.0;
qubit[2] q;
gpi(0.0) q[0];
gpi2(0.25) q[1];
ms(0.1, 0.2, 0.25) q[0], q[1];
