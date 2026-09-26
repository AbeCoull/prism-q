OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
gpi(0.0) q[0];
gpi2(0.25) q[1];
ms(0.0, 0.0, 0.25) q[0], q[1];
