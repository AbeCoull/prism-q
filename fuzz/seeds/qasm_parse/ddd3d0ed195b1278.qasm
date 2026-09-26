OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
h q[0];
syc q[0], q[1];
sqrt_iswap q[0], q[1];
