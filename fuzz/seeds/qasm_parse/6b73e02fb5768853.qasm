OPENQASM 3.0;
qubit[2] q;
h q[0];
iswap q[0], q[1];
inv @ iswap q[0], q[1];
