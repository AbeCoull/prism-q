OPENQASM 3.0;
qubit[2] q;
h q[0];
cnot q[0], q[1];
#pragma braket noise bit_flip(0.1) q[0]
#pragma braket result probability all
#pragma braket result expectation z(q[0]) @ z(q[1])
