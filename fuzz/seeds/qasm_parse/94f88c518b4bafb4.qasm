OPENQASM 3.0;
qubit[3] q;
bit[3] c;
h q[0];
rx(0.41) q[0];
cx q[0], q[1];
rz(1.27) q[1];
rzz(0.41) q[1], q[2];
