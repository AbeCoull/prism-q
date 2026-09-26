OPENQASM 3.0;
qubit[3] q;
crx(pi / 3) q[0], q[1];
cry(pi / 4) q[0], q[1];
crz(pi / 5) q[0], q[1];
cp(pi / 2) q[0], q[2];
ch q[1], q[2];
ccx q[0], q[1], q[2];
