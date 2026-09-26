OPENQASM 2.0;
qreg q[2];
creg c[2];
x q[0];
measure q[0] -> c[0];
if(c==2) x q[1];
