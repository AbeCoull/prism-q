OPENQASM 3.0;
qubit[2] q;
h q[0];
ecr q[0], q[1];
inv @ ecr q[0], q[1];
