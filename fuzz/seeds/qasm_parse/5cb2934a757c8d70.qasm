OPENQASM 3.0;
qubit[2] a;
qubit[3] b;
h a[0];
h b[2];
cx a[1], b[0];
