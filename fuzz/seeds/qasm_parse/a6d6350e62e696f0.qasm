OPENQASM 3.0;
qubit[2] a;
qubit[1] b;
bit[2] ca;
bit[1] cb;
h a[0];
cx a[0], a[1];
x b[0];
