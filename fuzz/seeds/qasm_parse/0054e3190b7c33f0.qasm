OPENQASM 3.0;
qubit[4] q;
bit[4] c;
const int width = 4;
float theta = pi / 8;
int cursor = 0;
for int k in [0:width - 1] { rx(theta * k) q[k]; }
cursor += 2;
h q[cursor];
