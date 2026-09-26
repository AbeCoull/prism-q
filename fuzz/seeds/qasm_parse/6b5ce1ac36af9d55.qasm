OPENQASM 3.0;
qubit[2] q;
bit[2] c;
if (c[0]) {
  x q[0];
  if (c[1]) {
    x q[1];
    z q[1];
  }
}
