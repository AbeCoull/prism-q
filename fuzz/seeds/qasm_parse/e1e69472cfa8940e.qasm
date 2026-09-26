OPENQASM 3.0;
qubit[3] q;
bit[2] c;
if (c == 2) {
  x q[2];
  measure q[2] -> c[1];
  reset q[0];
}
