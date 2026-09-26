OPENQASM 3.0;
qubit[3] q;
bit[2] c;
h q[0];
c[0] = measure q[0];
if (c[0]) {
  x q[1];
  if (!c[1]) {
    cx q[1], q[2];
  }
  reset q[2];
}
