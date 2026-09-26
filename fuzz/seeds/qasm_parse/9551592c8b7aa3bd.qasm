OPENQASM 3.0;
qubit[5] q;
bit[2] c;
x q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
switch (c) {
  case 0 { x q[2]; }
  case 2 { x q[3]; }
  default { x q[4]; }
}
