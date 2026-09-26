OPENQASM 3.0;
qubit[1] q;
bit[2] c;
switch (c) {
  case 1 { measure q[0] -> c[0]; }
  case 2 { x q[0]; }
}
