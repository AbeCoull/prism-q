OPENQASM 3.0;
qubit[1] q;
bit[2] c;
switch (c) {
  case 1 { x q[0]; }
  case 1 { z q[0]; }
}
