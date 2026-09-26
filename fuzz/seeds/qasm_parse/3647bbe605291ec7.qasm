OPENQASM 3.0;
qubit[4] q;
bit[2] c;
switch (c) {
  case 0 { x q[0]; cx q[0], q[1]; }
  case 1, 2 { x q[2]; }
  default { x q[3]; z q[3]; }
}
