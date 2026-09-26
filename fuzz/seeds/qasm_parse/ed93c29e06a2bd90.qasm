OPENQASM 3.0;
qubit[2] q;
h q[0];
defcal x $0 {
  shift_phase($1, 0.5);
}
