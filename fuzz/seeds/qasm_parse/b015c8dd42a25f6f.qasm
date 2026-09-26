OPENQASM 3.0;
gate r(p0, p1) _gate_q_0 {
  U(p0, -pi/2 + p1, pi/2 - p1) _gate_q_0;
}
qubit[1] q;
r(pi/3, pi/7) q[0];
