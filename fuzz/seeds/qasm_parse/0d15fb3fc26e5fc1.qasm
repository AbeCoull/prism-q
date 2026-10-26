OPENQASM 3.0;
qubit[1] q;
gate g0 a { x a; }
gate g1 a { g0 a; g0 a; }
gate g2 a { g1 a; g1 a; }
gate g3 a { g2 a; g2 a; }
gate g4 a { g3 a; g3 a; }
gate g5 a { g4 a; g4 a; }
gate g6 a { g5 a; g5 a; }
gate g7 a { g6 a; g6 a; }
gate g8 a { g7 a; g7 a; }
gate g9 a { g8 a; g8 a; }
gate g10 a { g9 a; g9 a; }
gate g11 a { g10 a; g10 a; }
gate g12 a { g11 a; g11 a; }
g12 q[0];
