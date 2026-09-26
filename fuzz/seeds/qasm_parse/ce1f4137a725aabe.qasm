OPENQASM 3.0; qubit[2] q; gate flip(t) a, b { rx(t) a; cx a, b; } flip(pi) q[0], q[1];
