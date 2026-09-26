OPENQASM 3.0; qubit[2] q; bit[2] c; measure q[0] -> c[0]; if (c[0]) { x q[1]; } else { y q[1]; }
