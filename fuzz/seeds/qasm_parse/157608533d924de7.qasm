OPENQASM 3.0;
qubit[1] q;
bit[1] c;
def mygate(qubit q) { measure q -> c[0]; }
