OPENQASM 3.0;
qubit[1] q;
bit[4] c;
def cond(int n, qubit a) {
    if (c == n) x a;
}
cond(3, q[0]);
