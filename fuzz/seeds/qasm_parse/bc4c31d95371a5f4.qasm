OPENQASM 3.0;
qubit[1] q;
def my_rx(float theta, qubit a) {
    rx(theta) a;
}
my_rx(0.5, q[0]);
