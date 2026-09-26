OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
def my_rx(float t, qubit a) {
    U(t, -pi / 2, pi / 2) a;
}
my_rx(pi, q[0]);
cx q[0], q[1];
