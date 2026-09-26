OPENQASM 3.0;
include "stdgates.inc";
qubit[4] q;
def zz_layer(float gamma, qubit a, qubit b) {
    cx a, b;
    rz(gamma) b;
    cx a, b;
}
for int i in [0:3] {
    h q[i];
}
for int i in [0:2] {
    zz_layer(0b1 * 0.4, q[i], q[i + 1]);
}
for int i in [0:3] {
    rx(0.3) q[i];
}
