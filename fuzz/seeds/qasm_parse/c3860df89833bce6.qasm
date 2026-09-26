OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
bit[3] c;
h q[0];
for int i in [1:2] {
    cp(pi / (2 * i)) q[0], q[i];
}
h q[1];
cp(pi / 2) q[1], q[2];
h q[2];
