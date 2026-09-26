OPENQASM 3.0;
qubit[2] q;
gate myrot(a) p, r { rz(a) p; cx p, r; ry(a) r; }
h q[0];
h q[1];
myrot(0.7) q[0], q[1];
inv @ myrot(0.7) q[0], q[1];
