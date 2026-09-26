OPENQASM 3.0;
bit[1] c;
h $0;
cx $0, $3;
c[0] = measure $3;
