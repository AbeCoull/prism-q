OPENQASM 3.0;
bit[2] c;
h $0;
cx $0, $1;
c[0] = measure $0;
c[1] = measure $1;
