import os

# The stabilizer GPU crossover defaults to 100000 qubits and is cached on first
# read, so a device stabilizer case has to lower it before any simulation runs.
# conftest is imported ahead of every test module, which a module-scope write in
# one of them would not guarantee.
os.environ.setdefault("PRISM_STABILIZER_GPU_MIN_QUBITS", "0")
