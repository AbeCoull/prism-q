// QuEST comparator for the cross-simulator comparison harness, driven through
// the same gate-list protocol as examples/compare_runner.rs:
//
//   quest_runner time <iterations>        (gate list on stdin)
//   quest_runner probabilities <out_path> (gate list on stdin)
//   quest_runner version
//
// The timed region covers register allocation, gate execution, and the
// probability read-out. OpenMP reads OMP_NUM_THREADS for the thread count.

#include <quest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

struct Op {
    char kind;  // h, x (cx), s (swap), y (ry), z (rz), p (cp)
    int a;
    int b;
    double angle;
};

struct Program {
    int numQubits = 0;
    std::vector<Op> ops;
};

[[noreturn]] void fail(int code, const std::string& message) {
    std::cerr << message << "\n";
    std::exit(code);
}

Program parseGateList(std::istream& in) {
    Program program;
    std::string line;
    bool header = false;
    while (std::getline(in, line)) {
        std::istringstream fields(line);
        std::string name;
        if (!(fields >> name)) continue;
        if (!header) {
            if (name != "qubits" || !(fields >> program.numQubits)) fail(3, "expected `qubits N`");
            header = true;
            continue;
        }
        Op op{};
        if (name == "h") {
            op.kind = 'h';
            fields >> op.a;
        } else if (name == "cx") {
            op.kind = 'x';
            fields >> op.a >> op.b;
        } else if (name == "swap") {
            op.kind = 's';
            fields >> op.a >> op.b;
        } else if (name == "ry") {
            op.kind = 'y';
            fields >> op.a >> op.angle;
        } else if (name == "rz") {
            op.kind = 'z';
            fields >> op.a >> op.angle;
        } else if (name == "cp") {
            op.kind = 'p';
            fields >> op.a >> op.b >> op.angle;
        } else {
            fail(3, "unsupported gate line `" + line + "`");
        }
        if (fields.fail()) fail(3, "malformed gate line `" + line + "`");
        program.ops.push_back(op);
    }
    if (!header) fail(3, "empty gate list");
    return program;
}

std::vector<double> run(const Program& program) {
    Qureg qureg = createQureg(program.numQubits);
    for (const Op& op : program.ops) {
        switch (op.kind) {
            case 'h': applyHadamard(qureg, op.a); break;
            case 'x': applyControlledPauliX(qureg, op.a, op.b); break;
            case 's': applySwap(qureg, op.a, op.b); break;
            case 'y': applyRotateY(qureg, op.a, op.angle); break;
            case 'z': applyRotateZ(qureg, op.a, op.angle); break;
            case 'p': applyTwoQubitPhaseShift(qureg, op.a, op.b, op.angle); break;
            default: fail(3, "unknown op");
        }
    }
    const qindex numAmps = qindex(1) << program.numQubits;
    std::vector<qcomp> amps(static_cast<size_t>(numAmps));
    getQuregAmps(amps.data(), qureg, 0, numAmps);
    std::vector<double> probs(static_cast<size_t>(numAmps));
    for (qindex i = 0; i < numAmps; ++i) probs[static_cast<size_t>(i)] = std::norm(amps[static_cast<size_t>(i)]);
    destroyQureg(qureg);
    return probs;
}

int threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

void printTimes(const Program& program, std::vector<double> times) {
    std::vector<double> sorted = times;
    std::sort(sorted.begin(), sorted.end());
    std::printf(
        "{\"schema\":\"prismq-compare-runner/2\",\"simulator\":\"quest\",\"version\":\"%s\","
        "\"num_qubits\":%d,\"num_operations\":%zu,\"threads\":\"%d\",\"median_ms\":%.4f,"
        "\"min_ms\":%.4f,\"times_ms\":[",
        QUEST_RUNNER_TAG, program.numQubits, program.ops.size(), threads(),
        sorted[sorted.size() / 2], sorted.front());
    for (size_t i = 0; i < times.size(); ++i) std::printf("%s%.4f", i ? "," : "", times[i]);
    std::printf("]}\n");
}

[[noreturn]] void usage() {
    fail(1,
         "Usage: quest_runner time <iterations>        (gate list on stdin)\n"
         "       quest_runner probabilities <out_path> (gate list on stdin)\n"
         "       quest_runner version");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) usage();
    const std::string command = argv[1];
    if (command == "version") {
        std::printf("{\"simulator\":\"quest\",\"version\":\"%s\"}\n", QUEST_RUNNER_TAG);
        return 0;
    }
    if (argc != 3) usage();
    initQuESTEnv();
    const Program program = parseGateList(std::cin);
    if (command == "time") {
        const int iterations = std::atoi(argv[2]);
        if (iterations <= 0) usage();
        volatile double sink = run(program)[0];
        std::vector<double> times;
        for (int i = 0; i < iterations; ++i) {
            const auto start = std::chrono::steady_clock::now();
            std::vector<double> probs = run(program);
            times.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
            sink = probs[0];
        }
        (void)sink;
        printTimes(program, times);
    } else if (command == "probabilities") {
        std::vector<double> probs = run(program);
        std::ofstream out(argv[2], std::ios::binary);
        out.write(reinterpret_cast<const char*>(probs.data()),
                  static_cast<std::streamsize>(probs.size() * sizeof(double)));
        if (!out) fail(4, "failed to write the probability vector");
        std::string path;
        for (const char* c = argv[2]; *c; ++c) {
            if (*c == '\\' || *c == '"') path += '\\';
            path += *c;
        }
        std::printf(
            "{\"schema\":\"prismq-compare-runner/2\",\"simulator\":\"quest\",\"num_qubits\":%d,"
            "\"num_operations\":%zu,\"length\":%zu,\"dtype\":\"<f8\",\"path\":\"%s\"}\n",
            program.numQubits, program.ops.size(), probs.size(), path.c_str());
    } else {
        usage();
    }
    finalizeQuESTEnv();
    return 0;
}
