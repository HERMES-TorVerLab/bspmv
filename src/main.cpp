#include <iostream>
#include <cmath>
#include <iomanip> 
#include "format/Matrix.hpp"
#include "kernel/MatrixKernel.hpp"
#include "utils/bspmvUtils.hpp"

void printHelp(const char* progName) {
    std::cout << "Usage: " << progName
              << " --matrix-file=<file> --matrix-format=<COO|CSR|ELL|HLL|BWC_COO|BWC_CSR|BWC_ELL|BWC_HLL> --exec-mode=<CPU|GPU>\n\n"
              << "Arguments:\n"
              << "  --matrix-file     Path to the Matrix Market (.mtx) file\n"
              << "  --matrix-format   Matrix format (COO, CSR, ELL, HLL, BWC_COO, BWC_CSR, BWC_ELL, BWC_HLL)\n"
              << "  --exec-mode       Execution mode (CPU or GPU)\n\n"
              << "Optional GPU parameters:\n"
              << "  --threads-per-block=<int>   (default: 512)\n"
              << "  --warp-size=<int>           (default: 32)\n"
              << "  --kernel-launches=<int>     (default: 100)\n\n"
              << "Example:\n"
              << "  " << progName
              << " --matrix-file=../bcsstk29.mtx --matrix-format=ELL --exec-mode=GPU "
                 "--threads-per-block=512 --warp-size=4 --kernel-launches=100\n";
}

std::string getOptionValue(const std::string& arg, const std::string& prefix) {
    if (arg.find(prefix) == 0) {
        return arg.substr(prefix.size());
    }
    return "";
}



int main(int argc, char* argv[]) {
    Matrix *matrix;
    uint32_t rows, cols, nnz;
    double mem;
    try {
        if (argc < 2) {
            printHelp(argv[0]);
            return 1;
        }
        std::cout << R"(
=========================================================
         _      _____       ___  ___ __       __
        | |    / ____/     |   \/   |\ \     / /
        | |__ | (___  ____ | |\  /| | \ \   / / 
        | '_ \ \___ \|  _ \| | \/ | |  \ \ / /  
        | |_) |____) | |_| | |    | |   \   /
        |_.__/|_____/|  __/|_|    |_|    \_/ 
                     | |
                     |_|

bSpMV v1.0 - Binary Sparse Matrix-Vector Multiplication
=========================================================
)" << "\n";

        std::string filename, formatStr, execStr;
        int threadsPerBlock = 512;  // default
        int warpSize        = 4;    // default
        int kernelLaunches  = 100;  // default

        // Parse long options
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--help" || arg == "-h") {
                printHelp(argv[0]);
                return 0;
            }

            if (arg.find("--matrix-file=") == 0) {
                filename = getOptionValue(arg, "--matrix-file=");
            } else if (arg.find("--matrix-format=") == 0) {
                formatStr = getOptionValue(arg, "--matrix-format=");
            } else if (arg.find("--exec-mode=") == 0) {
                execStr = getOptionValue(arg, "--exec-mode=");
            } else if (arg.find("--threads-per-block=") == 0) {
                threadsPerBlock = std::stoi(getOptionValue(arg, "--threads-per-block="));
            } else if (arg.find("--warp-size=") == 0) {
                warpSize = std::stoi(getOptionValue(arg, "--warp-size="));
            } else if (arg.find("--kernel-launches=") == 0) {
                kernelLaunches = std::stoi(getOptionValue(arg, "--kernel-launches="));
            } else {
                throw std::invalid_argument("Unknown argument: " + arg);
            }
        }

        if (filename.empty() || formatStr.empty() || execStr.empty()) {
            std::cerr << "Error: Missing required arguments.\n\n";
            printHelp(argv[0]);
            return 1;
        }

        // Validate file existence
        if (!std::filesystem::exists(filename)) {
            throw std::runtime_error("Matrix file does not exist: " + filename);
        }

        // Parse matrix format
        MatrixFormat format;
        if (formatStr == "COO")      format = MatrixFormat::COO;
        else if (formatStr == "CSR") format = MatrixFormat::CSR;
        else if (formatStr == "ELL") format = MatrixFormat::ELL;
        else if (formatStr == "HLL") format = MatrixFormat::HLL;
        else if (formatStr == "BWC_COO") format = MatrixFormat::BWC_COO;
        else if (formatStr == "BWC_CSR") format = MatrixFormat::BWC_CSR;
        else if (formatStr == "BWC_ELL") format = MatrixFormat::BWC_ELL;
        else if (formatStr == "BWC_HLL") format = MatrixFormat::BWC_HLL;
        else throw std::invalid_argument("Unknown format: " + formatStr);
        

        // Parse execution mode
        ExecutionMode execMode;
        if (execStr == "CPU")      execMode = ExecutionMode::CPU;
        else if (execStr == "GPU") execMode = ExecutionMode::GPU;
        else throw std::invalid_argument("Unknown execution mode: " + execStr);


        // --- Kernel and execution parameters ---
        std::cout << "========== Kernel Info ==========\n";
        std::cout << "[INFO]\t" << std::left << std::setw(20) << "Matrix file:"        << filename << "\n";
        std::cout << "[INFO]\t" << std::left << std::setw(20) << "Matrix format:"      << formatStr << "\n";
        std::cout << "[INFO]\t" << std::left << std::setw(20) << "Execution mode:"     << execStr << "\n";
        std::cout << "[INFO]\t" << std::left << std::setw(20) << "Threads per block:"  << threadsPerBlock << "\n";
        std::cout << "[INFO]\t" << std::left << std::setw(20) << "Warp size:"          << warpSize << "\n";
        std::cout << "[INFO]\t" << std::left << std::setw(20) << "Kernel launches:"    << kernelLaunches << "\n";
        std::cout << "=================================\n";

        // Create matrix
        matrix = new Matrix(filename, format);
        rows = matrix->getRows();
        cols = matrix->getCols();
        nnz = matrix->getNonZeros();
        mem = matrix->getMemoryOccupationMB();

        // --- Matrix info ---
        std::cout << "========== Matrix Info ==========\n";
        std::cout << "[INFO]\t" << std::left << std::setw(20) << "Rows:"        << rows << "\n";
        std::cout << "[INFO]\t" << std::left << std::setw(20) << "Cols:"        << cols << "\n";
        std::cout << "[INFO]\t" << std::left << std::setw(20) << "Non-zeros:"   << nnz << "\n";

        // Print memory occupancy
        std::cout << "[INFO]\t" << std::left << std::setw(20) << "Memory (MB):" << mem << "\n";

        if (matrix->getNumWord() > 0) {
            std::cout << "[INFO]\t" << std::left << std::setw(20) << "Num words:"  << matrix->getNumWord() << "\n";
            std::cout << "[INFO]\t" << std::setw(20) << "Min NNZ/word:" << matrix->getMinNnzWord() << "\n";
            std::cout << "[INFO]\t" << std::setw(20) << "Max NNZ/word:" << matrix->getMaxNnzWord() << "\n";
            std::cout << "[INFO]\t" << std::setw(20) << "Avg NNZ/word:" << matrix->getAvgNnzWord() << "\n";
        }

        std::cout << "=================================\n";


        // matrix->print();

        // Allocate vectors
        std::vector<uint32_t>* x = new std::vector<uint32_t>(
            std::ceil(matrix->getCols() / 32.0), 286331153); // 10001000100010001000100010001000
        std::vector<uint32_t>* y = new std::vector<uint32_t>(
            std::ceil(matrix->getRows() / 32.0), 0);

        // Run SpMV (pass GPU parameters)
        if(matrix->getMemoryOccupationMB() < 3.0 * 1024.0)
            spmv(matrix, x, y, execMode);
        else
            std::cout << "Matrix is too big for selected GPU\n";

        // printVectorBinary32(*y);
        std::cout << "\nResult vector written on file\n";

        delete matrix;
        delete x;
        delete y;

    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n";
        return 1;
    }

    return 0;
}

