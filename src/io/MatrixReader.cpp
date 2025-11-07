#include <string>
#include <filesystem>
#include <memory>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "MatrixReader.hpp"
#include "MatrixConverter.hpp"
#include "../format/Matrix.hpp"


#include <cstdint>
#if defined(_WIN32)
#include <windows.h>
size_t getAvailableRAM() {
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    return status.ullAvailPhys;
}
#elif defined(__linux__)
#include <sys/sysinfo.h>
size_t getAvailableRAM() {
    struct sysinfo info;
    if (sysinfo(&info) == 0)
        return (size_t)info.freeram * info.mem_unit;
    return 0;
}
#else
size_t getAvailableRAM() {
    return 0; // Fallback: unknown platform
}
#endif



MatrixMarketHeader readMatrixMarketHeader(const std::string &filename) {
    fs::path path = fs::absolute(filename);

    if (!fs::exists(path)) {
        throw std::runtime_error("Cannot open file: " + path.string());
    }

    std::ifstream infile(path);
    if (!infile.is_open()) {
        throw std::runtime_error("File exists but cannot be opened: " + path.string());
    }

    std::string line;
    std::istringstream stream;
    MatrixMarketHeader header;

    // Read Matrix Market banner/header line
    if (!std::getline(infile, line)) {
        throw std::runtime_error("Empty file or unable to read Matrix Market header.");
    }
    if (line.compare(0, 14, "%%MatrixMarket") != 0) {
        throw std::runtime_error("Invalid Matrix Market file: missing header line.");
    }

    std::string banner_line = line;
    std::transform(banner_line.begin(), banner_line.end(), banner_line.begin(), ::tolower);

    if (banner_line.find("pattern") != std::string::npos)
        header.isBinary = true;
    if (banner_line.find("symmetric") != std::string::npos)
        header.isSymmetric = true;

    // Skip comments starting with '%'
    do {
        if (!std::getline(infile, line)) {
            throw std::runtime_error("Unexpected end of file before header line.");
        }
    } while (!line.empty() && line[0] == '%');

    // Parse header line: rows cols nnz
    stream.clear();
    stream.str(line);
    stream >> header.rows >> header.cols >> header.nnz;
    if (stream.fail()) {
        throw std::runtime_error("Invalid header line in Matrix Market file.");
    }

    return header;
}



// get number of rows from .sparse.rw.bin
uint32_t getRowsFromFile(const std::string &rowFile) {
    std::ifstream in(rowFile, std::ios::binary | std::ios::ate);
    if (!in) {
        throw std::runtime_error("Failed to open file: " + rowFile);
    }
    std::streamsize size = in.tellg();
    if (size % 4 != 0) {
        throw std::runtime_error("Invalid .sparse.rw.bin file size");
    }
    return static_cast<uint32_t>(size / 4);
}

// get number of cols from .sparse.cw.bin
uint32_t getColsFromFile(const std::string &colFile) {
    std::ifstream in(colFile, std::ios::binary | std::ios::ate);
    if (!in) {
        throw std::runtime_error("Failed to open file: " + colFile);
    }
    std::streamsize size = in.tellg();
    if (size % 4 != 0) {
        throw std::runtime_error("Invalid .sparse.cw.bin file size");
    }
    return static_cast<uint32_t>(size / 4);
}


// replace ".sparse.bin" with suffix
std::string deriveFileName(const std::string &filename, const std::string &suffix) {
    const std::string token = ".sparse.bin";
    if (filename.size() <= token.size() ||
        filename.substr(filename.size() - token.size()) != token) {
        throw std::runtime_error("Filename does not end with .sparse.bin: " + filename);
    }
    return filename.substr(0, filename.size() - token.size()) + ".sparse." + suffix;
}


MatrixCOO *readMatrixBinCado(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    std::string rowFile = deriveFileName(filename, "rw.bin");
    std::string colFile = deriveFileName(filename, "cw.bin");
    uint32_t numRows = getRowsFromFile(rowFile);
    uint32_t numCols = getColsFromFile(colFile);

    MatrixCOO *matrix = new MatrixCOO();
    matrix->rows = numRows;
    matrix->cols = numCols;
    matrix->nnz = 0U;

    for (uint32_t i = 0; i < numRows; ++i) {
        uint32_t count;
        in.read(reinterpret_cast<char*>(&count), sizeof(uint32_t));
        if (!in) {
            throw std::runtime_error("Unexpected EOF reading row length");
        }

        std::vector<uint32_t> cols(count);
        if (count > 0) {
            in.read(reinterpret_cast<char*>(cols.data()), count * sizeof(uint32_t));
            if (!in) {
                throw std::runtime_error("Unexpected EOF reading row data");
            }

            std::sort(cols.begin(), cols.end()); // mimic qsort from C

            for (uint32_t col : cols) {
                matrix->rowIdx.push_back(i + 1);  // binary matrices are generated using 0-based indexing
                matrix->colIdx.push_back(col + 1); 
            }
            matrix->nnz += count;
        }
    }

    return matrix;
}



void sortByRow(MatrixCOO *matrix) {
    size_t nnz = matrix->rowIdx.size();
    std::vector<uint32_t> sortedRowIdx(nnz);
    std::vector<uint32_t> sortedColIdx(nnz);

    if (matrix->colIdx.size() != nnz) {
        throw std::runtime_error("MatrixCOO: inconsistent array sizes.");
    }

    // Create index vector [0, 1, 2, ..., nnz-1]
    std::vector<size_t> idx(nnz);
    for (size_t i = 0; i < nnz; ++i) {
        idx[i] = i;
    }

    // Sort the indices based on row (then col)
    std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
        return std::tie(matrix->rowIdx[a], matrix->colIdx[a]) < std::tie(matrix->rowIdx[b], matrix->colIdx[b]);
    });

    // Apply the sorted order


    for (size_t i = 0; i < nnz; ++i) {
        size_t idxSize = idx[i];
        sortedRowIdx[i] = matrix->rowIdx[idxSize];
        sortedColIdx[i] = matrix->colIdx[idxSize];
    }

    matrix->rowIdx = std::move(sortedRowIdx);
    matrix->colIdx = std::move(sortedColIdx);
    matrix->isRowMajor = true;
}


MatrixCOO *readMatrixMarket(const std::string &filename)
{
    fs::path path;
    std::ifstream infile;
    std::string line;
    std::istringstream stream;
    std::uint32_t rows = 0, cols = 0, nnz = 0;
    std::uint32_t i = 0, j = 0;
    double val = 0.0;
    bool isBinary = false;
    bool isSymmetric = false;
    MatrixCOO *matrix = new MatrixCOO();

    // Open the file using absolute path
    path = fs::absolute(filename);
    if (!fs::exists(path))
    {
        throw std::runtime_error("Cannot open file: " + path.string());
    }

    infile.open(path);
    if (!infile.is_open())
    {
        throw std::runtime_error("File exists but cannot be opened: " + path.string());
    }

    // Read the Matrix Market banner/header line starting with "%%MatrixMarket"
    if (!std::getline(infile, line))
        throw std::runtime_error("Empty file or unable to read Matrix Market header.");

    if (line.compare(0, 14, "%%MatrixMarket") != 0)
        throw std::runtime_error("Invalid Matrix Market file: missing header line.");

    std::string banner_line = line;
    std::transform(banner_line.begin(), banner_line.end(), banner_line.begin(), ::tolower);

    if (banner_line.find("pattern") != std::string::npos)
        isBinary = true;

    if (banner_line.find("symmetric") != std::string::npos)
        isSymmetric = true;

    // Skip comments starting with '%'
    do
    {
        if (!std::getline(infile, line))
            throw std::runtime_error("Unexpected end of file before header line.");
    } while (!line.empty() && line[0] == '%');

    // Parse header line: rows cols nnz
    stream.clear();
    stream.str(line);
    stream >> rows >> cols >> nnz;
    if (stream.fail())
        throw std::runtime_error("Invalid header line in Matrix Market file.");

    // Construct matrix
    matrix->rows = rows;
    matrix->cols = cols;

    // --- Memory safety check ---
    size_t expected_entries = isSymmetric ? 2ull * nnz : nnz;
    size_t bytes_needed = expected_entries * sizeof(uint32_t) * 2; // row + col

    bytes_needed += expected_entries * sizeof(double);

    size_t available = getAvailableRAM();
    if (available > 0 && bytes_needed > available) {
        throw std::runtime_error("Matrix too large: needs " +
                                std::to_string(bytes_needed / (1024*1024)) + " MB, " +
                                "but only " + std::to_string(available / (1024*1024)) + " MB available.");
    }


    // Reserve memory; if symmetric, we may double nnz for off-diagonal entries
    matrix->rowIdx.reserve(isSymmetric ? 2 * nnz : nnz);
    matrix->colIdx.reserve(isSymmetric ? 2 * nnz : nnz);

    // Read entries
    for (uint32_t k = 0; k < nnz; ++k)
    {
        if (!std::getline(infile, line))
            throw std::runtime_error("Unexpected end of file reading entries.");

        stream.clear();
        stream.str(line);

        if (isBinary)
        {
            if (!(stream >> i >> j))
                throw std::runtime_error("Invalid entry line: '" + line + "'");
            val = 1.0;
        }
        else
        {
            if (!(stream >> i >> j >> val))
                throw std::runtime_error("Invalid entry line: '" + line + "'");
        }

        // Store the entry (1-based indexing)
        matrix->rowIdx.push_back(i);
        matrix->colIdx.push_back(j);

        // If symmetric and off-diagonal, store the mirrored entry
        if (isSymmetric && i != j)
        {
            matrix->rowIdx.push_back(j);
            matrix->colIdx.push_back(i);
        }
    }

    // Update nnz
    matrix->nnz = matrix->rowIdx.size();

    // Sort by row
    sortByRow(matrix);

    return matrix;
}



MatrixBase *MatrixReader::readMatrixFromFile(const std::string& filename, MatrixFormat format){
    auto dotPos = filename.find_last_of('.');
    if (dotPos == std::string::npos)
        throw std::runtime_error("Malformed file: " + filename);

    std::string ext = filename.substr(dotPos + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    MatrixCOO* matrixCoo = nullptr;

    if (ext == "mtx") {
        matrixCoo = readMatrixMarket(filename);
    } else if (ext == "bin") {
        matrixCoo = readMatrixBinCado(filename);
    } else {
        throw std::runtime_error("Unsupported file extension: " + ext);
    }

    if (!matrixCoo) {
        throw std::runtime_error("Failed to read matrix from file: " + filename);
    }

    // Convert COO to the requested format
    return MatrixConverter::convert(matrixCoo, format);
}