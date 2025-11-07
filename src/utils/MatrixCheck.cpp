#include "MatrixCheck.hpp"

bool parseCooRow(const std::string &line, std::set<uint32_t> &cols) {
    cols.clear();
    std::stringstream ss(line);
    std::string token;

    while (ss >> token) {
        if (token.front() == '(' && token.back() == ')') {
            size_t comma = token.find(',');
            if (comma != std::string::npos) {
                uint32_t col = std::stoi(token.substr(comma + 1, token.size() - comma - 2));
                cols.insert(col);
            }
        }
    }

    return !cols.empty();
}

bool parseBwcCooRow(const std::vector<std::string> &lines, std::set<uint32_t> &cols) {
    cols.clear();
    for (const auto &line : lines) {
        if (line.empty()) continue;
        // line format: "(row=..., wordCol=...) <bits>"
        size_t pos = line.find(')');
        if (pos == std::string::npos) continue;
        std::string bits = line.substr(pos + 2); // skip ") "
        if (bits.size() != 32) continue; // assuming 32-bit words

        for (size_t i = 0; i < bits.size(); ++i) {
            if (bits[i] == '1') {
                // column = wordCol*32 + bit position (MSB = col1)
                size_t bitPos = 31 - i;
                uint32_t col = bitPos + 1; // start with 1-based col
                cols.insert(col);
            }
        }
    }
    return !cols.empty();
}



// ---------------- BWC-COO checker ----------------
bool checkBwcCoo(const std::string &baselineFile, const std::string &compareFile) {
    std::ifstream cooFile(baselineFile);
    std::ifstream bwcFile(compareFile);

    if (!cooFile.is_open() || !bwcFile.is_open()) {
        std::cerr << "Error opening files.\n";
        return false;
    }

    std::string cooLine, bwcLine;
    std::set<uint32_t> cooCols, bwcCols;
    std::vector<std::string> bwcBlock;
    uint32_t row = 0;

    while (std::getline(cooFile, cooLine)) {
        if (cooLine.find("ROW:") != std::string::npos) {
            row++;
            bwcBlock.clear();
            while (std::getline(bwcFile, bwcLine)) {
                if (bwcLine.find("ROW ") != std::string::npos && !bwcBlock.empty()) {
                    bwcFile.seekg(-static_cast<int>(bwcLine.size()) - 1, std::ios_base::cur);
                    break;
                }
                if (!bwcLine.empty() && bwcLine.find("(row=") != std::string::npos)
                    bwcBlock.push_back(bwcLine);
            }

            if (!parseCooRow(cooLine, cooCols)) continue;
            if (!parseBwcCooRow(bwcBlock, bwcCols)) {
                std::cerr << "Error parsing BWC_COO row " << row << "\n";
                return false;
            }

            if (cooCols != bwcCols) {
                std::cerr << "Mismatch at row " << row << "\n";
                std::cerr << "COO columns: ";
                for (auto c : cooCols) std::cerr << c << " ";
                std::cerr << "\nBWC columns: ";
                for (auto c : bwcCols) std::cerr << c << " ";
                std::cerr << "\n";
                return false;
            }
        }
    }

    std::cout << "Files match!\n";
    return true;
}

// ---------------- Matrixcheck methods implementations ----------------
MatrixCheck::MatrixCheck() {
    
}

bool MatrixCheck::check(const std::string &baselineFileName, MatrixFormat baselineFormat,
                        const std::string &compareFileName, MatrixFormat compareFormat) {
    switch (baselineFormat) {
        case MatrixFormat::COO:
            switch (compareFormat) {
                case MatrixFormat::BWC_COO:
                    return checkBwcCoo(baselineFileName, compareFileName);
                case MatrixFormat::CSR:
                    // TODO: implement checkCooVsCsr
                    std::cerr << "COO vs CSR check not implemented.\n";
                    return false;
                case MatrixFormat::ELL:
                    std::cerr << "COO vs ELL check not implemented.\n";
                    return false;
                case MatrixFormat::HLL:
                    std::cerr << "COO vs HLL check not implemented.\n";
                    return false;
                case MatrixFormat::BWC_CSR:
                    std::cerr << "COO vs BWC_CSR check not implemented.\n";
                    return false;
                case MatrixFormat::BWC_ELL:
                    std::cerr << "COO vs BWC_ELL check not implemented.\n";
                    return false;
                case MatrixFormat::BWC_HLL:
                    std::cerr << "COO vs BWC_HLL check not implemented.\n";
                    return false;
                default:
                    std::cerr << "Unknown compare format.\n";
                    return false;
            }
            break;
        // Add other baseline formats here if needed
        default:
            std::cerr << "Unsupported baseline format.\n";
            return false;
    }
}