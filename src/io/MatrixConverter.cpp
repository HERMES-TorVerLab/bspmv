/**
 * @file MatrixConverter.cpp
 * @author Stack1
 * @brief 
 * @version 1.0
 * @date 19-09-2025
 * 
 * 
 */
#include <map> 
#include "MatrixConverter.hpp"
#include "../format/Matrix.hpp"

namespace fs = std::filesystem;


MatrixCSR *convertToCSR(const MatrixCOO *matrixCOO) {
    MatrixCSR *matrixCSR = new MatrixCSR();
    uint32_t rows = matrixCOO->rows;
    uint32_t cols = matrixCOO->cols;
    uint32_t nnz  = matrixCOO->nnz;


    matrixCSR->rows = rows;
    matrixCSR->cols = cols;
    matrixCSR->nnz = nnz;

    if (matrixCOO->colIdx.size() != nnz) {
        throw std::runtime_error("Invalid COO matrix: mismatched array sizes.");
    }
    
    matrixCSR->rowPtr.resize(rows + 1);
    matrixCSR->colIdx.resize(nnz);

    // Step 1: Count non-zeros per row
    for (uint32_t i = 0; i < nnz; ++i) {
        matrixCSR->rowPtr[matrixCOO->rowIdx[i]]++;
    }

    // Step 2: Cumulative sum to get rowPtr
    for (uint32_t i = 1; i <= rows; ++i) {
        matrixCSR->rowPtr[i] += matrixCSR->rowPtr[i - 1];
    }


    // Step 3: Fill colIdx and values
    for (uint32_t i = 0; i < nnz; ++i) {
        matrixCSR->colIdx[i] = matrixCOO->colIdx[i];
    }

    return matrixCSR;
}

MatrixELL *convertToELL(const MatrixCOO *matrixCOO) {
    MatrixELL *matrixELL = new MatrixELL();
    uint32_t lastRow, colCount;
    uint32_t rows = matrixCOO->rows;
    uint32_t cols = matrixCOO->cols;
    uint32_t nnz  = matrixCOO->nnz;

    matrixELL->rows = rows;
    matrixELL->cols = cols;
    matrixELL->nnz  = nnz;

    // Step 1: Count nonzeros per row
    std::vector<uint32_t> rowCounts(rows, 0);
    for (uint32_t i = 0; i < nnz; ++i) {
        rowCounts[matrixCOO->rowIdx[i] - 1]++;  // assuming COO is 1-based
    }

    // Step 2: Find maximum nonzeros per row (ELL parameter k)
    uint32_t k = 0;
    for (uint32_t r = 0; r < rows; ++r) {
        if (rowCounts[r] > k) {
            k = rowCounts[r];
        }
    }
    matrixELL->k = k;

    // Step 3: Allocate storage (column-major layout)
    matrixELL->colIdx.assign(rows * k, 0);  // padded with 0 (sentinel)

    // Step 4: Build per-row lists of col indices
    colCount = 0U;
    lastRow = 0U;
    for (uint32_t i = 0; i < nnz; ++i) {
        uint32_t row = matrixCOO->rowIdx[i] - 1; 
        uint32_t col = matrixCOO->colIdx[i] - 1;     // keep 1-based if your SpMV expects it
        if(lastRow < row){
            colCount = 0U;
            lastRow = row;
        }
        matrixELL->colIdx[colCount * rows + row] = col + 1;
        colCount++;
    }

    return matrixELL;
}


MatrixHLL *convertToHLL(const MatrixCOO *matrixCOO) {
    MatrixHLL *matrixHLL = new MatrixHLL();
    uint32_t rows = matrixCOO->rows;
    uint32_t cols = matrixCOO->cols;
    uint32_t nnz  = matrixCOO->nnz;
    uint32_t hackSize = 4;

    matrixHLL->rows = rows;
    matrixHLL->cols = cols;
    matrixHLL->nnz  = nnz;
    matrixHLL->hackSize = hackSize;

    // Step 1: Count nonzeros per row
    std::vector<uint32_t> rowCounts(rows, 0);
    for (uint32_t i = 0; i < nnz; ++i) {
        rowCounts[matrixCOO->rowIdx[i] - 1]++;
    }

    // Step 2: Determine number of hacks
    uint32_t numHacks = (rows + hackSize - 1) / hackSize;
    matrixHLL->hack.resize(numHacks + 1, 0); // hack-wise offset

    // Step 3: For each hack, determine max nonzeros in its rows
    std::vector<uint32_t> hackWidths(numHacks, 0);
    for (uint32_t h = 0; h < numHacks; ++h) {
        uint32_t start = h * hackSize;
        uint32_t end = std::min(start + hackSize, rows);
        uint32_t maxNNZ = 0;
        for (uint32_t r = start; r < end; ++r) {
            if (rowCounts[r] > maxNNZ)
                maxNNZ = rowCounts[r];
        }
        hackWidths[h] = maxNNZ;
        matrixHLL->hack[h + 1] = matrixHLL->hack[h] + maxNNZ * (end - start);
    }

    // Step 4: Allocate colIdx array (column-major inside each hack)
    matrixHLL->colIdx.assign(matrixHLL->hack.back(), 0); // 0 = padding sentinel

    // Temp counter for each row to track position
    std::vector<uint32_t> currentPos(rows, 0);

    // Step 5: Fill in column indices in column-major per hack
    for (uint32_t i = 0; i < nnz; ++i) {
        uint32_t row = matrixCOO->rowIdx[i] - 1;
        uint32_t col = matrixCOO->colIdx[i]; // remain 1-based if needed

        uint32_t hack = row / hackSize;
        uint32_t rowInHack = row % hackSize;
        uint32_t hackRowStart = hack * hackSize;

        uint32_t offset = matrixHLL->hack[hack];
        uint32_t index = currentPos[row] * (std::min(hackSize, rows - hackRowStart)) + rowInHack;

        matrixHLL->colIdx[offset + index] = col;
        currentPos[row]++;
    }

    return matrixHLL;
}

MatrixBwcCoo *convertToBwcCoo(const MatrixCOO *matrixCOO) {
    MatrixBwcCoo *matrixBwcCoo = new MatrixBwcCoo();
    uint32_t row, col, wordCol, bit, nnzWord;
    uint32_t minNnzWord = std::numeric_limits<uint32_t>::max();
    uint32_t maxNnzWord = 0;
    uint64_t sumNnzWord = 0;
    uint32_t rows = matrixCOO->rows;
    uint32_t cols = matrixCOO->cols;
    uint32_t nnz  = matrixCOO->nnz;
    std::map<std::pair<uint32_t,uint32_t>, uint32_t> acc;

    matrixBwcCoo->rows = rows;
    matrixBwcCoo->cols = cols;
    matrixBwcCoo->nnz  = nnz;
    matrixBwcCoo->numWord = 0U;

    for (size_t i = 0; i < matrixCOO->rowIdx.size(); ++i) {
        row = matrixCOO->rowIdx[i];
        col = matrixCOO->colIdx[i];

        wordCol = (col - 1) >> 5;   
        bit     = (col - 1) & 31U; 

        auto key = std::make_pair(row, wordCol);

        acc[key] |= (1U << bit);
    }

    // copy from map into output arrays
    matrixBwcCoo->rowIdx.reserve(acc.size());
    matrixBwcCoo->colIdx.reserve(acc.size());
    matrixBwcCoo->word.reserve(acc.size());
    matrixBwcCoo->numWord = acc.size();

    for (auto &[key, word] : acc) {
        matrixBwcCoo->rowIdx.push_back(key.first);
        matrixBwcCoo->colIdx.push_back(key.second + 1);
        matrixBwcCoo->word.push_back(word);

        // compute popcount
        nnzWord = popcount32(word);
        minNnzWord = std::min(minNnzWord, nnzWord);
        maxNnzWord = std::max(maxNnzWord, nnzWord);
        sumNnzWord += nnzWord;
    }

    // Store statistics (guard empty case)
    if (matrixBwcCoo->numWord > 0) {
        matrixBwcCoo->minNnzWord = minNnzWord;
        matrixBwcCoo->maxNnzWord = maxNnzWord;
        matrixBwcCoo->avgNnzWord = static_cast<double>(sumNnzWord) / matrixBwcCoo->numWord;
    } else {
        matrixBwcCoo->minNnzWord = 0;
        matrixBwcCoo->maxNnzWord = 0;
        matrixBwcCoo->avgNnzWord = 0.0;
    }

    return matrixBwcCoo;
}


MatrixBwcCsr *convertToBwcCsr(const MatrixCOO *matrixCOO) {
    MatrixBwcCsr *matrixBwcCsr = new MatrixBwcCsr();
    uint32_t row, col, wordCol, bit, nnzWord;
    uint32_t minNnzWord = std::numeric_limits<uint32_t>::max();
    uint32_t maxNnzWord = 0;
    uint64_t sumNnzWord = 0;
    uint32_t rows = matrixCOO->rows;
    uint32_t cols = matrixCOO->cols;
    uint32_t nnz  = matrixCOO->nnz;

    matrixBwcCsr->rows = rows;
    matrixBwcCsr->cols = cols;
    matrixBwcCsr->nnz  = nnz;

    // Temporary accumulator
    std::map<std::pair<uint32_t, uint32_t>, uint32_t> acc;

    for (size_t i = 0; i < matrixCOO->rowIdx.size(); ++i) {
        row = matrixCOO->rowIdx[i];
        col = matrixCOO->colIdx[i];

        if (row == 0 || col == 0) {
            throw std::runtime_error("COO row/col are 1-based, please confirm!");
        }

        if (row > rows) {
            throw std::runtime_error("Row index out of bounds");
        }
        if (col > cols) {
            throw std::runtime_error("Col index out of bounds");
        }

        wordCol = (col - 1) >> 5;   // col / 32
        bit     = (col - 1) & 31U;  // col % 32

        auto key = std::make_pair(row, wordCol);
        acc[key] |= (1U << bit);
    }

    // Reserve storage
    matrixBwcCsr->colIdx.reserve(acc.size());
    matrixBwcCsr->word.reserve(acc.size());
    matrixBwcCsr->rowPtr.assign(rows + 1, 0);
    matrixBwcCsr->numWord = acc.size();

    // Bucket by row
    std::vector<std::vector<std::pair<uint32_t, uint32_t>>> rowBuckets(rows);
    for (auto &[key, word] : acc) {
        uint32_t row     = key.first;
        uint32_t wordCol = key.second + 1;
        rowBuckets[row - 1].emplace_back(wordCol, word);

        nnzWord = popcount32(word);
        minNnzWord = std::min(minNnzWord, nnzWord);
        maxNnzWord = std::max(maxNnzWord, nnzWord);
        sumNnzWord += nnzWord;
    }

    // Fill CSR arrays
    uint32_t counter = 0;
    for (uint32_t r = 0; r < rows; ++r) {
        matrixBwcCsr->rowPtr[r] = counter;

        for (auto &[wordCol, word] : rowBuckets[r]) {
            matrixBwcCsr->colIdx.push_back(wordCol);
            matrixBwcCsr->word.push_back(word);
            ++counter;
        }
    }
    matrixBwcCsr->rowPtr[rows] = counter;

    if (matrixBwcCsr->numWord > 0) {
        matrixBwcCsr->minNnzWord = minNnzWord;
        matrixBwcCsr->maxNnzWord = maxNnzWord;
        matrixBwcCsr->avgNnzWord = static_cast<double>(sumNnzWord) / matrixBwcCsr->numWord;
    } else {
        matrixBwcCsr->minNnzWord = 0;
        matrixBwcCsr->maxNnzWord = 0;
        matrixBwcCsr->avgNnzWord = 0.0;
    }


    return matrixBwcCsr;
}


MatrixBwcEll *convertToBwcEll(const MatrixCOO *matrixCOO) {
    MatrixBwcEll *matrixBwcEll = new MatrixBwcEll();
    uint32_t row, col, wordCol, bit, idx, w, nnzWord;
    uint32_t minNnzWord = std::numeric_limits<uint32_t>::max();
    uint32_t maxNnzWord = 0;
    uint64_t sumNnzWord = 0;
    uint32_t rows = matrixCOO->rows;
    uint32_t cols = matrixCOO->cols;
    uint32_t nnz  = matrixCOO->nnz;


    matrixBwcEll->rows = rows;
    matrixBwcEll->cols = cols;
    matrixBwcEll->nnz  = nnz;
    matrixBwcEll->k = 0;
    matrixBwcEll->numWord = 0;

    // Build per-row maps
    std::vector<std::map<uint32_t, uint32_t>> rowWordMaps(rows);
    for (uint32_t i = 0; i < nnz; ++i) {
        row = matrixCOO->rowIdx[i] - 1;
        col = matrixCOO->colIdx[i] - 1;
        wordCol = col >> 5;
        bit     = col & 31U;
        rowWordMaps[row][wordCol] |= (1U << bit);
    }

    for (uint32_t r = 0; r < rows; ++r) {
        matrixBwcEll->k = std::max<uint32_t>(matrixBwcEll->k, (uint32_t)rowWordMaps[r].size());
        for (auto& [wordCol, word] : rowWordMaps[r]) {
            nnzWord = popcount32(word);
            minNnzWord = std::min(minNnzWord, nnzWord);
            maxNnzWord = std::max(maxNnzWord, nnzWord);
            sumNnzWord += nnzWord;
        }
    }

    matrixBwcEll->numWord = matrixBwcEll->k * rows;
    matrixBwcEll->colIdx.assign(matrixBwcEll->numWord, 0);
    matrixBwcEll->word.assign(matrixBwcEll->numWord, 0);

    // Place words in column-major: index = w*rows + r
    for (uint32_t r = 0; r < rows; ++r) {
        w = 0;
        for (auto &p : rowWordMaps[r]) {
            idx = w * rows + r;
            matrixBwcEll->colIdx[idx] = p.first + 1;
            matrixBwcEll->word[idx]   = p.second;
            ++w;
        }
    }

    if (matrixBwcEll->numWord > 0) {
        matrixBwcEll->minNnzWord = minNnzWord;
        matrixBwcEll->maxNnzWord = maxNnzWord;
        matrixBwcEll->avgNnzWord = static_cast<double>(sumNnzWord) / matrixBwcEll->numWord;
    } else {
        matrixBwcEll->minNnzWord = 0;
        matrixBwcEll->maxNnzWord = 0;
        matrixBwcEll->avgNnzWord = 0.0;
    }

    return matrixBwcEll;
}


MatrixBwcHll *convertToBwcHll(const MatrixCOO *matrixCOO) {
    MatrixBwcHll *matrixBwcHll = new MatrixBwcHll();
    uint32_t row, col, wordCol, bit, numHacks, start, end, maxWords, nnzWord, totalWords, numWord, hackStart, hackEnd, hack, rowInHack, localRows, offset, w, idx;
    uint32_t minNnzWord = std::numeric_limits<uint32_t>::max();
    uint32_t maxNnzWord = 0;
    uint64_t sumNnzWord = 0;
    uint32_t rows = matrixCOO->rows;
    uint32_t cols = matrixCOO->cols;
    uint32_t nnz  = matrixCOO->nnz;
    uint32_t hackSize = 4; 


    matrixBwcHll->rows = rows;
    matrixBwcHll->cols = cols;
    matrixBwcHll->nnz  = nnz;
    matrixBwcHll->hackSize = hackSize;

    // Step 1: Build BWC per row: wordCol → 32-bit word (bitmask)
    std::vector<std::map<uint32_t, uint32_t>> rowWords(rows);
    for (uint32_t i = 0; i < nnz; ++i) {
        row = matrixCOO->rowIdx[i] - 1;
        col = matrixCOO->colIdx[i] - 1;
        wordCol = col >> 5;
        bit     = col & 31U;
        rowWords[row][wordCol] |= (1U << bit);
    }

    // Step 2: Determine number of hacks
    numHacks = (rows + hackSize - 1) / hackSize;
    matrixBwcHll->hack.resize(numHacks + 1, 0);

    // Step 3: Compute max number of word-cols per row per hack
    std::vector<uint32_t> hackWidths(numHacks, 0);
    for (uint32_t h = 0; h < numHacks; ++h) {
        start = h * hackSize;
        end   = std::min(start + hackSize, rows);
        maxWords = 0;
        for (uint32_t r = start; r < end; ++r) {
            maxWords = std::max<uint32_t>(maxWords, rowWords[r].size());
            for (auto& [wordCol, word] : rowWords[r]) {
                nnzWord = popcount32(word);
                minNnzWord = std::min(minNnzWord, nnzWord);
                maxNnzWord = std::max(maxNnzWord, nnzWord);
                sumNnzWord += nnzWord;
            }
        }
        hackWidths[h] = maxWords;
        matrixBwcHll->hack[h + 1] = matrixBwcHll->hack[h] + maxWords * (end - start);
    }

    // Step 4: Allocate memory
    totalWords = matrixBwcHll->hack.back();
    matrixBwcHll->colIdx.assign(totalWords, 0);
    matrixBwcHll->word.assign(totalWords, 0);

    // Step 5: Set number of non zero words iterating over each hack
    numWord = 0;

    for (uint32_t h = 0; h < numHacks; ++h) {
        hackStart = matrixBwcHll->hack[h];
        hackEnd   = matrixBwcHll->hack[h + 1];

        numWord += (hackEnd - hackStart);
    }
    matrixBwcHll->numWord = numWord;

    // Step 6: Fill colIdx and word arrays in column-major per hack
    std::vector<uint32_t> rowWordPos(rows, 0);

    for (uint32_t r = 0; r < rows; ++r) {
        hack = r / hackSize;
        rowInHack = r % hackSize;
        localRows = std::min(hackSize, rows - hack * hackSize);
        offset = matrixBwcHll->hack[hack];

        w = 0;
        for (auto& entry : rowWords[r]) {
            idx = offset + w * localRows + rowInHack;
            matrixBwcHll->colIdx[idx] = entry.first + 1;  // 1-based index
            matrixBwcHll->word[idx]   = entry.second;
            ++w;
        }
    }


    if (totalWords > 0) {
        matrixBwcHll->minNnzWord = minNnzWord;
        matrixBwcHll->maxNnzWord = maxNnzWord;
        matrixBwcHll->avgNnzWord = static_cast<double>(sumNnzWord) / totalWords;
        matrixBwcHll->numWord    = totalWords;
    } else {
        matrixBwcHll->minNnzWord = 0;
        matrixBwcHll->maxNnzWord = 0;
        matrixBwcHll->avgNnzWord = 0.0;
        matrixBwcHll->numWord    = 0;
    }

    return matrixBwcHll;
}


namespace MatrixConverter {

MatrixBase* convert(MatrixCOO* matrixCOO, MatrixFormat format) {
    switch(format) {
        case MatrixFormat::COO:      return matrixCOO;
        case MatrixFormat::CSR:      return convertToCSR(matrixCOO);
        case MatrixFormat::ELL:      return convertToELL(matrixCOO);
        case MatrixFormat::HLL:      return convertToHLL(matrixCOO);
        case MatrixFormat::BWC_COO:  return convertToBwcCoo(matrixCOO);
        case MatrixFormat::BWC_CSR:  return convertToBwcCsr(matrixCOO);
        case MatrixFormat::BWC_ELL:  return convertToBwcEll(matrixCOO);
        case MatrixFormat::BWC_HLL:  return convertToBwcHll(matrixCOO);
        default:
            throw std::runtime_error("Unsupported matrix format requested.");
    }
}

} // namespace MatrixConverter