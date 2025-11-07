#include "MatrixKernel.hpp"

 /**
 * @brief This function implements the sequential operation of
 * matrix-vector multiplication, using a matrix stored in COO
 * format. (y = Ax)
 *
 * @param A THe matrix to be multiplied
 * @param x The vector to multiply the matrix with
 * @param y The output vector to populate with the result of the multiplication
 */
void spmvCOO(const MatrixCOO* A, const std::vector<uint32_t>& x, std::vector<uint32_t>& y) {
    for (uint32_t i = 0; i < A->nnz; i++) {
        uint32_t rowBitIdx = A->rowIdx[i] - 1; 
        uint32_t colBitIdx = A->colIdx[i] - 1; 

        // Extract the bit from x
        uint32_t wordX = x[colBitIdx / 32];
        uint32_t bit  = (wordX >> (colBitIdx % 32)) & 1U;

        // Apply XOR to the corresponding bit in y
        y[rowBitIdx / 32] ^= (bit << (rowBitIdx % 32));
    }
}



 /**
 * @brief This function implements the sequential operation of
 * matrix-vector multiplication, using a matrix stored in CSR
 * format. (y = Ax)
 *
 * @param A THe matrix to be multiplied
 * @param x The vector to multiply the matrix with
 * @param y The output vector to populate with the result of the multiplication
 */
void spmvCSR(const MatrixCSR* A, const std::vector<uint32_t>& x, std::vector<uint32_t>& y) {
    for (uint32_t i = 0; i < A->rows; i++) {
        uint32_t row_start = A->rowPtr[i];
        uint32_t row_end   = A->rowPtr[i + 1];

        uint32_t result_bit = 0U;

        for (uint32_t j = row_start; j < row_end; j++) {
            uint32_t colBitIdx = A->colIdx[j] - 1;
            uint32_t word = x[colBitIdx / 32];
            uint32_t bit  = (word >> (colBitIdx % 32)) & 1U;

            result_bit ^= bit;
        }

        if (result_bit) {
            y[i / 32] |= (1U << (i % 32));
        }
    }
}

/**
 * @brief Sequential SpMV for a matrix in ELL format (y = Ax).
 *        Works with bit-packed x and y (1 bit per entry).
 *
 * @param A The matrix in ELL format
 * @param x The input vector (bit-packed)
 * @param y The output vector (bit-packed, must be preallocated & zeroed)
 */
void spmvELL(const MatrixELL* A, const std::vector<uint32_t>& x, std::vector<uint32_t>& y) {
    for (uint32_t i = 0; i < A->rows; i++) {
        uint32_t result_bit = 0U;

        for (uint32_t j = 0; j < A->k; j++) {
            uint32_t col = A->colIdx[j * A->rows + i]; 
            if (col != 0) {
                col -= 1;  // ELL indices are 1-based
                uint32_t word = x[col / 32];
                uint32_t bit  = (word >> (col % 32)) & 1U;
                result_bit ^= bit;
                    
            }
        }

        if(result_bit){
            y[i / 32] ^= (1U << (i % 32));
        }

    }
}


void spmvHLL(const MatrixHLL* A, const std::vector<uint32_t>& x, std::vector<uint32_t>& y) {
    uint32_t numHacks = A->hack.size() - 1;

    for (uint32_t h = 0; h < numHacks; ++h) {
        uint32_t hackStart = A->hack[h];
        uint32_t hackEnd   = A->hack[h + 1];

        // Number of actual rows in this hack (could be < hackSize at end)
        uint32_t localRows = std::min(A->hackSize, A->rows - h * A->hackSize);
        uint32_t hackWidth = (hackEnd - hackStart) / localRows;

        for (uint32_t r = 0; r < localRows; ++r) {
            uint32_t globalRow = h * A->hackSize + r;
            uint32_t result_bit = 0U;

            for (uint32_t c = 0; c < hackWidth; ++c) {
                uint32_t idx = hackStart + c * localRows + r;
                uint32_t col = A->colIdx[idx];

                if (col != 0) {
                    col -= 1;  // HLL is 1-based
                    uint32_t word = x[col / 32];
                    uint32_t bit  = (word >> (col % 32)) & 1U;
                    result_bit ^= bit;
                }
            }

            if (result_bit) {
                y[globalRow / 32] ^= (1U << (globalRow % 32));
            }
        }
    }
}



/**
 * @brief 
 * 
 * @param A 
 * @param x 
 * @param y 
 */
void spmvBwcCoo(const MatrixBwcCoo * A, const std::vector<uint32_t>& x, std::vector<uint32_t>& y)
{
    uint32_t i, tmp, rowIdxWord, colIdxWord;

    for (i = 0; i < A->numWord; i++)
    {
        colIdxWord = A->colIdx[i] - 1;
        rowIdxWord = A->rowIdx[i] - 1;

        tmp = A->word[i] & x[colIdxWord];

        /* Check for parity */
        tmp ^= tmp >> 16;
        tmp ^= tmp >> 8;
        tmp ^= tmp >> 4;
        tmp &= 0x0F;

        tmp = (((0x6996 >> tmp) & 1) << (rowIdxWord % 32));

        /* Update the row in the word */
        y[rowIdxWord / 32] ^= tmp;
    }
}

void spmvBwcCsr(const MatrixBwcCsr * A, const std::vector<uint32_t>& x, std::vector<uint32_t>& y)
{
    uint32_t rowStart, rowEnd;
    uint32_t tmp, globalTmp;

    for (uint32_t i = 0; i < A->rows; i++)
    {
        globalTmp = 0U;
        tmp = 0U;
        rowStart = A->rowPtr[i];
        rowEnd = A->rowPtr[i + 1];

        for (uint32_t j = rowStart; j < rowEnd; j++)
        {
            tmp = A->word[j] & x[A->colIdx[j] - 1];

            /* Check for parity */
            tmp ^= tmp >> 16;
            tmp ^= tmp >> 8;
            tmp ^= tmp >> 4;
            tmp &= 0x0F;

            /* Set the right word index */
            tmp = (((0x6996 >> tmp) & 1U) << (i % 32));

            globalTmp ^= tmp;
        }

        y[i / 32] ^= globalTmp;
    }
}

void spmvBwcEll(const MatrixBwcEll * A, const std::vector<uint32_t>& x, std::vector<uint32_t>& y)
{
    uint32_t tmp, globalTmp, col;

    for (uint32_t i = 0; i < A->rows; i++)
    {
        globalTmp = 0U;
        tmp = 0U;

        for (uint32_t j = 0; j < A->k; j++)
        {
            col = A->colIdx[j * A->rows + i];

            if (col != 0)
            {
                col -= 1;
                tmp = A->word[j * A->rows + i] & x[col];

                /* Check for parity */
                tmp ^= tmp >> 16;
                tmp ^= tmp >> 8;
                tmp ^= tmp >> 4;
                tmp &= 0x0F;

                /* Set the right word index */
                tmp = (((0x6996 >> tmp) & 1) << (i % 32));

                globalTmp ^= tmp;
            }
        }

        y[i / 32] |= globalTmp; 
    }

}


void spmvBwcHll(const MatrixBwcHll* A, const std::vector<uint32_t>& x, std::vector<uint32_t>& y) {
    uint32_t numHacks = A->hack.size() - 1;

    for (uint32_t h = 0; h < numHacks; ++h) {
        uint32_t hackStart = A->hack[h];
        uint32_t hackEnd   = A->hack[h + 1];

        uint32_t localRows = std::min(A->hackSize, A->rows - h * A->hackSize);
        uint32_t hackWidth = (hackEnd - hackStart) / localRows;

        for (uint32_t r = 0; r < localRows; ++r) {
            uint32_t globalRow = h * A->hackSize + r;
            uint32_t globalTmp = 0U;

            for (uint32_t c = 0; c < hackWidth; ++c) {
                uint32_t idx = hackStart + c * localRows + r;
                uint32_t col = A->colIdx[idx];

                if (col != 0) {
                    col -= 1; // 1-based to 0-based

                    uint32_t tmp = A->word[idx] & x[col];

                    // Parity check - fold bits to get parity of tmp
                    tmp ^= tmp >> 16;
                    tmp ^= tmp >> 8;
                    tmp ^= tmp >> 4;
                    tmp &= 0x0F;

                    // Use parity to index into 0x6996 bitmask, shift by bit position
                    tmp = (((0x6996 >> tmp) & 1) << (globalRow % 32));

                    globalTmp ^= tmp;
                }
            }

            y[globalRow / 32] |= globalTmp;
        }
    }
}





 /**
 * @brief This function implements the wrapper of the sequential operation of
 * matrix-vector multiplication.
 *
 * @param A THe matrix to be multiplied
 * @param x The vector to multiply the matrix with
 * @param y The output vector to populate with the result of the multiplication
 */
void spmvCpu(const MatrixBase& matrixBase, const std::vector<uint32_t> *x, std::vector<uint32_t> *y) {
    switch (matrixBase.getFormat()) {
        case MatrixFormat::COO:
            spmvCOO(static_cast<const MatrixCOO*>(&matrixBase), *x, *y);
            break;
        case MatrixFormat::CSR:
            spmvCSR(static_cast<const MatrixCSR*>(&matrixBase), *x, *y);
            break;
        case MatrixFormat::ELL:
            spmvELL(static_cast<const MatrixELL*>(&matrixBase), *x, *y);
            break;
        case MatrixFormat::HLL:
            spmvHLL(static_cast<const MatrixHLL*>(&matrixBase), *x, *y);
            break;
        case MatrixFormat::BWC_COO:
            spmvBwcCoo(static_cast<const MatrixBwcCoo*>(&matrixBase), *x, *y);
            break;
        case MatrixFormat::BWC_CSR:
            spmvBwcCsr(static_cast<const MatrixBwcCsr*>(&matrixBase), *x, *y);
            break;
        case MatrixFormat::BWC_ELL:
            spmvBwcEll(static_cast<const MatrixBwcEll*>(&matrixBase), *x, *y);
            break;
        case MatrixFormat::BWC_HLL:
            spmvBwcHll(static_cast<const MatrixBwcHll*>(&matrixBase), *x, *y);
            break;        
        default:
            throw std::runtime_error("Unsupported matrix type in spmv");
    }
}