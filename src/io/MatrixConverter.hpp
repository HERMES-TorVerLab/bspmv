/**
 * @file MatrixConverter.hpp
 * @author Stack1
 * @brief 
 * @version 1.0
 * @date 19-09-2025
 * 
 * 
 */
#pragma once
#include "../format/Matrix.hpp"

class MatrixBase;

namespace MatrixConverter {
    MatrixBase* convert(MatrixCOO* matrixcOO, MatrixFormat format);
}

inline uint32_t popcount32(uint32_t x) {
#if defined(__GNUG__) || defined(__clang__)
    return __builtin_popcount(x);   // fast builtin on GCC/Clang
#elif defined(_MSC_VER)
    return __popcnt(x);             // MSVC intrinsic
#else
    // fallback
    uint32_t count = 0;
    while (x) {
        x &= (x - 1);
        count++;
    }
    return count;
#endif
}