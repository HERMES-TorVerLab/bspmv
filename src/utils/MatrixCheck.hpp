#ifndef MATRIX_CHECK_HPP
#define MATRIX_CHECK_HPP

#include "bspmvUtils.hpp"

class MatrixCheck {
public:
    // Constructor
    MatrixCheck();

    bool check(const std::string &baslineFileName, MatrixFormat baslineFormat, const std::string &compareFileName, MatrixFormat compareFormat);
};

#endif // MATRIX_CHECK_HPP
