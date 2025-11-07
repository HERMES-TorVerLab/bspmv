#include "Matrix.hpp"
#include "../io/MatrixReader.hpp" 


// ===== Methods implementation ===== 

Matrix::Matrix(const std::string& filename, MatrixFormat format){
    matrixImplementation = MatrixReader::readMatrixFromFile(filename, format);
    this->format = format;
}

void Matrix::print() const {
    if (matrixImplementation)
        matrixImplementation->print();
    else
        throw std::runtime_error("No actual implementation of matrix given when calling print.");
}

MatrixFormat Matrix::getFormat() const {
    if (matrixImplementation)
        return matrixImplementation->getFormat();
    else
        throw std::runtime_error("No actual implementation of matrix given when calling getFormat.");
}

uint32_t Matrix::getRows() const{
    if (matrixImplementation)
        return matrixImplementation->getRows();
    else
        throw std::runtime_error("No actual implementation of matrix given when calling getRows."); 
}

uint32_t Matrix::getCols() const{
    if (matrixImplementation)
        return matrixImplementation->getCols();
    else
        throw std::runtime_error("No actual implementation of matrix given when calling getCols."); 
}

uint32_t Matrix::getNonZeros() const{
    if (matrixImplementation)
        return matrixImplementation->getNonZeros();
    else
        throw std::runtime_error("No actual implementation of matrix given when calling getNonZeros."); 
}