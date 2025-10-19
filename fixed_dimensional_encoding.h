#ifndef FIXED_DIMENSIONAL_ENCODING_H_
#define FIXED_DIMENSIONAL_ENCODING_H_

#include <cstdint>
#include <vector>

#include "Eigen/Core"
#include "fde_config.h"

namespace fde {

// These utility functions implement the process of generating a randomized
// "Fixed Dimensional Encoding" (FDE) from a variable sized set of vectors
// (called a point cloud). Specifically, the functions take as input a
// `point_cloud`, which is a concatenated list of vectors of the same dimension
// config.dimension(). The functions output a single vector (the FDE), such that
// the dot product between a query FDE and a document FDE approximates the
// Chamfer Similarity between the original query point cloud and document point
// cloud. See https://arxiv.org/pdf/2405.19504v1 for further details.

// This is a wrapper method that routes to either
// `GenerateQueryFixedDimensionalEncoding` or
// `GenerateDocumentFixedDimensionalEncoding` based on config.encoding_type.
// Throws std::invalid_argument on invalid configuration.
std::vector<float> GenerateFixedDimensionalEncoding(
    const std::vector<float>& point_cloud,
    const FixedDimensionalEncodingConfig& config);

// Generates query-side Fixed Dimensional Encoding (uses sum aggregation).
// Throws std::invalid_argument on invalid configuration.
std::vector<float> GenerateQueryFixedDimensionalEncoding(
    const std::vector<float>& point_cloud,
    const FixedDimensionalEncodingConfig& config);

// Generates document-side Fixed Dimensional Encoding (uses average aggregation).
// Throws std::invalid_argument on invalid configuration.
std::vector<float> GenerateDocumentFixedDimensionalEncoding(
    const std::vector<float>& point_cloud,
    const FixedDimensionalEncodingConfig& config);

namespace internal {  // For testing only

// Returns the partition index of the given vector. This is computed by
// thresholding `input_vector` by mapping positive entries to 1 and negative
// entries to 0, and then interpreting the result as a binary vector and
// converting it to an int using the Gray code conversion from binary vectors to
// ints.
uint32_t SimHashPartitionIndex(const Eigen::VectorXf& input_vector);

uint32_t DistanceToSimHashPartition(const Eigen::VectorXf& input_vector,
                                    uint32_t index);

// Applies a random projection to a vector using Count-Sketch matrix, which is a
// sparse random matrix. Specifically, each each entry from the input is added
// to a single random entry in the output vector with a random sign.
std::vector<float> ApplyCountSketchToVector(
    const std::vector<float>& input_vector, uint32_t final_dimension,
    uint32_t seed);
}  // namespace internal

}  // namespace fde

#endif  // FIXED_DIMENSIONAL_ENCODING_H_
