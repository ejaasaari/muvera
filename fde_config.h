#ifndef FDE_CONFIG_H_
#define FDE_CONFIG_H_

#include <cstdint>

namespace fde {

// Configuration for Fixed Dimensional Encoding generation.
struct FixedDimensionalEncodingConfig {
  // How embeddings are added to the FDE.
  enum EncodingType {
    DEFAULT_SUM = 0,  // Points are summed (used for queries)
    AVERAGE = 1       // Points are averaged (used for documents)
  };

  // How the original embeddings are projected.
  enum ProjectionType {
    DEFAULT_IDENTITY = 0,  // No projection, use original embeddings
    AMS_SKETCH = 1         // Dense AMS sketch projection
  };

  // Dimension of the input embeddings.
  int32_t dimension = 0;

  // Number of independent repetitions for FDE generation.
  int32_t num_repetitions = 1;

  // Number of SimHash projections used to partition space in each repetition.
  // Must be in range [0, 30].
  int32_t num_simhash_projections = 0;

  // Seed for the FDE generation process. Must be set to the same value for
  // query and document FDE generation to ensure consistency.
  int32_t seed = 1;

  // How embeddings are added to the FDE.
  EncodingType encoding_type = DEFAULT_SUM;

  // If a random projection encoding_type is being used, this is the
  // dimension to which points are reduced via random projections.
  int32_t projection_dimension = 0;

  // The ProjectionType sets how the original embeddings are projected.
  ProjectionType projection_type = DEFAULT_IDENTITY;

  // If true, empty partitions are filled with the nearest point.
  // Only valid for document encoding (AVERAGE).
  bool fill_empty_partitions = false;

  // If set to a positive value, the final FDE is projected down to this
  // dimension using Count-Sketch.
  int32_t final_projection_dimension = -1;

  // Helper to check if final projection is enabled.
  bool has_final_projection_dimension() const {
    return final_projection_dimension > 0;
  }
};

}  // namespace fde

#endif  // FDE_CONFIG_H_
