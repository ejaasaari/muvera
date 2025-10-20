#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "fixed_dimensional_encoding.h"
#include "fde_config.h"

namespace py = pybind11;

PYBIND11_MODULE(fde, m) {
    m.doc() = "Fixed Dimensional Encoding (FDE) library - Python bindings";

    py::enum_<fde::FixedDimensionalEncodingConfig::EncodingType>(m, "EncodingType")
        .value("DEFAULT_SUM", fde::FixedDimensionalEncodingConfig::DEFAULT_SUM)
        .value("AVERAGE", fde::FixedDimensionalEncodingConfig::AVERAGE)
        .export_values();

    py::enum_<fde::FixedDimensionalEncodingConfig::ProjectionType>(m, "ProjectionType")
        .value("DEFAULT_IDENTITY", fde::FixedDimensionalEncodingConfig::DEFAULT_IDENTITY)
        .value("AMS_SKETCH", fde::FixedDimensionalEncodingConfig::AMS_SKETCH)
        .export_values();

    py::class_<fde::FixedDimensionalEncodingConfig>(m, "FDEConfig")
        .def(py::init<>())
        .def(py::init<const fde::FixedDimensionalEncodingConfig&>())
        .def("__copy__", [](const fde::FixedDimensionalEncodingConfig& self) {
            return fde::FixedDimensionalEncodingConfig(self);
        })
        .def("__deepcopy__", [](const fde::FixedDimensionalEncodingConfig& self, py::dict) {
            return fde::FixedDimensionalEncodingConfig(self);
        })
        .def_readwrite("dimension", &fde::FixedDimensionalEncodingConfig::dimension)
        .def_readwrite("num_repetitions", &fde::FixedDimensionalEncodingConfig::num_repetitions)
        .def_readwrite("num_simhash_projections", &fde::FixedDimensionalEncodingConfig::num_simhash_projections)
        .def_readwrite("seed", &fde::FixedDimensionalEncodingConfig::seed)
        .def_readwrite("encoding_type", &fde::FixedDimensionalEncodingConfig::encoding_type)
        .def_readwrite("projection_dimension", &fde::FixedDimensionalEncodingConfig::projection_dimension)
        .def_readwrite("projection_type", &fde::FixedDimensionalEncodingConfig::projection_type)
        .def_readwrite("fill_empty_partitions", &fde::FixedDimensionalEncodingConfig::fill_empty_partitions)
        .def_readwrite("final_projection_dimension", &fde::FixedDimensionalEncodingConfig::final_projection_dimension)
        .def("has_final_projection_dimension", &fde::FixedDimensionalEncodingConfig::has_final_projection_dimension);

    m.def("generate_query_fixed_dimensional_encoding",
        [](py::array_t<float> point_cloud, const fde::FixedDimensionalEncodingConfig& config) {
            py::buffer_info buf = point_cloud.request();
            std::vector<float> vec(static_cast<float*>(buf.ptr),
                                  static_cast<float*>(buf.ptr) + buf.size);

            std::vector<float> result = fde::GenerateQueryFixedDimensionalEncoding(vec, config);
            return py::array_t<float>(result.size(), result.data());
        },
        py::arg("point_cloud"),
        py::arg("config"),
        R"pbdoc(
            Generate query-side Fixed Dimensional Encoding (uses sum aggregation).

            Args:
                point_cloud: Flattened numpy array of shape (num_points * dimension,)
                            containing the point cloud vectors concatenated.
                config: FDEConfig object with configuration parameters.

            Returns:
                numpy array containing the fixed dimensional encoding.

            Raises:
                ValueError: If configuration is invalid.
        )pbdoc");

    m.def("generate_document_fixed_dimensional_encoding",
        [](py::array_t<float> point_cloud, const fde::FixedDimensionalEncodingConfig& config) {
            py::buffer_info buf = point_cloud.request();
            std::vector<float> vec(static_cast<float*>(buf.ptr),
                                  static_cast<float*>(buf.ptr) + buf.size);

            std::vector<float> result = fde::GenerateDocumentFixedDimensionalEncoding(vec, config);
            return py::array_t<float>(result.size(), result.data());
        },
        py::arg("point_cloud"),
        py::arg("config"),
        R"pbdoc(
            Generate document-side Fixed Dimensional Encoding (uses average aggregation).

            Args:
                point_cloud: Flattened numpy array of shape (num_points * dimension,)
                            containing the point cloud vectors concatenated.
                config: FDEConfig object with configuration parameters.

            Returns:
                numpy array containing the fixed dimensional encoding.

            Raises:
                ValueError: If configuration is invalid.
        )pbdoc");
}
