// pydppp.cc: python bindings to create python step
// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include "PyDPStep.h"
#include "PyDPStepImpl.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <casacore/measures/Measures/MDirection.h>

namespace py = pybind11;

namespace DP3 {
namespace DPPP {

template <typename T>
void register_cube(py::module &m, const char *name) {
  py::class_<casacore::Cube<T>>(m, name, py::buffer_protocol())
      .def_buffer([](casacore::Cube<T> &cube) -> py::buffer_info {
        return py::buffer_info(
            cube.data(),                        /* Pointer to buffer */
            sizeof(T),                          /* Size of one scalar */
            py::format_descriptor<T>::format(), /* Python struct-style format
                                                   descriptor */
            3,                                  /* Number of dimensions */
            {cube.shape()[2], cube.shape()[1],
             cube.shape()[0]}, /* Buffer dimensions */
            {sizeof(T) * cube.shape()[1] *
                 cube.shape()[0], /* Strides (in bytes) for each index */
             sizeof(T) * cube.shape()[0], sizeof(T)});
      });
}

template <typename T>
void register_matrix(py::module &m, const char *name) {
  py::class_<casacore::Matrix<T>>(m, name, py::buffer_protocol())
      .def_buffer([](casacore::Matrix<T> &matrix) -> py::buffer_info {
        return py::buffer_info(
            matrix.data(),                      /* Pointer to buffer */
            sizeof(T),                          /* Size of one scalar */
            py::format_descriptor<T>::format(), /* Python struct-style format
                                                   descriptor */
            2,                                  /* Number of dimensions */
            {matrix.shape()[1], matrix.shape()[0]}, /* Buffer dimensions */
            {sizeof(T) *
                 matrix.shape()[0], /* Strides (in bytes) for each index */
             sizeof(T)});
      });
}

template <typename T>
void register_vector(py::module &m, const char *name) {
  py::class_<casacore::Vector<T>>(m, name, py::buffer_protocol())
      .def_buffer([](casacore::Vector<T> &vector) -> py::buffer_info {
        return py::buffer_info(
            vector.data(),                      /* Pointer to buffer */
            sizeof(T),                          /* Size of one scalar */
            py::format_descriptor<T>::format(), /* Python struct-style format
                                                   descriptor */
            1,                                  /* Number of dimensions */
            {vector.shape()[0]},                /* Buffer dimensions */
            {sizeof(T)} /* Strides (in bytes) for each index */
        );
      });
}

PYBIND11_MODULE(pydppp, m) {
  m.doc() = "pybind11 example plugin";  // optional module docstring

  register_cube<float>(m, "Cube_float");
  register_cube<bool>(m, "Cube_bool");
  register_cube<casacore::Complex>(m, "Cube_complex_float");
  register_matrix<double>(m, "Matrix_double");
  register_vector<double>(m, "Vector_double");
  register_vector<int>(m, "Vector_int");

  py::class_<ostream>(m, "ostream").def("write", &ostream::write);

  py::class_<DPStepWrapper,
             std::shared_ptr<DPStepWrapper>,  // holder type
             PyDPStepImpl                     /* <--- trampoline*/
             >(m, "DPStep")
      .def(py::init<>())
      .def("show", &DPStepWrapper::show)
      .def("update_info", &DPStepWrapper::updateInfo)
      .def("info", &DPStepWrapper::info, py::return_value_policy::reference)
      .def("get_next_step", &DPStepWrapper::get_next_step,
           py::return_value_policy::copy)
      .def("process_next_step", &DPStepWrapper::process_next_step)
      .def("get_count", &DPStepWrapper::get_count)
      .def_readwrite("fetch_uvw", &DPStepWrapper::m_fetch_uvw)
      .def_readwrite("fetch_weights", &DPStepWrapper::m_fetch_weights);

  py::class_<DPBuffer, std::shared_ptr<DPBuffer>>(m, "DPBuffer")
      .def(py::init<>())
      .def("get_data", (casacore::Cube<casacore::Complex> &
                        (DPBuffer::*)())(&DPBuffer::getData))
      .def("get_weights",
           (casacore::Cube<float> & (DPBuffer::*)())(&DPBuffer::getWeights))
      .def("get_flags",
           (casacore::Cube<bool> & (DPBuffer::*)())(&DPBuffer::getFlags))
      .def("get_uvw",
           (casacore::Matrix<double> & (DPBuffer::*)())(&DPBuffer::getUVW));

  py::class_<DPInfo>(m, "DPInfo")
      .def("antenna_names",
           [](DPInfo &self) -> py::array {
             // Convert casa vector of casa strings to std::vector of strings
             casacore::Vector<casacore::String> names_casa =
                 self.antennaNames();
             std::vector<std::string> names;
             for (size_t i = 0; i < names_casa.size(); ++i) {
               names.push_back(names_casa[i]);
             }
             py::array ret = py::cast(names);
             return ret;
           })
      .def("antenna_positions",
           [](DPInfo &self) -> py::array {
             // Convert vector of casa MPositions to std::vector of positions
             std::vector<casacore::MPosition> positions_casa =
                 self.antennaPos();
             std::vector<std::array<double, 3>> positions;
             for (size_t i = 0; i < positions_casa.size(); ++i) {
               casacore::MVPosition position_mv = positions_casa[i].getValue();
               std::array<double, 3> position_array = {
                   position_mv(0), position_mv(1), position_mv(2)};
               positions.push_back(position_array);
             }
             py::array ret = py::cast(positions);
             return ret;
           })
      .def("set_need_vis_data", &DPInfo::setNeedVisData)
      .def("get_channel_frequencies", &DPInfo::chanFreqs,
           py::arg("baseline") = 0)
      .def("get_antenna1", &DPInfo::getAnt1)
      .def("get_antenna2", &DPInfo::getAnt2)
      .def("nantenna", &DPInfo::nantenna)
      .def("nchan", &DPInfo::nchan)
      .def("start_time", &DPInfo::startTime)
      .def("time_interval", &DPInfo::timeInterval)
      .def("ntime", &DPInfo::ntime);
}

void test() {}

}  // namespace DPPP
}  // namespace DP3