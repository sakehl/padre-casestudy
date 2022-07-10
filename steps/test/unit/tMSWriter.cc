// Copyright (C) 2021 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <boost/test/unit_test.hpp>

#include "../../MSWriter.h"

using dp3::steps::MSWriter;

BOOST_AUTO_TEST_SUITE(mswriter)

BOOST_AUTO_TEST_CASE(insert_number_in_filename) {
  BOOST_CHECK_EQUAL(MSWriter::InsertNumberInFilename("", 1982), "-1982");
  BOOST_CHECK_EQUAL(MSWriter::InsertNumberInFilename("test", 0), "test-000");
  BOOST_CHECK_EQUAL(MSWriter::InsertNumberInFilename("afile", 12), "afile-012");
  BOOST_CHECK_EQUAL(MSWriter::InsertNumberInFilename("another-file", 123),
                    "another-file-123");
  BOOST_CHECK_EQUAL(
      MSWriter::InsertNumberInFilename("file-with-extension.ms", 42),
      "file-with-extension-042.ms");
  BOOST_CHECK_EQUAL(MSWriter::InsertNumberInFilename("/and.also/path.ms", 42),
                    "/and.also/path-042.ms");
}

BOOST_AUTO_TEST_SUITE_END()
