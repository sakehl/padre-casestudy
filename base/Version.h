// Copyright (C) 2021 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef DP3_VERSION_H
#define DP3_VERSION_H

#define DP3_VERSION          "6.0.0"
#define DP3_VERSION_MAJOR    6
#define DP3_VERSION_MINOR    0
#define DP3_VERSION_PATCH    0

#define DP3_INSTALL_PATH     "/usr/local"

#include <string>

class DP3Version {
 public:
  static std::string AsString() { return "DP3 " + std::string(DP3_VERSION); }
};

#endif
