#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <iostream>
#include "Config.hpp"
#include "Object.hpp"

class IPoseModel
{
public:
  static IPoseModel *createPoseModel(const std::string &engine_path, const Config &config);
  virtual ~IPoseModel() {};
  virtual std::vector<Object> execute(float *data) = 0;
  virtual int getInputHeight() = 0; 
  virtual int getInputWidth() = 0;
  virtual int getMapHeight() = 0;
  virtual int getMapWidth() = 0;
};


