#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <iostream>

#include "NvInfer.h"

#include "cuda_runtime.h"

#include "Config.hpp"
#include "ParserConfig.hpp"
#include "ParseObjects.hpp"

class PoseModel
{
public:

  int getInputHeight(); 

  int getInputWidth();

  int getMapHeight();

  int getMapWidth();

  PoseModel(const std::string &engine_path, const Config &config); 

  ~PoseModel();

  std::vector<std::unordered_map<int, std::pair<int, int>>> execute(float *data);

private:
  int input_binding_idx, cmap_binding_idx, paf_binding_idx;
  float *cmap_buffer_h, *cmap_buffer_d, *paf_buffer_h, *paf_buffer_d;
  size_t cmap_size, paf_size;
  ParserConfig parser_config;
  nvinfer1::IRuntime *trt_runtime;
  nvinfer1::ICudaEngine *trt_engine;
  nvinfer1::IExecutionContext *trt_context;
};
