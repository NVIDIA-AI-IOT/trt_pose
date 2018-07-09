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

class PoseModelLogger : public nvinfer1::ILogger
{
  void log (Severity severity, const char *msg) override
  {
    if (severity != Severity::kINFO)
    {
      std::cout << msg << std::endl;
    }
  }
} poseModelLogger;

size_t dimSize(nvinfer1::Dims dims)
{
  size_t size = 1;
  for (int i = 0; i < dims.nbDims; i++)
  {
    size *= dims.d[i];
  }
  return size;
}

class PoseModel
{
public:

  int getInputHeight() {
    return trt_engine->getBindingDimensions(input_binding_idx).d[1];
  }

  int getInputWidth() {
    return trt_engine->getBindingDimensions(input_binding_idx).d[2];
  }

  int getMapHeight() {
    return trt_engine->getBindingDimensions(cmap_binding_idx).d[1];
  }

  int getMapWidth() {
    return trt_engine->getBindingDimensions(cmap_binding_idx).d[2];
  }

  PoseModel(const std::string &engine_path, const Config &config) 
  {
    std::ifstream engine_file;
    engine_file.open(engine_path);
    std::stringstream engine_stream;
    engine_stream << engine_file.rdbuf();
    std::string engine_string = engine_stream.str();
    
    trt_runtime = nvinfer1::createInferRuntime(poseModelLogger);
    trt_engine = trt_runtime->deserializeCudaEngine((void*)engine_string.data(), engine_string.size(), nullptr);
    trt_context = trt_engine->createExecutionContext();

    // allocate output buffers
    input_binding_idx = trt_engine->getBindingIndex(config.trt_input_name.c_str());
    cmap_binding_idx = trt_engine->getBindingIndex(config.trt_cmap_name.c_str());
    paf_binding_idx = trt_engine->getBindingIndex(config.trt_paf_name.c_str());
    auto paf_dims = trt_engine->getBindingDimensions(paf_binding_idx);
    auto cmap_dims = trt_engine->getBindingDimensions(cmap_binding_idx);

    paf_size = dimSize(paf_dims);//todo: handle multiple batch size
    cmap_size = dimSize(cmap_dims);

    // override config dimensions
    parser_config.map_height = ((nvinfer1::DimsCHW*)&cmap_dims)->h();
    parser_config.map_width = ((nvinfer1::DimsCHW*)&cmap_dims)->w();
    parser_config.num_parts = config.part_names.size();
    parser_config.topology = config.topology;
    parser_config.paf_cost_num_samples = config.paf_cost_num_samples;
    parser_config.peak_threshold = config.peak_threshold;

    cmap_buffer_h = (float*) malloc(sizeof(float) * cmap_size);
    cudaMalloc(&cmap_buffer_d, sizeof(float) * cmap_size);
    paf_buffer_h = (float*) malloc(sizeof(float) * paf_size);
    cudaMalloc(&paf_buffer_d, sizeof(float) * paf_size);
  }

  ~PoseModel()
  {
    trt_runtime->destroy();
    trt_engine->destroy();
    trt_context->destroy();
    free(cmap_buffer_h);
    cudaFree(cmap_buffer_d);
    free(paf_buffer_h);
    cudaFree(paf_buffer_d);
  }

  std::vector<std::unordered_map<int, std::pair<int, int>>> execute(float *data)
  {
    void *bindings[3];
    bindings[input_binding_idx] = data;
    bindings[cmap_binding_idx] = cmap_buffer_d;
    bindings[paf_binding_idx] = paf_buffer_d;

    // execute engine
    // TODO: multi batch size
    trt_context->execute(1, bindings);

    // copy output to host
    cudaMemcpy(cmap_buffer_h, cmap_buffer_d, sizeof(float) * cmap_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(paf_buffer_h, paf_buffer_d, sizeof(float) * paf_size, cudaMemcpyDeviceToHost);

    // parse results
    return parseObjects(cmap_buffer_h, paf_buffer_h, parser_config);
  }

private:
  int input_binding_idx, cmap_binding_idx, paf_binding_idx;
  float *cmap_buffer_h, *cmap_buffer_d, *paf_buffer_h, *paf_buffer_d;
  size_t cmap_size, paf_size;
  ParserConfig parser_config;
  nvinfer1::IRuntime *trt_runtime;
  nvinfer1::ICudaEngine *trt_engine;
  nvinfer1::IExecutionContext *trt_context;
};
