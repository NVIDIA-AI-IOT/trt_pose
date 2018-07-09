#include "gtest/gtest.h"

#include <memory>
#include <opencv2/opencv.hpp>
#include "../src/tensor.h"
#include "cuda_runtime.h"
#include "../src/PoseModel.hpp"
#include "../src/Config.hpp"

#define TEST_IMAGE_PATH "data/test.jpg"
#define TEST_ENGINE_PATH "data/pose.plan"
#define TEST_ENGINE_HEIGHT 368
#define TEST_ENGINE_WIDTH 368
#define TEST_ENGINE_INPUT_NAME "image"
#define TEST_ENGINE_CMAP_NAME "Mconv7_stage6_L2"
#define TEST_ENGINE_PAF_NAME "Mconv7_stage6_L1"

size_t createInput(float **data, std::string path, int height, int width)
{
  cv::Mat image = cv::imread(path.c_str()); 
  cv::resize(image, image, { width, height });
  cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
  size_t size = sizeof(float) * image.channels() * image.rows * image.cols;
  *data = (float*) malloc(size);
  for (int c = 0; c < image.channels(); c++) 
  {
    for (int i = 0; i < image.rows; i++)
    {
      for (int j = 0; j < image.cols; j++)
      {
        // convert HWC to CHW and scale and offset
        (*data)[IDX_3D(c, i, j, image.rows, image.cols)] = ((float) image.data[IDX_3D(i, j, c, image.cols, image.channels())]) / 255.0f - 0.5;
      }
    }
  }
  return size;
}

TEST(pose_model, sample_image)
{
  std::string test_engine_path = TEST_ENGINE_PATH;
  const int height = TEST_ENGINE_HEIGHT;
  const int width = TEST_ENGINE_WIDTH;
  float *data_h, *data_d;
  size_t size = createInput(&data_h, TEST_IMAGE_PATH, height, width);
  cudaMalloc(&data_d, size);
  cudaMemcpy(data_d, data_h, size, cudaMemcpyHostToDevice);

  Config config;
  config.trt_cmap_name = TEST_ENGINE_CMAP_NAME;
  config.trt_paf_name = TEST_ENGINE_PAF_NAME;
  config.trt_input_name = TEST_ENGINE_INPUT_NAME;
  config.peak_threshold = 0.3;
  config.paf_cost_num_samples = 10;
  config.part_names = {
    "NOSE",
    "NECK",
    "RSHOULDER",
    "RELBOW",
    "RWRIST",
    "LSHOULDER",
    "LELBOW",
    "LWRIST",
    "RHIP",
    "RKNEE",
    "RANKLE",
    "LHIP",
    "LKNEE",
    "LANKLE",
    "REYE",
    "LEYE",
    "REAR",
    "LEAR",
  };
  config.topology = {
    { config.partIndex("NECK"), config.partIndex("RHIP") },
    { config.partIndex("RHIP"), config.partIndex("RKNEE") },
    { config.partIndex("RKNEE"), config.partIndex("RANKLE") },
    { config.partIndex("NECK"), config.partIndex("LHIP") },
    { config.partIndex("LHIP"), config.partIndex("LKNEE") },
    { config.partIndex("LKNEE"), config.partIndex("LANKLE") },
    { config.partIndex("NECK"), config.partIndex("RSHOULDER") },
    { config.partIndex("RSHOULDER"), config.partIndex("RELBOW") },
    { config.partIndex("RELBOW"), config.partIndex("RWRIST") },
    { config.partIndex("RSHOULDER"), config.partIndex("REAR") },
    { config.partIndex("NECK"), config.partIndex("LSHOULDER") },
    { config.partIndex("LSHOULDER"), config.partIndex("LELBOW") },
    { config.partIndex("LELBOW"), config.partIndex("LWRIST") },
    { config.partIndex("LSHOULDER"), config.partIndex("LEAR") },
    { config.partIndex("NECK"), config.partIndex("NOSE") },
    { config.partIndex("NOSE"), config.partIndex("REYE") },
    { config.partIndex("NOSE"), config.partIndex("LEYE") },
    { config.partIndex("REYE"), config.partIndex("REAR") },
    { config.partIndex("LEYE"), config.partIndex("LEAR") },
  };

  PoseModel model(test_engine_path, config);

  auto objects = model.execute(data_d);

  free(data_h);
  cudaFree(data_d);
}

#ifndef EXCLUDE_MAIN
int main(int argc, char *argv[])
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
#endif
