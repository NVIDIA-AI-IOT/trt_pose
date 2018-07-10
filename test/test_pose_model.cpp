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
  Config config = DEFAULT_HUMAN_POSE_CONFIG();
  std::unique_ptr<IPoseModel> model;
  model.reset(IPoseModel::createPoseModel(test_engine_path, config));
  
  float *data_h, *data_d;
  size_t size = createInput(&data_h, 
      TEST_IMAGE_PATH, model->getInputHeight(), model->getInputWidth());
  cudaMalloc(&data_d, size);
  cudaMemcpy(data_d, data_h, size, cudaMemcpyHostToDevice);

  auto objects = model->execute(data_d);
  ASSERT_EQ(2, objects.size());

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
