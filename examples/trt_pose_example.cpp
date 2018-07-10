#include <memory>
#include <iostream>

#include <opencv2/opencv.hpp>

#include "argus_camera/ArgusCamera.hpp"

#include "trt_pose/PoseModel.hpp"
#include "trt_pose/Config.hpp"
#include "trt_pose/tensor.h"

#include "cuda_runtime.h"

using namespace std;

void formatInput(uint8_t *src, float *dst, int height, int width)
{
  for (int c = 0; c < 3; c++)
  {
    for (int i = 0; i < height; i++)
    {
      for (int j = 0; j < width; j++)
      {
        dst[IDX_3D(c, i, j, height, width)] = src[IDX_3D(i, j, c, width, 3)] / 255.0 - 0.5;
      }
    }
  }
}

int main()
{
  // create pose model
  Config pose_config = DEFAULT_COCO_HUMAN_POSE_CONFIG();
  pose_config.trt_cmap_name = "Mconv7_stage2_L2";
  pose_config.trt_paf_name = "Mconv7_stage2_L1";
  pose_config.peak_threshold = 0.4;
  std::unique_ptr<IPoseModel> model(IPoseModel::createPoseModel("data/pose_256_2.plan", pose_config));
   
  unsigned int image_width = model->getInputWidth();
  unsigned int image_height = model->getInputHeight();

  // create argus camera
  //ArgusCameraConfig camera_config = ArgusCameraConfig::createDefaultDevkitConfig();
  //camera_config.setStreamResolution({ image_height, image_width });
  //camera_config.setVideoConverterResolution({ image_height, image_width });
  //std::unique_ptr<IArgusCamera> camera(IArgusCamera::createArgusCamera(camera_config));

  cv::Mat raw;//(image_height, image_width, CV_8UC4); // to hold image
  //uint8_t *data = (uint8_t *) malloc(sizeof(uint8_t) * 4 * image_height * image_width);
  size_t image_size = sizeof(float) * 3 * image_height * image_width;
  float *image_h = (float*) malloc(image_size);
  float *image_d;
  cudaMalloc(&image_d, image_size);
  auto cap = cv::VideoCapture(1);
  float wscale = (float) image_width / model->getMapWidth();
  float hscale = (float) image_height / model->getMapHeight();
  cv::namedWindow("image", cv::WINDOW_NORMAL);
  while (cv::waitKey(1) < 0)
  {
    cap.read(raw);
    cv::resize(raw, raw, {image_width, image_height});
    cv::cvtColor(raw, raw, cv::COLOR_BGR2RGB);
    //camera->read(data);

    formatInput(raw.data, image_h, image_height, image_width);
    cudaMemcpy(image_d, image_h, image_size, cudaMemcpyHostToDevice);

    auto objects = model->execute(image_d);
    for (int j = 0; j < objects.size(); j++) 
    {
      cv::Scalar color = { 0, 255, 0 };
      if (j == 0)
        color = { 0, 255, 0 };
      else
        color = { 255, 0 , 0};

      auto object = objects[j];
      for (int i = 0; i < pose_config.part_names.size(); i++)
      {
        if (object.count(i) > 0) 
        {
                    cout << pose_config.part_names[i] << " ";
          cv::circle(raw, { (object[i].second + 0.5f) * wscale, (object[i].first + 0.5) * hscale}, 5, color,5);
        }
      }
      cout << endl;
    }

    cv::imshow("image", raw);
  }

  //free(raw.);
  free(image_h);
  cudaFree(image_d);

  return 0;
}
