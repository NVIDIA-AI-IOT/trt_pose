# 256x256
python3 -m trt_pose.data.create_dataset \
  coco_person_256x256 \
  train2017 \
  annotations/person_keypoints_train2017_modified.json \
  person \
  256 \
  256 \
  64 \
  64 \
  1.4

# 368x368
# python3 -m trt_pose.data.create_dataset \
#   coco_person_368x368 \
#   train2017 \
#   annotations/person_keypoints_train2017_modified.json \
#   person \
#   368 \
#   368 \
#   92 \
#   92 \
#   2.0