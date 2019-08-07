# RESNET18 - 256x256
python3 -m trt_pose.train \
  --model=resnet18_baseline \
  --epochs=200 \
  --batch_size=128 \
  --dataset=coco_person_256x256 \
  --output_dir=checkpoints/resnet18_baseline_256
  --save_interval=10 \
  --num_loader_workers=8

# RESNET50 - 368x368
python3 -m trt_pose.train \
  --model=resnet50_baseline \
  --epochs=200 \
  --batch_size=128 \
  --dataset=coco_person_256x256 \
  --output_dir=checkpoints/resnet50_baseline_256
  --save_interval=10 \
  --num_loader_workers=8
  
# # RESNET18 - 256x256
# python3 -m trt_pose.train \
#   --model=resnet18_baseline \
#   --epochs=200 \
#   --batch_size=64 \
#   --dataset=coco_person_368x368 \
#   --output_dir=checkpoints/resnet18_baseline_368
#   --save_interval=10 \
#   --num_loader_workers=8

# # RESNET18 - 368x368
# python3 -m trt_pose.train \
#   --model=resnet50_baseline \
#   --epochs=200 \
#   --batch_size=64 \
#   --dataset=coco_person_368x368 \
#   --output_dir=checkpoints/resnet50_baseline_368
#   --save_interval=10 \
#   --num_loader_workers=8
  