'''
The MIT License (MIT)
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

"""
usage: export_for_isaac.py [-h] --input_checkpoint INPUT_CHECKPOINT
                           [--input_model INPUT_MODEL]
                           [--input_topology INPUT_TOPOLOGY]
                           [--input_width INPUT_WIDTH]
                           [--input_height INPUT_HEIGHT]
                           [--output_model OUTPUT_MODEL]

example: ./export_for_isaac.py --input_checkpoint resnet18_baseline_att_224x224_A_epoch_249.pth

NVIDIA AI IOT 'TensorRT Pose Estimation' to Isaac 'OpenPose Inference' model conversion tool.
Converts the NVIDIA-AI-IOT/trt_pose neural network from the PyTorch into ONNX format compatible
with the Isaac OpenPose Inference codelet.

Please refer to Isaac Documentation and https://github.com/NVIDIA-AI-IOT/trt_pose documentation.

Required arguments:
  --input_checkpoint INPUT_CHECKPOINT
                        Input model weights (.pth)

Optional arguments:
  -h, --help            show this help message and exit
  --input_model INPUT_MODEL
                        Input model (trt_pose.models function)
  --input_topology INPUT_TOPOLOGY
                        Input topology (.json)
  --input_width INPUT_WIDTH
                        Input image width
  --input_height INPUT_HEIGHT
                        Input image width
  --output_model OUTPUT_MODEL
                        Output ONNX model (.onnx)
"""

import argparse, json, os, re
import torch                    # Please refer to: https://pytorch.org/get-started/locally/
import trt_pose.models


class InputReNormalization(torch.nn.Module):
    """
        This defines "(input - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]" custom operation
        to conform to "Unit" normalized input RGB data.
    """
    def __init__(self):
        super(InputReNormalization, self).__init__()
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).reshape((1,3,1,1)).cuda()
        self.std = torch.Tensor([0.229, 0.224, 0.225]).reshape((1,3,1,1)).cuda()

    def forward(self, x):
        return (x - self.mean) / self.std


class HeatmapMaxpoolAndPermute(torch.nn.Module):
    """
        This defines MaxPool2d(kernel_size = 3, stride = 1) and permute([0,2,3,1]) custom operation
        to conform to [part_affinity_fields, heatmap, maxpool_heatmap] output format.
    """
    def __init__(self):
        super(HeatmapMaxpoolAndPermute, self).__init__()
        self.maxpool = torch.nn.MaxPool2d(3, stride=1, padding=1)

    def forward(self, x):
        heatmap, part_affinity_fields = x
        maxpool_heatmap = self.maxpool(heatmap)

        part_affinity_fields = part_affinity_fields.permute([0,2,3,1])
        heatmap = heatmap.permute([0,2,3,1])
        maxpool_heatmap = maxpool_heatmap.permute([0,2,3,1])
        return [part_affinity_fields, heatmap, maxpool_heatmap]

def main(args):
    """
    Loads PyTorch model from args.input_checkpoint, converts and saves it into args.output_model path

    Arguments:
    args: the parsed command line arguments
    """


    # Load model topology and define the model
    if not args.input_topology.endswith(".json") or not os.path.exists(args.input_topology):
        raise SystemExit("Input topology %s is not a valid (.json) file." % args.input_topology)

    with open(args.input_topology, 'r') as f:
        topology = json.load(f)

    num_parts, num_links = len(topology['keypoints']), len(topology['skeleton'])
    model = trt_pose.models.MODELS[args.input_model](num_parts, num_links * 2).cuda().eval()

    # Load model weights
    if not args.input_checkpoint.endswith(".pth"):
        raise SystemExit("Unsupported input model %s. Only (.pth) model weights are supported." %
                        args.input_checkpoint)

    if not os.path.exists(args.input_checkpoint):
        raise SystemExit("Input model file %s doesn't exists." % args.input_checkpoint)

    model.load_state_dict(torch.load(args.input_checkpoint))

    # Add InputReNormalization pre-processing and HeatmapMaxpoolAndPermute post-processing operations
    converted_model = torch.nn.Sequential(InputReNormalization(), model, HeatmapMaxpoolAndPermute())

    # Define input and output names for ONNX exported model.
    input_names = ["input"]
    output_names = ["part_affinity_fields", "heatmap", "maxpool_heatmap"]

    # Export the model to ONNX.
    dummy_input = torch.zeros((1, 3, args.input_height, args.input_width)).cuda()
    torch.onnx.export(converted_model, dummy_input, args.output_model,
                      input_names=input_names, output_names=output_names)
    print("Successfully completed convertion of %s to %s." % (args.input_checkpoint, args.output_model))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "NVIDIA AI IOT 'TensorRT Pose Estimation' to "
        "Isaac 'OpenPose Inference' model conversion tool. Converts the NVIDIA-AI-IOT/trt_pose "
        "neural network from the PyTorch into ONNX format compatible with the Isaac OpenPose "
        "Inference codelet. Please refer to Isaac Documentation and "
        "https://github.com/NVIDIA-AI-IOT/trt_pose documentation.", epilog = "example:"
        "./export_for_isaac.py --input_checkpoint resnet18_baseline_att_224x224_A_epoch_249.pth")

    parser.add_argument("--input_checkpoint", help="Input checkpoint file (.pth)", required = True)
    parser.add_argument("--input_model", help="Input model (trt_pose.models function)", required = False)
    parser.add_argument("--input_topology", help="Input topology (.json)", default = "human_pose.json")
    parser.add_argument("--input_width", help="Input image width", type = int, required = False)
    parser.add_argument("--input_height", help="Input image width", type = int, required = False)
    parser.add_argument("--output_model", help="Output ONNX model (.onnx)", required = False)
    args = parser.parse_args()

    if args.input_topology is None:
        args.input_topology = "human_pose.json"
        print("Input topology is not specified, using %s as a default." % args.input_topology)

    if args.input_model is None:
        match = re.match("^(\w+)_\d+x\d+\w+.pth", args.input_checkpoint)
        if match:
            args.input_model, = match.groups()
            print("Input model is not specified, using %s as a default." % args.input_model)
        else:
            raise SystemExit("Input model is not specified and can not be inferenced from the "
                             "name of the checkpoint %s. Please specify the model name "
                             "(trt_pose.models function name). " % args.input_checkpoint)

    if args.input_width is None or args.input_height is None:
        match = re.match("^\w+_(\d+)x(\d+)_\w+.pth", args.input_checkpoint)
        if match:
            args.input_height,args.input_width = map(int, match.groups())
            print("Input width/height are not specified, using %dx%d as a default." %
                  (args.input_height, args.input_width))
        else:
            raise SystemExit("Input model height or width is not specified and can not be "
                            "inferenced from the name of the checkpoint %s. Please specify the "
                            "model height and width (for example 480x640)" % args.input_checkpoint)



    if args.output_model is None:
        args.output_model = os.path.splitext(args.input_checkpoint)[0] + ".onnx"
        print("Output path is not specified, using %s as a default." % args.output_model)

    main(args)
