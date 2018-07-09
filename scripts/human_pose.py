import subprocess
import os
import re

PROTOTXT_URL = 'https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/pose/coco/pose_deploy_linevec.prototxt'
CAFFEMODEL_URL = 'http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/pose_iter_440000.caffemodel'
GIEXEC_PATH = '/usr/src/tensorrt/bin/giexec'
PROTOTXT_NAME = 'pose.prototxt'
CAFFEMODEL_NAME = 'pose.caffemodel'


def get_paf_name(num_stages):
    if num_stages == 1:
        return 'conv5_5_CPM_L1'
    return 'Mconv7_stage%d_L1' % num_stages


def get_cmap_name(num_stages):
    if num_stages == 1:
        return 'conv5_5_CPM_L2'
    return 'Mconv7_stage%d_L2' % num_stages


def build_pose_model(input_shape=(368, 368), num_stages=6, output_dir='.', precision='FP32'):
    """Downloades a pretrained pose model and builds an optimized TensorRT engine.

    :param input_shape: the input shape (height, width)
    :type input_shape: tuple of integers
    :param num_stages: the number of CMAP / PAF refinement stages
    :type num_stages: integer
    :param output_dir: the directory to output the serialized engine

    :return output_path: the path of the serialized TensorRT engine
    :type output_path: string
    """
    subprocess.call(['mkdir', '-p', output_dir])

    prototxt_path = os.path.join(output_dir, PROTOTXT_NAME)
    caffemodel_path = os.path.join(output_dir, CAFFEMODEL_NAME)

    # download prototxt
    if not os.path.exists(prototxt_path):
        subprocess.call(['wget', '--no-check-certificate',
            PROTOTXT_URL,
            '-O', prototxt_path])

    # write input dimensions
    with open(prototxt_path, 'r') as f:
        prototxt_str = f.read()
        exp = 'input_dim:.*\ninput_dim:.*\ninput_dim:.*\ninput_dim:.*\n'
        p = re.compile(exp)
        prototxt_str = p.sub('input_dim: 1\ninput_dim: 3\ninput_dim: %d\ninput_dim: %d\n' % input_shape, prototxt_str)

    with open(prototxt_path, 'w') as f:
        f.write(prototxt_str)

    # download caffemodel
    if not os.path.exists(caffemodel_path):
        subprocess.call(['wget', CAFFEMODEL_URL, '-O', caffemodel_path])
    
    engine_path = os.path.join(output_dir, os.path.basename(prototxt_path).split('.')[0] + '.plan')

    # run giexec
    giexec_args = [GIEXEC_PATH, 
        '--deploy=%s' % prototxt_path, 
        '--model=%s' % caffemodel_path,
        '--output=%s' % get_paf_name(num_stages),
        '--output=%s' % get_cmap_name(num_stages),
        '--engine=%s' % engine_path]

    if 'FP16' == precision:
        giexec_args += ['--half2']
    elif 'INT8' == precision:
        giexec_args += ['--int8']

    print(giexec_args)
    subprocess.call(giexec_args)
