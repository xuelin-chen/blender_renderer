# A simple script that uses blender to render views of a single object by rotation the camera around it.
# Also produces depth map at the same time.
#

# car train
# find /workspace/nn_project/implicit-decoder/IMGAN/car_3dsamples_train_10k_reso128/ -name '*.obj' -print0 | xargs -0 -n1 -P8 -I {} /workspace/nn_project/blender-2.79-linux-glibc219-x86_64/blender --background --python generate_pass_IMGAN.py -- --output_folder ./nips_data/im_car_renderings_10k_shape_reso128_im_reso1024_train {}
# car test
# find /workspace/nn_project/implicit-decoder/IMGAN/car_3dsamples_test_4k_reso128/ -name '*.obj' -print0 | xargs -0 -n1 -P8 -I {} /workspace/nn_project/blender-2.79-linux-glibc219-x86_64/blender --background --python generate_pass_IMGAN.py -- --nb_view 3 --output_folder ./nips_data/im_car_renderings_4k_shape_reso128_im_reso1024_test {}

# car demo
# find /workspace/nn_project/implicit-decoder/IMGAN/car_samples_4096_reso128_demo -name '*.obj' -print0 | xargs -0 -n1 -P1 -I {} /workspace/nn_project/blender-2.79-linux-glibc219-x86_64/blender --background --python generate_pass_IMGAN.py -- --demo --output_folder ./car_renderings_demo {}

# chair
# find /workspace/nn_project/implicit-decoder/IMGAN/chair_3dsamples_train_10k_reso128 -name '*.obj' -print0 | xargs -0 -n1 -P8 -I {} /workspace/nn_project/blender-2.79-linux-glibc219-x86_64/blender --background --python generate_pass_IMGAN.py -- --output_folder ./nips_data/im_chair_renderings_10k_shape_reso128_im_reso1024_train {}

import argparse, sys, os
import numpy as np

import bpy
from math import radians

import OpenEXR as exr
import Imath
import array

from PIL import Image
sys.path.append('.')
import util
import blender_camera_util
import blender_util

from scipy import spatial

from tqdm import tqdm
import trimesh
import random

parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
parser.add_argument('--reso', type=int, default=1024,
                    help='resolution')
parser.add_argument('--nb_view', type=int, default=6,
                    help='number of views per model to render passes')
parser.add_argument('--orth_scale', type=int, default=1,
                    help='view scale of orthogonal camera')
parser.add_argument('obj', type=str,
                    help='Path to the obj file to be rendered.')
parser.add_argument('--output_folder', type=str, default='./tmp',
                    help='The path the output will be dumped to.')
parser.add_argument('--normalization_mode', type=str, default='diag2sphere',
                    help='if scale the mesh to be within a unit sphere.')
#parser.add_argument('--vox_resolution', type=int, default=256,
#                    help='voxelization model resolution')
parser.add_argument('--split_file', type=str, default='',
                    help='(not used)')
parser.add_argument('--min_ele', type=float, default=0.,
                    help='minimum elevation angle of the view point.')
parser.add_argument('--max_ele', type=float, default=20.,
                    help='maximum elevation angle of the view point.')
# usually fix below args
parser.add_argument('--remove_doubles', type=bool, default=True,
                    help='Remove double vertices to improve mesh quality.')
parser.add_argument('--remove_iso_verts', type=bool, default=True,
                    help='Remove isolated vertices.')
parser.add_argument('--edge_split', type=bool, default=True,
                    help='Adds edge split filter.')
parser.add_argument('--depth_scale', type=float, default=0.5,
                    help='Scaling that is applied to depth. Depends on size of mesh. Try out various values until you get a good result. Ignored if format is OPEN_EXR.')
parser.add_argument('--color_depth', type=str, default='16',
                    help='Number of bit per channel used for output. Either 8 or 16.')
parser.add_argument('--format', type=str, default='OPEN_EXR',
                    help='Format of files generated. Either PNG or OPEN_EXR')
parser.add_argument('--demo', action='store_true', help='if this is set, camera will be put around the object densely.')

argv = sys.argv[sys.argv.index("--") + 1:]
args = parser.parse_args(argv)

# generate random camera rotations
rot_angles_list = []
if not args.demo:
  for i in range(args.nb_view):
    rot_x_angle = random.randint(args.min_ele, args.max_ele)
    rot_y_angle = 0 # do not rot around y, no in-plane rotation
    rot_z_angle = random.randint(-90, 90)
    if rot_z_angle < 0: rot_z_angle = rot_z_angle + 360.
    rot_angles_list.append([rot_x_angle, rot_y_angle, rot_z_angle])
else:
  print('Generating from dense views...')
  for x_angle in range(10, 16, 5): 
    for z_angle in range(-90, 91, 3): 
      rot_x_angle = x_angle
      rot_y_angle = 0 # do not rot around y, no in-plane rotation
      rot_z_angle = z_angle
      rot_angles_list.append([rot_x_angle, rot_y_angle, rot_z_angle])
blender_util.clear_scene_objects()
depth_file_output,normal_file_output,albedo_file_output,matidx_file_output = blender_util.rendering_pass_setup(args)

# this axis conversion does not change the data in-place
bpy.ops.import_scene.obj(filepath=args.obj, use_smooth_groups=False, use_split_objects=False, use_split_groups=False)
blender_util.process_scene_objects(args) # including normalization

# disable transparency for all materials
for i, mat in enumerate(bpy.data.materials):
  if mat.name in ['Material']: continue
  mat.use_transparency  = False

# setup camera resolution etc
blender_util.setup_render(args)
scene = bpy.context.scene

# render passes for shapenet shape
blender_util.render_passes(depth_file_output, normal_file_output, albedo_file_output, args, rot_angles_list, subfolder_name='IMGAN', output_format='png')
print('Shapenet shape passes done!')

#bpy.ops.wm.save_as_mainfile(filepath='test.blend')