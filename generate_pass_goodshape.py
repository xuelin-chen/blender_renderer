# A simple script that uses blender to render views of a single object by rotation the camera around it.
# Also produces depth map at the same time.
#
# Example:
# /workspace/nn_project/blender-2.79-linux-glibc219-x86_64/blender --background --python generate_pass_shapenet.py -- --output_folder ./tmp /workspace/dataset/ShapeNetCore.v2/02958343/1a1de15e572e039df085b75b20c2db33/models/model_normalized.obj

# car
# find /workspace/dataset/ShapeNetCore.v2/02958343 -name '*.obj' -print0 | xargs -0 -n1 -P10 -I {} /workspace/nn_project/blender-2.79-linux-glibc219-x86_64/blender --background --python generate_pass_shapenet.py -- --split_file /workspace/nn_project/pytorch-CycleGAN-and-pix2pix/datasets/shapenet_car_white_list.txt --output_folder ./car_renderings {}

# chair
# find /workspace/dataset/ShapeNetCore.v2/03001627 -name '*.obj' -print0 | xargs -0 -n1 -P10 -I {} /workspace/nn_project/blender-2.79-linux-glibc219-x86_64/blender --background --python generate_pass_shapenet.py -- --split_file /workspace/nn_project/pytorch-CycleGAN-and-pix2pix/datasets/shapenet_chair_white_list.txt --output_folder ./chair_renderings {}
import bpy
import argparse, sys, os
import numpy as np

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
parser.add_argument('--reso', type=int, default=768,
                    help='resolution')
parser.add_argument('--nb_view', type=int, default=1,
                    help='number of views per model to render passes')
parser.add_argument('--orth_scale', type=int, default=1,
                    help='view scale of orthogonal camera')
parser.add_argument('obj', type=str,
                    help='Path to the scene file to be rendered.')
parser.add_argument('--output_folder', type=str, default='./tmp',
                    help='The path the output will be dumped to.')
parser.add_argument('--normalization_mode', type=str, default='diag2sphere',
                    help='if scale the mesh to be within a unit sphere.')
#parser.add_argument('--vox_resolution', type=int, default=256,
#                    help='voxelization model resolution')
parser.add_argument('--split_file', type=str, default='',
                    help='if scale the mesh to be within a unit sphere.')
parser.add_argument('--min_ele', type=float, default=15.,
                    help='minimum elevation angle of the view point.')
parser.add_argument('--max_ele', type=float, default=15.,
                    help='maximum elevation angle of the view point.')
# usually fix below args
parser.add_argument('--remove_doubles', type=bool, default=False,
                    help='Remove double vertices to improve mesh quality.')
parser.add_argument('--remove_iso_verts', type=bool, default=False,
                    help='Remove isolated vertices.')
parser.add_argument('--edge_split', type=bool, default=False,
                    help='Adds edge split filter.')
parser.add_argument('--depth_scale', type=float, default=0.5,
                    help='Scaling that is applied to depth. Depends on size of mesh. Try out various values until you get a good result. Ignored if format is OPEN_EXR.')
parser.add_argument('--color_depth', type=str, default='16',
                    help='Number of bit per channel used for output. Either 8 or 16.')
parser.add_argument('--format', type=str, default='OPEN_EXR',
                    help='Format of files generated. Either PNG or OPEN_EXR')

argv = sys.argv[sys.argv.index("--") + 1:]
args = parser.parse_args(argv)

# generate random camera rotations
rot_angles_list = []
for i in range(args.nb_view):
  rot_x_angle = random.randint(args.min_ele, args.max_ele)
  rot_y_angle = 0 # do not rot around y, no in-plane rotation
  rot_z_angle = random.randint(0, 360)
  rot_angles_list.append([rot_x_angle, rot_y_angle, rot_z_angle])

#blender_util.clear_scene_objects()
bpy.ops.wm.open_mainfile(filepath=args.obj)

win      = bpy.context.window
scr      = win.screen
areas3d  = [area for area in scr.areas if area.type == 'VIEW_3D']
region   = [region for region in areas3d[0].regions if region.type == 'WINDOW']
override = {'window':win,
            'screen':scr,
            'area'  :areas3d[0],
            'region':region,
            'scene' :bpy.context.scene,
            }


depth_file_output,normal_file_output,albedo_file_output,matidx_file_output, glossdir_file_output = blender_util.rendering_pass_setup_CYCLES(args)
# this axis conversion does not change the data in-place
#bpy.ops.import_scene.obj(filepath='unit_sphere.obj', use_smooth_groups=False, use_split_objects=False, use_split_groups=False)
#bpy.data.scenes['Scene'].render.engine = 'BLENDER_RENDER'
#bpy.ops.import_scene.obj(filepath=args.obj, use_smooth_groups=False, use_split_objects=False, use_split_groups=False)
#blender_util.convert_quad_mesh_to_triangle_mesh()
diag_length = blender_util.process_scene_objects_CYCLES(args) # including normalization

# disable transparency for all materials
for i, mat in enumerate(bpy.data.materials):
  mat.pass_index = i
  #if mat.name != 'Karbon.001': continue
  mat.use_transparency  = False
  mat.alpha = 1.

  # debug
  #print(mat.name)
  #print(mat.node_tree.nodes.keys())
  #print(mat.node_tree.nodes["Glossy BSDF.001"].inputs[1].default_value)
  #print(blender_util.get_material_roughness(i))


# setup camera resolution etc
blender_util.setup_render(args)
scene = bpy.context.scene

# render passes for shapenet shape
blender_util.render_passes_CYCLES(depth_file_output, normal_file_output, albedo_file_output, matidx_file_output, glossdir_file_output, args, rot_angles_list, diag_length=diag_length, subfolder_name='shapenet', output_format='png')
print('Shapenet shape passes done!')

bpy.ops.wm.save_as_mainfile(filepath='test.blend')