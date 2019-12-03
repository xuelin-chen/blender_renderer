# A simple script that uses blender to render views of a single object by rotation the camera around it.
# Also produces depth map at the same time.
#
# Example:
# /workspace/nn_project/blender-2.79-linux-glibc219-x86_64/blender --background --python generate_pass.py -- --output_folder ./tmp --split_file /workspace/nn_project/pytorch-CycleGAN-and-pix2pix/datasets/shapenet_car_all_split.txt /workspace/dataset/ShapeNetCore.v2/02958343/1a1de15e572e039df085b75b20c2db33/models/model_normalized.obj

# car
# find /workspace/dataset/ShapeNetCore.v2/02958343 -name '*.obj' -print0 | xargs -0 -n1 -P10 -I {} /workspace/nn_project/blender-2.79-linux-glibc219-x86_64/blender --background --python generate_pass.py -- --split_file /workspace/nn_project/pytorch-CycleGAN-and-pix2pix/datasets/shapenet_car_all_split.txt --vox_resolution 256 --nb_view 1 --output_folder ./tmp {}

# chair
# find /workspace/dataset/ShapeNetCore.v2/03001627 -name '*.obj' -print0 | xargs -0 -n1 -P10 -I {} /workspace/nn_project/blender-2.79-linux-glibc219-x86_64/blender --background --python generate_pass.py -- --split_file /workspace/nn_project/pytorch-CycleGAN-and-pix2pix/datasets/shapenet_chair_car_split.txt --vox_resolution 256 --output_folder ./tmp {}

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
parser.add_argument('--reso', type=int, default=400,
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
parser.add_argument('--vox_resolution', type=int, default=256,
                    help='voxelization model resolution')
parser.add_argument('--split_file', type=str, default='',
                    help='if scale the mesh to be within a unit sphere.')
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

argv = sys.argv[sys.argv.index("--") + 1:]
args = parser.parse_args(argv)

# generate random camera rotations
rot_angles_list = []
if not '_demo.txt' in args.split_file or args.split_file == '':
  for i in range(args.nb_view):
    rot_x_angle = random.randint(5, 20)
    rot_y_angle = 0 # do not rot around y, no in-plane rotation
    rot_z_angle = random.randint(0, 360)
    rot_angles_list.append([rot_x_angle, rot_y_angle, rot_z_angle])
else: # for demo data
  for x_angle in range(15, 21, 10):
    for z_angle in range(0, 360, 6):
      rot_x_angle = x_angle
      rot_y_angle = 0 # do not rot around y, no in-plane rotation
      rot_z_angle = z_angle
      rot_angles_list.append([rot_x_angle, rot_y_angle, rot_z_angle])

cls_id, modelname = util.get_shapenet_clsID_modelname_from_filename(args.obj)
if args.split_file != '':
  valid_modelname_list = util.items_in_txt_file(args.split_file)
  if modelname not in valid_modelname_list:
    print('Not in split file %s, skip!'%(args.split_file))
    bpy.ops.wm.quit_blender()

# now get the voxel mesh, get some attributes on
vox_mat_filename = 'hsp_shapenet_data/modelBlockedVoxels256/%s/%s.mat'%(cls_id, modelname)
if not os.path.exists(vox_mat_filename):
  print('Voxelization file not exist, skip!')
  bpy.ops.wm.quit_blender()
vox_mesh = util.mesh_from_voxels(vox_mat_filename, int(256/args.vox_resolution)) # already diagonal=1, center at zero
#vox_mesh.export('vox_mesh.obj')

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
blender_util.render_passes(depth_file_output, normal_file_output, albedo_file_output, args, rot_angles_list, subfolder_name='gt', output_format='png')
print('Shapenet shape passes done!')

# clear the objects imported previously
blender_util.clear_scene_objects()

# after reference queries
# switch axis before rendering
# transform to switch axis
vox_mesh.apply_transform(blender_util.R_axis_switching_StoB)
vox_bmesh = bpy.data.meshes.new('voxmesh')
vox_bmesh.from_pydata(vox_mesh.vertices.tolist(), [], vox_mesh.faces.tolist())
obj = bpy.data.objects.new('voxmesh', vox_bmesh)
bpy.context.scene.objects.link(obj)

# render passes for vox shape
blender_util.render_passes(depth_file_output, normal_file_output, albedo_file_output, args, rot_angles_list, subfolder_name='input', output_format='png')
print('Vox shape passes done!')

#bpy.ops.wm.save_as_mainfile(filepath='test.blend')