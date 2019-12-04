# A simple script that uses blender to render views of a single object by rotation the camera around it.
# Also produces depth map at the same time.
#
# Example:
# /workspace/nn_project/blender-2.79-linux-glibc219-x86_64/blender --background --python generate_point_cloud.py -- --output_folder ./tmp /workspace/dataset/ShapeNetCore.v2/02958343/1a1de15e572e039df085b75b20c2db33/models/model_normalized.obj

# car
# find /workspace/dataset/ShapeNetCore.v2/02958343 -name '*.obj' -print0 | xargs -0 -n1 -P10 -I {} /workspace/nn_project/blender-2.79-linux-glibc219-x86_64/blender --background --python generate_point_cloud.py -- --split_file /workspace/nn_project/pytorch-CycleGAN-and-pix2pix/datasets/shapenet_car_train_split_demo.txt --vox_resolution 256 --output_folder ./car_rendering_passes_train_fordemo {}

# chair
# find /workspace/dataset/ShapeNetCore.v2/03001627 -name '*.obj' -print0 | xargs -0 -n1 -P10 -I {} /workspace/nn_project/blender-2.79-linux-glibc219-x86_64/blender --background --python generate_point_cloud.py -- --split_file /workspace/nn_project/pytorch-CycleGAN-and-pix2pix/datasets/shapenet_chair_train_split.txt --vox_resolution 128 --output_folder ./chair_renderings_128_train {}

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
parser.add_argument('--reso', type=int, default=640,
                    help='resolution')
parser.add_argument('--nb_view', type=int, default=18,
                    help='number of views per model to render passes')
parser.add_argument('--orth_scale', type=int, default=1,
                    help='view scale of orthogonal camera')
parser.add_argument('obj', type=str,
                    help='Path to the obj file to be rendered.')
parser.add_argument('--output_folder', type=str, default='./tmp',
                    help='The path the output will be dumped to.')
parser.add_argument('--normalization_mode', type=str, default='diag2sphere',
                    help='if scale the mesh to be within a unit sphere.')
parser.add_argument('--vox_resolution', type=int, default=128,
                    help='voxelization model resolution')
parser.add_argument('--split_file', type=str, default='/workspace/nn_project/pytorch-CycleGAN-and-pix2pix/datasets/shapenet_car_test_split_demo.txt',
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
if not '_demo.txt' in args.split_file:
  for i in range(args.nb_view):
    rot_x_angle = random.randint(0, 30)
    rot_y_angle = 0 # do not rot around y, no in-plane rotation
    rot_z_angle = random.randint(0, 360)
    rot_angles_list.append([rot_x_angle, rot_y_angle, rot_z_angle])
else: # for demo data
  for x_angle in range(15, 21, 10):
    for z_angle in range(0, 360, 3):
      rot_x_angle = x_angle
      rot_y_angle = 0 # do not rot around y, no in-plane rotation
      rot_z_angle = z_angle
      rot_angles_list.append([rot_x_angle, rot_y_angle, rot_z_angle])

cls_id, modelname = util.get_shapenet_clsID_modelname_from_filename(args.obj)
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

# assign each material a unique id
# disable transparency for all materials
for i, mat in enumerate(bpy.data.materials):
  if mat.name in ['Material']: continue
  mat.pass_index = i
  mat.use_transparency  = False

# setup camera resolution etc
blender_util.setup_render(args)
scene = bpy.context.scene

# render shapenet shape to get color point cloud
all_points_normals_colors_mindices = blender_util.scan_point_cloud(depth_file_output, normal_file_output, albedo_file_output, matidx_file_output, args)
all_points_normals_colors_mindices = util.sample_from_point_cloud(all_points_normals_colors_mindices, int(all_points_normals_colors_mindices.shape[0]/10))
#util.write_ply(all_points_normals_colors_mindices[:, :3], 'point_cloud.ply', colors=all_points_normals_colors_mindices[:, 6:9], normals=all_points_normals_colors_mindices[:, 3:6])
print('Shapenet point cloud scanning done!')


# render passes for shapenet shape
blender_util.render_passes(depth_file_output, normal_file_output, albedo_file_output, args, rot_angles_list, subfolder_name='gt')
print('Shapenet shape passes done!')

# clear the objects imported previously
blender_util.clear_scene_objects()

print('Obtaining reference point indices...')
ref_pts_indices = util.get_ref_point_idx_from_point_cloud(vox_mesh, all_points_normals_colors_mindices)
print('Reference point indices obtained.')
vox_mesh_face_matidx_list = all_points_normals_colors_mindices[ref_pts_indices, -1]

# after reference queries
# switch axis before rendering
# transform to switch axis
vox_mesh.apply_transform(blender_util.R_axis_switching_StoB)

# separate faces into groups based on matidx
print('Separating submeshes from material pass indices...')
submesh_face_list = dict()
for i in range(len(vox_mesh.faces)):
  matidx_cur = vox_mesh_face_matidx_list[i]
  if matidx_cur not in submesh_face_list.keys():
    submesh_face_list[matidx_cur] = [i]
  else:
    submesh_face_list[matidx_cur].append(i)
print('Submeshes done.')

print('Creating blender objects from submeshes...')
for matidx, submesh_face_list in submesh_face_list.items():
  sub_trimesh = vox_mesh.submesh([submesh_face_list], append=True)

  sub_bmesh = bpy.data.meshes.new('submesh_%d'%(matidx))
  sub_bmesh.from_pydata(sub_trimesh.vertices.tolist(), [], sub_trimesh.faces.tolist())
  obj = bpy.data.objects.new('submesh_%d'%(matidx), sub_bmesh)
  bpy.context.scene.objects.link(obj)

  mat = blender_util.get_material_from_passIdx(matidx)
  if mat is None: print('Warning: material from pass index is None.')
  obj.data.materials.append(mat)
print('Blender objects created.')

# render passes for vox shape
blender_util.render_passes(depth_file_output, normal_file_output, albedo_file_output, args, rot_angles_list, subfolder_name='input')
print('Vox shape passes done!')

#bpy.ops.wm.save_as_mainfile(filepath='test.blend')