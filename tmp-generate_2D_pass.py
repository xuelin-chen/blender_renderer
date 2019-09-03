# A simple script that uses blender to render views of a single object by rotation the camera around it.
# Also produces depth map at the same time.
#
# Example:
# /workspace/nn_project/blender-2.79-linux-glibc219-x86_64/blender --background --python generate_degraded_mesh.py -- --output_folder ./tmp /workspace/dataset/ShapeNetCore.v2/02958343/1a1de15e572e039df085b75b20c2db33/models/model_normalized.obj
# find /mnt/xuelin/dataset/ShapeNetCore.v2/02958343 -name '*.obj' -print0 | xargs -0 -n1 -P10 -I {} ../../blender-2.79b-linux-glibc219-x86_64/blender --background --python render_blender.py -- --output_folder ./all_renderings {}
#

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

from scipy import spatial
import trimesh

parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
parser.add_argument('--reso', type=int, default=640,
                    help='resolution')
parser.add_argument('--orth_scale', type=int, default=1,
                    help='view scale of orthogonal camera')
parser.add_argument('obj', type=str,
                    help='Path to the obj file to be rendered.')
parser.add_argument('--output_folder', type=str, default='./tmp',
                    help='The path the output will be dumped to.')
parser.add_argument('--normalization_mode', type=str, default='unit_sphere',
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

def parent_obj_to_camera(b_camera):
    origin = (0, 0, 0)
    b_empty = bpy.data.objects.new("Empty", None)
    b_empty.location = origin
    b_camera.parent = b_empty  # setup parenting

    scn = bpy.context.scene
    scn.objects.link(b_empty)
    scn.objects.active = b_empty
    return b_empty

# Set up rendering of depth map.
bpy.context.scene.use_nodes = True
tree = bpy.context.scene.node_tree
links = tree.links

# Add passes for additionally dumping albedo and normals.
bpy.context.scene.render.layers["RenderLayer"].use_pass_normal = True
bpy.context.scene.render.layers["RenderLayer"].use_pass_color = True
bpy.context.scene.render.image_settings.file_format = args.format
bpy.context.scene.render.image_settings.color_depth = args.color_depth

# Clear default nodes
for n in tree.nodes:
    tree.nodes.remove(n)

# Create input render layer node.
render_layers = tree.nodes.new('CompositorNodeRLayers')

# depth pass
depth_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
depth_file_output.label = 'Depth Output'
if args.format == 'OPEN_EXR':
  links.new(render_layers.outputs['Depth'], depth_file_output.inputs[0])
else:
  # Remap as other types can not represent the full range of depth.
  map = tree.nodes.new(type="CompositorNodeMapValue")
  # Size is chosen kind of arbitrarily, try out until you're satisfied with resulting depth map.
  map.offset = [-0.7]
  map.size = [args.depth_scale]
  map.use_min = True
  map.min = [0]
  links.new(render_layers.outputs['Depth'], map.inputs[0])

  links.new(map.outputs[0], depth_file_output.inputs[0])

# normal pass
if args.format == 'OPEN_EXR':
  normal_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
  normal_file_output.label = 'Normal Output'
  links.new(render_layers.outputs['Normal'], normal_file_output.inputs[0])
else:
  print('Unknow format.')

# color pass
albedo_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
albedo_file_output.label = 'Albedo Output'
links.new(render_layers.outputs['Color'], albedo_file_output.inputs[0])

# Delete default cube
bpy.data.objects['Cube'].select = True
bpy.ops.object.delete()

# object -z -> blender world y
# object y -> blender world z
# object x -> blender world x
# this axis conversion does not change the data in-place
if args.obj.endswith('.obj'):
    bpy.ops.import_scene.obj(filepath=args.obj, use_smooth_groups=False, use_split_objects=False, use_split_groups=False)
elif args.obj.endswith('.ply'):
    '''
    verts, faces_colors = util.read_ply(args.obj, return_faces=True)
    print('Read ply: ', verts.shape, faces_colors.shape)
    # transform, switch axis
    R_axis_switching = np.array([[-1, 0, 0, 0],
                                  [ 0, 0, -1, 0],
                                  [ 0, 1, 0, 0],
                                  [ 0, 0, 0, 1]])
    verts[:, :3] = util.transform_points(verts[:, :3], R_axis_switching)

    ply_mesh = bpy.data.meshes.new('ply_obj')
    ply_mesh.from_pydata(verts.tolist(), [], (faces_colors[:, :3]).tolist())
    obj = bpy.data.objects.new('ply_obj', ply_mesh)
    bpy.context.scene.objects.link(obj)
    '''
    bpy.ops.import_mesh.ply(filepath=args.obj)
    for obj in bpy.context.scene.objects:
        if obj.name == 'test_vc_bin':
            # material
            mat = bpy.data.materials.new('material_tmp')
            mat.use_object_color = True
            obj.data.materials.append(mat)
#
bpy.ops.import_scene.obj(filepath='unit_sphere.obj', use_smooth_groups=False, use_split_objects=False, use_split_groups=False)
for object in bpy.context.scene.objects:
    print(object.name)
    if object.name in ['Camera', 'Lamp']:
        continue
    bpy.context.scene.objects.active = object
    object.select = True
    
    if object.name == 'unit_sphere' or object.name == 'sphere':
        # do not touch the sphere model, which intends to give white albedo color for the background
        continue 
    else:
        if args.remove_doubles:
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.remove_doubles()
            bpy.ops.object.mode_set(mode='OBJECT')
        if args.remove_iso_verts:
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.delete_loose(use_verts=True, use_edges=True, use_faces=False)
            bpy.ops.object.mode_set(mode='OBJECT')
        if args.edge_split:
            bpy.ops.object.modifier_add(type='EDGE_SPLIT')
            bpy.context.object.modifiers["EdgeSplit"].split_angle = 1.32645
            bpy.ops.object.modifier_apply(apply_as='DATA', modifier="EdgeSplit")
        if args.normalization_mode is not None:
            # scale to be within a unit sphere (r=0.5, d=1)
            v = object.data.vertices
            verts_np = util.read_verts(object)
            trans_v, scale_f = util.pc_normalize(verts_np, norm_type=args.normalization_mode)
            # the axis conversion of importing does not change the data in-place,
            # so we do it manually
            trans_v_axis_replaced = trans_v.copy()
            trans_v_axis_replaced[1] = -trans_v[2]
            trans_v_axis_replaced[2] = trans_v[1]
            bpy.ops.transform.translate(value=(trans_v_axis_replaced[0], trans_v_axis_replaced[1], trans_v_axis_replaced[2]))
            bpy.ops.object.transform_apply(location=True)
            bpy.ops.transform.resize(value=(scale_f, scale_f, scale_f))
            bpy.ops.object.transform_apply(scale=True)
            #bpy.ops.export_scene.obj(filepath='test.obj', use_selection=True)
        
        object.select = False

# Make light just directional, disable shadows.
lamp = bpy.data.lamps['Lamp']
lamp.type = 'SUN'
lamp.shadow_method = 'NOSHADOW'
# Possibly disable specular shading:
lamp.use_specular = True

# Add another light source so stuff facing away from light is not completely dark
bpy.ops.object.lamp_add(type='SUN')
lamp2 = bpy.data.lamps['Sun']
lamp2.shadow_method = 'NOSHADOW'
lamp2.use_specular = True
lamp2.energy = 0.015
bpy.data.objects['Sun'].rotation_euler = bpy.data.objects['Lamp'].rotation_euler
bpy.data.objects['Sun'].rotation_euler[0] += 180

# disable transparency for all materials
#Iterate over all members of the material struct
for item in bpy.data.materials:
    item.use_transparency  = False

########## camera settings ##################
scene = bpy.context.scene
scene.render.resolution_x = args.reso
scene.render.resolution_y = args.reso
scene.render.resolution_percentage = 100
scene.render.alpha_mode = 'TRANSPARENT'

######### filename for output ##############
if 'ShapeNetCore' not in args.obj:
  model_identifier = args.obj.split('/')[-1].split('.')[0]
  correct_normal = False
else:
  model_identifier = args.obj.split('/')[-3]
  correct_normal = True
fp = os.path.join(args.output_folder, model_identifier)
scene.render.image_settings.file_format = 'PNG'  # set output format to .png
for output_node in [depth_file_output, normal_file_output, albedo_file_output]:
    output_node.base_path = ''

rotation_mode = 'XYZ'
all_points_normals_colors = None

for i in range(0, 1):
  cam = scene.objects['Camera']
  if i == 1: cam.location = (0, 0, 0.5)
  else: cam.location = (0, 0.5, 0)
  cam.data.type = 'ORTHO'
  cam.data.ortho_scale = args.orth_scale
  cam.data.clip_start = 0
  cam.data.clip_end = 100 # a value that is large enough
  cam_constraint = cam.constraints.new(type='TRACK_TO')
  cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
  cam_constraint.up_axis = 'UP_Y'
  b_empty = parent_obj_to_camera(cam)
  cam_constraint.target = b_empty # track to a empty object at the origin

  for rot_angle in range(0, 359, 20):

      if i == 0:
        xyz_angle = [rot_angle, 0, 0]
      elif i == 1:
        xyz_angle = [0, rot_angle, 0]
      elif i == 2:
        xyz_angle = [0, 0, rot_angle]

      scene.render.filepath = fp + 'rotx=%.2f_roty=%.2f_rotz=%.2f'%(xyz_angle[0], xyz_angle[1], xyz_angle[2])
      depth_file_output.file_slots[0].path = scene.render.filepath + "_depth"
      normal_file_output.file_slots[0].path = scene.render.filepath + "_normal"
      albedo_file_output.file_slots[0].path = scene.render.filepath + "_albedo"

      # check if it is already rendered
      if False and os.path.exists(scene.render.filepath + "_albedo.exr") and os.path.exists(scene.render.filepath + "_normal.exr") and os.path.exists(scene.render.filepath + "_depth.exr"):
          print('Skip!')
          continue

      bpy.ops.render.render(write_still=True)  # render still
      depth_arr, hard_mask_arr = util.read_depth_and_get_mask(scene.render.filepath + "_depth0001.exr")
      normal_arr = util.read_and_correct_normal(scene.render.filepath + "_normal0001.exr", correct_normal=correct_normal, mask_arr=hard_mask_arr)
      albedo_arr = util.read_exr_image(scene.render.filepath + "_albedo0001.exr")
      # and the clip value range
      depth_arr = np.clip(depth_arr, a_min=0, a_max=1)
      normal_arr = np.clip(normal_arr, a_min=-1, a_max=1)
      albedo_arr = np.clip(albedo_arr, a_min=0, a_max=1)

      # blender mask
      depth_arr = np.concatenate([np.expand_dims(depth_arr, -1), np.expand_dims(hard_mask_arr, -1)], -1)
      normal_arr = np.concatenate([normal_arr, np.expand_dims(hard_mask_arr, -1)], -1)
      albedo_arr = np.concatenate([albedo_arr, np.expand_dims(hard_mask_arr, -1)], -1)

      # write out final renderings
      util.write_exr_image(depth_arr, scene.render.filepath + "_depth.exr")
      util.write_exr_image(normal_arr, scene.render.filepath + "_normal.exr")
      util.write_exr_image(albedo_arr, scene.render.filepath + "_albedo.exr")

      # tmp remove renderings
      #os.remove(scene.render.filepath+'.png')
      os.remove(scene.render.filepath + "_normal0001.exr")
      os.remove(scene.render.filepath + "_depth0001.exr")
      os.remove(scene.render.filepath + "_albedo0001.exr")      

      b_empty.rotation_euler[0] = radians(xyz_angle[0])
      b_empty.rotation_euler[1] = radians(xyz_angle[1])
      b_empty.rotation_euler[2] = radians(xyz_angle[2])
