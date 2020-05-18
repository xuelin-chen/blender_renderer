# A simple script that uses blender to render views of a single object by rotation the camera around it.
# Also produces depth map at the same time.
#
# Example:
# /workspace/nn_project/blender-2.79-linux-glibc219-x86_64/blender --background --python render_shapenet_v2.py -- --output_folder ./tmp /workspace/dataset/ShapeNetCore.v2/02958343/1a1de15e572e039df085b75b20c2db33/models/model_normalized.obj

# car
# find /workspace/dataset/ShapeNetCore.v2/02958343 -name '*.obj' -print0 | xargs -0 -n1 -P10 -I {} /workspace/nn_project/blender-2.79-linux-glibc219-x86_64/blender --background --python render_shapenet_v2.py -- --split_file /workspace/nn_project/pytorch-CycleGAN-and-pix2pix/datasets/shapenet_car_white_list.txt --output_folder ./car_renderings {}
# find /workspace/dataset/ShapeNetCore.v2/02958343 -name '*.obj' -print0 | xargs -0 -n1 -P10 -I {} /workspace/nn_project/blender-2.79-linux-glibc219-x86_64/blender --background --python render_shapenet_v2.py -- --demo --nb_view 6 --cam_dist 1.3 --focal_len 525 --reso 512 --output_folder ./srns_car_test_data/ {}

# chair
# find /workspace/dataset/ShapeNetCore.v2/03001627 -name '*.obj' -print0 | xargs -0 -n1 -P10 -I {} /workspace/nn_project/blender-2.79-linux-glibc219-x86_64/blender --background --python render_shapenet_v2.py -- --split_file /workspace/nn_project/pytorch-CycleGAN-and-pix2pix/datasets/shapenet_chair_white_list.txt --output_folder ./chair_renderings {}
# find /workspace/dataset/ShapeNetCore.v2/03001627 -name '*.obj' -print0 | xargs -0 -n1 -P10 -I {} /workspace/nn_project/blender-2.79-linux-glibc219-x86_64/blender --background --python render_shapenet_v2.py -- --demo --nb_view 6 --cam_dist 2.0 --focal_len 131.25 --reso 128 --output_folder ./srns_chair_test_data/ {}

import argparse, sys, os
import numpy as np

import bpy
from math import radians
import mathutils
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
from transforms3d.euler import euler2mat

parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
parser.add_argument('--cam_dist', type=float, default=2.0,
                    help='camera distance')
parser.add_argument('--focal_len', type=float, default=131.25,
                    help='camera distance')           
parser.add_argument('--reso', type=int, default=256,
                    help='resolution')
parser.add_argument('--nb_view', type=int, default=250,
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
                    help='if scale the mesh to be within a unit sphere.')
parser.add_argument('--min_ele', type=float, default=5.,
                    help='minimum elevation angle of the view point.')
parser.add_argument('--max_ele', type=float, default=20.,
                    help='maximum elevation angle of the view point.')
parser.add_argument('--demo', action='store_true', help='if this is set, camera will be put around the object densely.')
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

def render(depth_file_output,normal_file_output,albedo_file_output,args, rot_angles_list, subfolder_name='gt', output_format='exr'):
    scene = bpy.context.scene
    scene.render.resolution_x = args.reso
    scene.render.resolution_y = args.reso
    scene.render.resolution_percentage = 100
    scene.render.alpha_mode = 'TRANSPARENT'
    scene.render.image_settings.file_format = 'PNG'  # set output format to .png
    ######### filename for output ##############
    if 'ShapeNetCore' not in args.obj:
        model_identifier = args.obj.split('/')[-1].split('.')[0]
    else:
        model_identifier = args.obj.split('/')[-3]
    fp = os.path.join(args.output_folder, subfolder_name, model_identifier)
    
    # setup camera and render
    #cam_init_location = (0., 0., -1.3)
    cam = blender_util.get_default_camera()
    #cam.location = (0,0,0)
    cam.matrix_world = mathutils.Matrix.Identity(4)
    cam.data.sensor_height = args.reso
    cam.data.sensor_width = args.reso
    cam.data.lens = args.focal_len
    cam_constraint = cam.constraints.new(type='TRACK_TO')
    cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    cam_constraint.up_axis = 'UP_Y'
    cam_constraint.use_target_z = True
    b_empty = blender_util.get_lookat_target(cam)
    for object in bpy.context.scene.objects:
        if object.name == 'Empty':
            object.select = True
        else:
            object.select = False
    bpy.ops.transform.rotate(value=-np.pi/2.0, axis=(1,0,0))
    cam_constraint.target = b_empty # track to a empty object at the origin

    K = blender_camera_util.get_calibration_matrix_K_from_blender(cam.data)
    
    for aidx, xyz_angle in enumerate(rot_angles_list):
        mat_loc = mathutils.Matrix.Translation((0.0, 0.0, args.cam_dist))
        mat_rot_x = mathutils.Matrix.Rotation(radians(xyz_angle[0]), 4, 'X')
        mat_rot_y = mathutils.Matrix.Rotation(radians(xyz_angle[1]), 4, 'Y')
        mat_rot_z = mathutils.Matrix.Rotation(radians(xyz_angle[2]), 4, 'Z')
        mat_comb = mat_rot_z * mat_rot_y * mat_rot_x * mat_loc
        cam.matrix_world = mat_comb

        # the sun lamp follows
        #sun_lamp.location = new_cam_location
        scene.render.filepath = fp + '/_rgb_tmp/{:06d}'.format(aidx)
        #scene.render.filepath = fp + '-rotx=%.2f_roty=%.2f_rotz=%.2f'%(xyz_angle[0], xyz_angle[1], xyz_angle[2])
        depth_file_output.file_slots[0].path = scene.render.filepath + "_depth"
        normal_file_output.file_slots[0].path = scene.render.filepath + "_normal"
        albedo_file_output.file_slots[0].path = scene.render.filepath + "_albedo"

        # render and write out
        bpy.ops.render.render(write_still=True)  # render still

        depth_arr, hard_mask_arr = util.read_depth_and_get_mask(scene.render.filepath + "_depth0001.exr", far_thre=args.cam_dist+0.6)
        normal_arr = util.read_and_correct_normal(scene.render.filepath + "_normal0001.exr", correct_normal=True, mask_arr=hard_mask_arr)
        albedo_arr = util.read_exr_image(scene.render.filepath + "_albedo0001.exr")
        # and the clip value range
        depth_arr = np.clip(depth_arr, a_min=0, a_max=1)
        normal_arr = np.clip(normal_arr, a_min=-1, a_max=1)
        albedo_arr = np.clip(albedo_arr, a_min=0, a_max=1)

        if output_format == 'png':
            depth_arr = np.array(depth_arr*255, dtype=np.uint8)
            depth_pil = Image.fromarray(depth_arr)
            if not os.path.exists(os.path.join(fp, '_depth')):
              os.makedirs(os.path.join(fp, '_depth'))
            depth_pil.save(os.path.join(fp, '_depth', '{:06d}.png'.format(aidx)))

            normal_arr = np.array((normal_arr+1)/2.*255, dtype=np.uint8)
            normal_pil = Image.fromarray(normal_arr)
            if not os.path.exists(os.path.join(fp, '_normal')):
              os.makedirs(os.path.join(fp, '_normal'))
            normal_pil.save(os.path.join(fp, '_normal', '{:06d}.png'.format(aidx)))
        
            albedo_arr = np.array(albedo_arr*255, dtype=np.uint8)
            albedo_pil = Image.fromarray(albedo_arr)
            if not os.path.exists(os.path.join(fp, '_albedo')):
              os.makedirs(os.path.join(fp, '_albedo'))
            albedo_pil.save(os.path.join(fp, '_albedo', '{:06d}.png'.format(aidx)))

            hard_mask_arr = np.array(hard_mask_arr*255, dtype=np.uint8)
            mask_pil = Image.fromarray(hard_mask_arr)
            if not os.path.exists(os.path.join(fp, '_mask')):
              os.makedirs(os.path.join(fp, '_mask'))
            mask_pil.save(os.path.join(fp, '_mask', '{:06d}.png'.format(aidx)))

        # remove renderings
        #os.remove(scene.render.filepath+'.png')
        os.remove(scene.render.filepath + "_normal0001.exr")
        os.remove(scene.render.filepath + "_depth0001.exr")
        os.remove(scene.render.filepath + "_albedo0001.exr")
        #os.remove('Image0001.exr')

        # intrinsic output
        if not os.path.exists(os.path.join(fp, 'intrinsics')):
            os.makedirs(os.path.join(fp, 'intrinsics'))
        intrinsic_params = np.array(K).flatten()
        np.savetxt(os.path.join(fp, 'intrinsics', '{:06d}.txt'.format(aidx)), intrinsic_params[None], fmt='%1.1f')

        # get camera pose: cam2world matrix
        RT = blender_camera_util.get_3x4_RT_matrix_from_blender(cam)
        RT4 = np.zeros((4,4))
        RT4[:3,:4] = RT
        RT4[3,3] = 1
        cam2world_mat = np.linalg.inv(RT4)
        #print(cam2world_mat)
        if not os.path.exists(os.path.join(fp, 'pose')):
            os.makedirs(os.path.join(fp, 'pose'))
        pose_params = np.array(cam2world_mat).flatten()
        np.savetxt(os.path.join(fp, 'pose', '{:06d}.txt'.format(aidx)), pose_params[None], fmt='%1.6f')
    
    # intrinsic output
    #print(K)
    f_out = K[0][0]
    cx = K[0][2]
    cy = K[1][2]
    im_width = args.reso
    im_height = args.reso
    with open(os.path.join(fp, 'intrinsics.txt'), 'w') as f:
      f.write("%1.6f %1.6f %1.6f %1.2f\n" % (float(f_out), float(cx), float(cy), 0.))
      f.write("0. 0. 0.\n")
      f.write("1.\n")
      f.write("%d %d\n" % (im_width, im_height))
    #np.savetxt(os.path.join(fp, 'intrinsics.txt'), intrinsic_params[None], fmt='%1.1f')
        

# generate random camera rotations in blender coordinate system
rot_angles_list = []
if not args.demo:
  for i in range(args.nb_view):
    rot_x_angle = random.randint(0, 360)
    rot_y_angle = random.randint(0, 360)
    rot_z_angle = random.randint(0, 360)
    rot_angles_list.append([rot_x_angle, rot_y_angle, rot_z_angle])
else:
  print('Generating from dense views...')
  for i in range(args.nb_view):
    rot_x_angle = random.randint(-20, 0)
    rot_y_angle = random.randint(90, 270)
    rot_z_angle = 0
    rot_angles_list.append([rot_x_angle, rot_y_angle, rot_z_angle])

cls_id, modelname = util.get_shapenet_clsID_modelname_from_filename(args.obj)
if args.split_file != '':
  valid_modelname_list = util.items_in_txt_file(args.split_file)
  if modelname not in valid_modelname_list:
    print('Not in split file %s, skip!'%(args.split_file))
    bpy.ops.wm.quit_blender()

blender_util.clear_scene_objects()
depth_file_output,normal_file_output,albedo_file_output,matidx_file_output = blender_util.rendering_pass_setup(args)

# shapenet v2 coordinate system: Y - up, -Z - face
# after imported to blender, the up of the object will be the Z axis, the face will be Y in blender world...
#bpy.ops.import_scene.obj(filepath=args.obj, use_smooth_groups=False, use_split_objects=False, use_split_groups=False)
# import to stay in shapenet v2 system
bpy.ops.import_scene.obj(filepath=args.obj, axis_forward='Y', axis_up='Z', use_smooth_groups=False, use_split_objects=False, use_split_groups=False)
#blender_util.process_scene_objects(args) # including normalization

# disable transparency for all materials
for i, mat in enumerate(bpy.data.materials):
  #if mat.name in ['Material']: continue
  mat.use_transparency  = False
  mat.use_mist = False
  mat.use_raytrace = False
  mat.diffuse_shader = 'LAMBERT'
  mat.diffuse_intensity = 0.6
  mat.specular_shader = 'BLINN'
  mat.specular_intensity = 0.4
  # debug
  #print(mat.roughness)
# setup camera resolution etc

# render passes for shapenet shape
render(depth_file_output,normal_file_output,albedo_file_output,args, rot_angles_list, subfolder_name='shapenet', output_format='png')
print('Shapenet shape passes done!')

#bpy.ops.wm.save_as_mainfile(filepath='test_test.blend')