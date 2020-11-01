# A simple script that uses blender to render views of a single object by rotation the camera around it.
# Also produces depth map at the same time.
#
# Example:
# /home/kpl/xuelin/project/blender-2.79-linux-glibc219-x86_64/blender --background --python render_spinning_obj.py -- --scene ./data/Barrel.blend --output_folder ./output/barrel



import argparse, sys, os
import numpy as np
import json

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

parser = argparse.ArgumentParser(description='Renders a spinning scene.')

parser.add_argument('--reso', type=int, default=256,
                    help='resolution')
parser.add_argument('--nb_views', type=int, default=60,
                    help='number of views per model to render passes')
parser.add_argument('--scene', type=str,
                    help='Path to the scene file to be rendered.')
parser.add_argument('--output_folder', type=str, default='./tmp',
                    help='The path the output will be dumped to.')
parser.add_argument('--normalize_obj', action='store_true', 
                    help='if to normalize the object in the scene.')

#camera
parser.add_argument('--cam_dist', type=float, default=2.0,
                    help='camera distance')
parser.add_argument('--focal_lens', type=float, default=50.,
                    help='in mm.')
parser.add_argument('--cam_sensor_width', type=int, default=25.00,
                    help='camera sensor width')

# usually fix below args
parser.add_argument('--remove_doubles', type=bool, default=True,
                    help='Remove double vertices to improve mesh quality.')
parser.add_argument('--remove_iso_verts', type=bool, default=True,
                    help='Remove isolated vertices.')
parser.add_argument('--edge_split', type=bool, default=True,
                    help='Adds edge split filter.')
parser.add_argument('--color_depth', type=str, default='16',
                    help='Number of bit per channel used for output. Either 8 or 16.')
parser.add_argument('--format', type=str, default='OPEN_EXR',
                    help='Format of files generated. Either PNG or OPEN_EXR')

argv = sys.argv[sys.argv.index("--") + 1:]
args = parser.parse_args(argv)

def render_spinning_obj(args, camera_location=[0, 2, 0], data_split_name='train'):
    ######### filename for output ##############
    fp = args.output_folder

    # Get the filename only from the initial file path.
    filename = os.path.basename(args.scene)
    # Use splitext() to get filename and extension separately.
    (target_obj_name, ext) = os.path.splitext(filename)
    
    # set up renderer
    scene = bpy.context.scene
    scene.render.resolution_x = args.reso
    scene.render.resolution_y = args.reso
    scene.render.resolution_percentage = 100
    scene.render.alpha_mode = 'TRANSPARENT'
    scene.render.image_settings.file_format = 'PNG'  # set output format to .png

    # setup camera
    cam = blender_util.get_default_camera()
    cam.matrix_world = mathutils.Matrix.Translation(camera_location)
    cam.data.sensor_width = args.cam_sensor_width
    cam.data.sensor_height = args.cam_sensor_width
    cam.data.lens = args.focal_lens
    cam.data.lens_unit = 'MILLIMETERS'
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
    cam_constraint.target = b_empty # track to a empty object at the origin    

    # lamp
    #bpy.context.scene.objects['Lamp'].matrix_world = cam.matrix_world

    euler_list = []
    view_angle_step = 360. / args.nb_views * 2.
    for i in range(int(args.nb_views/2)):
        euler_list.append([0, 0, view_angle_step])
    for i in range(int(args.nb_views/2)):
        euler_list.append([view_angle_step, 0, 0])   

    frames = []
    for aidx, euler_angle in enumerate(euler_list):
        for object in bpy.context.scene.objects:
            if object.name != target_obj_name: continue
          
            bpy.context.scene.objects.active = object
            #object.select = True
            blender_util.rotate_obj(object, euler_angle )
            #object.select = False

        scene.render.filepath = fp + '/images/{:s}/{:06d}'.format(data_split_name, aidx)

        # render and write out
        bpy.ops.render.render(write_still=True)  # render still

        frame_here = {}
        frame_here['file_path'] = 'images/{:s}/{:06d}'.format(data_split_name, aidx) + '.png'
        frame_here['camera_pose_matrix'] = np.array(cam.matrix_world).tolist() # get camera pose: cam2world matrix
        frame_here['object_transform_matrix_relative'] = np.array(mathutils.Matrix.Rotation(radians(view_angle_step), 4, 'Z')).tolist()
        frame_here['object_transform_matrix_absolute'] = np.array(mathutils.Matrix.Rotation(radians(view_angle_step * aidx), 4, 'Z')).tolist()
        
        frames.append(frame_here)

    # intrinsic
    K = blender_camera_util.get_calibration_matrix_K_from_blender(cam.data)
    f = K[0][0]
    W = args.reso
    H = args.reso

    out_dict = {}
    out_dict['image_height'] = H
    out_dict['image_width'] = W
    out_dict['camera_focal_lens'] = f
    out_dict['frames'] = frames

    with open(os.path.join(args.output_folder, data_split_name+'.json'), 'w') as outfile:  
        json.dump(out_dict, outfile, indent=4) 


if __name__=='__main__':
    # open a scene
    bpy.ops.wm.open_mainfile(filepath=args.scene)

    # process the object in the scene
    if args.normalize_obj:
        # Get the filename only from the initial file path.
        filename = os.path.basename(args.scene)
        # Use splitext() to get filename and extension separately.
        (target_obj_name, ext) = os.path.splitext(filename)
        for object in bpy.context.scene.objects:
            if object.name != target_obj_name: continue

            bpy.context.scene.objects.active = object
            object.select = True

            verts_np = blender_util.get_obj_verts(object, read_global=True)
            trans_v, scale_f = util.pc_normalize(verts_np, norm_type='diag2sphere')
            trans_v_axis_replaced = trans_v

            bpy.ops.transform.translate(value=(trans_v_axis_replaced[0], trans_v_axis_replaced[1], trans_v_axis_replaced[2]))
            bpy.ops.object.transform_apply(location=True)
            bpy.ops.transform.resize(value=(scale_f, scale_f, scale_f))
            bpy.ops.object.transform_apply(scale=True)
            #bpy.ops.export_scene.obj(filepath='./test.obj', use_selection=True)
            object.select = False

    bpy.data.scenes["Scene"].render.engine = 'CYCLES'
    depth_file_output,normal_file_output,albedo_file_output,matidx_file_output = blender_util.rendering_pass_setup(args)

    # render passes for shapenet shape
    render_spinning_obj(args, camera_location=(0, args.cam_dist, 0), data_split_name='train')
    render_spinning_obj(args, camera_location=(0, args.cam_dist, 0), data_split_name='val')
    render_spinning_obj(args, camera_location=(0,-args.cam_dist, 0), data_split_name='test')
    print('Done!')

    #bpy.ops.wm.save_as_mainfile(filepath='test_test.blend')