import numpy as np
from scipy.io import loadmat
import skimage.measure
import trimesh
import util

#vox_mat_filename = 'hsp_shapenet_data/modelBlockedVoxels256/02958343/1a1dcd236a1e6133860800e6696b8284.mat'
vox_mat_filename = 'hsp_shapenet_data/modelBlockedVoxels256/02958343/1a1de15e572e039df085b75b20c2db33.mat'

reference_color_pointcloud = 'test_point_scan.ply'
points_normals_colors = util.read_ply(reference_color_pointcloud)

voxel_model_mat = loadmat(vox_mat_filename)
voxel_model_b = voxel_model_mat['b'][:].astype(np.int32)
voxel_model_bi = voxel_model_mat['bi'][:].astype(np.int32)-1
voxel_model_256 = np.zeros([256,256,256],np.uint8)
for i in range(16):
	for j in range(16):
		for k in range(16):
			voxel_model_256[i*16:i*16+16,j*16:j*16+16,k*16:k*16+16] = voxel_model_b[voxel_model_bi[i,j,k]]
#add flip&transpose to convert coord from shapenet_v1 to shapenet_v2
voxel_model_256 = np.flip(np.transpose(voxel_model_256, (2,1,0)), 2)

voxel_size = 1/256.
verts, faces, normals_, values = skimage.measure.marching_cubes_lewiner(
    voxel_model_256, level=0.0, spacing=[voxel_size] * 3
    )

# move to the orgine
verts = verts - [0.5,0.5,0.5]
# flip the index order for all faces
faces = faces[:, [0, 2, 1]]
mesh = trimesh.Trimesh(vertices=verts, faces=faces)
mesh.export('test_grey.ply')

mesh = util.get_color_from_reference_pointcloud(mesh, points_normals_colors)

mesh.export('test_vc_ascii.ply', encoding='ascii')
mesh.export('test_vc_bin.ply')
