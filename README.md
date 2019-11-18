# blender_renderer
Scripts for rendering shapenet data in blender, the scripts are tested using blender 2.79.

# for generating ShapeNet point cloud, use generate_shapenet_point_cloud.py script:
/workspace/nn_project/blender-2.79-linux-glibc219-x86_64/blender --background --python generate_shapenet_point_cloud.py -- --output_folder ./tmp /workspace/dataset/ShapeNetCore.v2/02958343/1a1de15e572e039df085b75b20c2db33/models/model_normalized.obj

Ack.:
This repo uses some code from https://github.com/panmari/stanford-shapenet-renderer
