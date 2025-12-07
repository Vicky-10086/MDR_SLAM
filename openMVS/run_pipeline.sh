#!/bin/bash

DATA_DIR="/data/dense"

echo "Step 1: Generate scene.mvs"
cd $DATA_DIR
/usr/local/bin/OpenMVS/InterfaceCOLMAP \
    -i . \
    -o scene.mvs \
    --image-folder $DATA_DIR/images

echo "Step 2: Dense Reconstruction (Densify)"
/usr/local/bin/OpenMVS/DensifyPointCloud scene.mvs

echo "Step 3: Mesh Reconstruction (Mesh)"
/usr/local/bin/OpenMVS/ReconstructMesh scene_dense.mvs

echo "Step 4: Texture Mapping (Texture)"
/usr/local/bin/OpenMVS/TextureMesh scene_dense.mvs --mesh-file scene_dense_mesh.ply

echo "Pipeline finished! Results saved in $DATA_DIR"
