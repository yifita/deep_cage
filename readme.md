# Neural Cages for Detail-Preserving 3D Deformations
## Demo
```
git clone --recursive {GIT}
# install dependency
conda env create --name pytorch-all --file environment.yml
cd pytorch_points
python setup.py develop
# install pymesh2
# if this step fails, try to install pymesh from source as instructed here
# https://pymesh.readthedocs.io/en/latest/installation.html
# make sure that the cmake 3.15+ is used
pip install pymesh/pymesh2-0.2.1-cp37-cp37m-linux_x86_64.whl
# install other dependecies
pip install -r requirements.txt

python cage_deformer_3d.py --dataset SHAPENET --full_net --bottleneck_size 256 --n_fold 2 --ckpt log/cage_deformer_3d-chair_ablation_full/net_final.pth --target_model vanilla_data/shapenet_target/**/*.obj  --source_model vanilla_data/elaborated_chairs/throne_no_base.obj vanilla_data/elaborated_chairs/Chaise_longue_noir_House_Doctor.ply --subdir fancy_chairs --phase test --is_poly
```
