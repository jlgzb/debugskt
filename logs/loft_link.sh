#!/bin/bash
#rm /lustre/chaixiujuan/gzb/mmlab/mmclassification/checkpoints
#rm /lustre/chaixiujuan/gzb/mmlab/mmclassification/mmgzb
#rm /lustre/chaixiujuan/gzb/mmlab/mmclassification/data

#rm /lustre/chaixiujuan/gzb/mmlab/mmclassification/configs/configs_orange

#rm /lustre/chaixiujuan/gzb/mmlab/mmclassification/configs/_base_/datasets/orange21_bs64.py
#rm /lustre/chaixiujuan/gzb/mmlab/mmclassification/configs/_base_/datasets/orange_diameter_bs64.py
#rm /lustre/chaixiujuan/gzb/mmlab/mmclassification/configs/_base_/datasets/orange_grade_bs64.py

#rm /lustre/chaixiujuan/gzb/mmlab/mmclassification/tools/train.py
#rm /lustre/chaixiujuan/gzb/mmlab/mmclassification/mmcls/datasets/__init__.py
#rm /lustre/chaixiujuan/gzb/mmlab/mmclassification/mmcls/datasets/dataset_orange.py
#rm /lustre/chaixiujuan/gzb/mmlab/mmclassification/mmcls/datasets/pipelines/__init__.py
#rm /lustre/chaixiujuan/gzb/mmlab/mmclassification/mmcls/datasets/pipelines/transforms.py

#rm /lustre/chaixiujuan/gzb/mmlab/mmclassification/mmcls/models/necks/__init__.py
#rm /lustre/chaixiujuan/gzb/mmlab/mmclassification/mmcls/models/necks/gmp.py

# mmcls/models/necks
#ln -s /lustre/chaixiujuan/gzb/ownEnv/mmgzb/mmcls_setting/mmcls_models/necks/__init__.py /lustre/chaixiujuan/gzb/mmlab/mmclassification/mmcls/models/necks/__init__.py
#ln -s /lustre/chaixiujuan/gzb/ownEnv/mmgzb/mmcls_setting/mmcls_models/necks/gmp.py /lustre/chaixiujuan/gzb/mmlab/mmclassification/mmcls/models/necks/gmp.py
#ln -s /lustre/chaixiujuan/gzb/ownEnv/mmgzb/mmcls_setting/mmcls_models/necks/gcn.py /lustre/chaixiujuan/gzb/mmlab/mmclassification/mmcls/models/necks/gcn.py

# checkpoints, data
#ln -s /lustre/chaixiujuan/gzb/checkpoints_all /lustre/chaixiujuan/gzb/mmlab/mmclassification/checkpoints
#ln -s /lustre/chaixiujuan/gzb/ownEnv/mmgzb /lustre/chaixiujuan/gzb/mmlab/mmclassification/mmgzb
#ln -s /lustre/chaixiujuan/gzb/data /lustre/chaixiujuan/gzb/mmlab/mmclassification/data

# configs
#ln -s /lustre/chaixiujuan/gzb/ownEnv/mmgzb/mmcls_setting/configs_orange /lustre/chaixiujuan/gzb/mmlab/mmclassification/configs/configs_orange
#ln -s /lustre/chaixiujuan/gzb/ownEnv/mmgzb/mmcls_setting/configs_base_datasets/orange21_bs64.py /lustre/chaixiujuan/gzb/mmlab/mmclassification/configs/_base_/datasets/orange21_bs64.py
#ln -s /lustre/chaixiujuan/gzb/ownEnv/mmgzb/mmcls_setting/configs_base_datasets/orange_diameter_bs64.py /lustre/chaixiujuan/gzb/mmlab/mmclassification/configs/_base_/datasets/orange_diameter_bs64.py
#ln -s /lustre/chaixiujuan/gzb/ownEnv/mmgzb/mmcls_setting/configs_base_datasets/orange_grade_bs64.py /lustre/chaixiujuan/gzb/mmlab/mmclassification/configs/_base_/datasets/orange_grade_bs64.py

# pipeline: dataset, compose
#ln -s /lustre/chaixiujuan/gzb/ownEnv/mmgzb/mmcls_setting/mmcls_datasets/__init__.py /lustre/chaixiujuan/gzb/mmlab/mmclassification/mmcls/datasets/__init__.py
#ln -s /lustre/chaixiujuan/gzb/ownEnv/mmgzb/mmcls_setting/mmcls_datasets/dataset_orange.py /lustre/chaixiujuan/gzb/mmlab/mmclassification/mmcls/datasets/dataset_orange.py

#ln -s /lustre/chaixiujuan/gzb/ownEnv/mmgzb/mmcls_setting/mmcls_datasets/pipelines/__init__.py /lustre/chaixiujuan/gzb/mmlab/mmclassification/mmcls/datasets/pipelines/__init__.py
#ln -s /lustre/chaixiujuan/gzb/ownEnv/mmgzb/mmcls_setting/mmcls_datasets/pipelines/transforms.py /lustre/chaixiujuan/gzb/mmlab/mmclassification/mmcls/datasets/pipelines/transforms.py

# train
#ln -s /lustre/chaixiujuan/gzb/ownEnv/mmgzb/mmcls_setting/tools/train.py /lustre/chaixiujuan/gzb/mmlab/mmclassification/tools/train.py


# for mmaction2
# checkpoints
ln -s /lustre/chaixiujuan/gzb/checkpoints_all /lustre/chaixiujuan/gzb/mmlab/mmaction2/checkpoints
ln -s /lustre/chaixiujuan/gzb/ownEnv/mmgzb /lustre/chaixiujuan/gzb/mmlab/mmaction2/mmgzb

# configs
ln -s /lustre/chaixiujuan/gzb/ownEnv/mmgzb/mmaction2_setting/configs_har /lustre/chaixiujuan/gzb/mmlab/mmaction2/configs/configs_har