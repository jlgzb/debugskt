model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='STGCNSkip2TCNStride12',
        gcn_adaptive='init',
        gcn_with_res=True,
        tcn_type='mstcn',
        graph_cfg=dict(layout='cocolr', mode='spatial_lr3a'),
        num_stages=5,
        inflate_stages=[2, 4],
        down_stages=[2, 4]),
    cls_head=dict(type='GCNHead', num_classes=60, in_channels=256))
num_clips_test=20
data = dict(
    videos_per_gpu=96,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type='PoseDataset',
            ann_file='data/nturgbd/ntu60_hrnet.pkl',
            pipeline=[
                dict(type='PreNormalize2D'),
                dict(type='GenSkeFeat', dataset='coco', feats=['j']),
                dict(type='UniformSample', clip_len=100),
                dict(type='PoseDecode'),
                dict(type='FormatGCNInput', num_person=2),
                dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
                dict(type='ToTensor', keys=['keypoint'])
            ],
            split='xsub_train')),
    val=dict(
        type='PoseDataset',
        ann_file='data/nturgbd/ntu60_hrnet.pkl',
        pipeline=[
            dict(type='PreNormalize2D'),
            dict(type='GenSkeFeat', dataset='coco', feats=['j']),
            dict(
                type='UniformSample',
                clip_len=100,
                num_clips=1,
                test_mode=True),
            dict(type='PoseDecode'),
            dict(type='FormatGCNInput', num_person=2),
            dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['keypoint'])
        ],
        split='xsub_val'),
    test=dict(
        type='PoseDataset',
        ann_file='data/nturgbd/ntu60_hrnet.pkl',
        pipeline=[
            dict(type='PreNormalize2D'),
            dict(type='GenSkeFeat', dataset='coco', feats=['j']),
            dict(
                type='UniformSample',
                clip_len=100,
                num_clips=num_clips_test,
                test_mode=True),
            dict(type='PoseDecode'),
            dict(type='FormatGCNInput', num_person=2),
            dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['keypoint'])
        ],
        split='xsub_val'))
optimizer = dict(
    type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=True)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='CosineAnnealing', min_lr=0, by_epoch=False)
total_epochs = 120
checkpoint_config = dict(interval=120)
evaluation = dict(interval=1, metrics=['top_k_accuracy'], topk=(1, 2))
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
log_level = 'INFO'
work_dir = './work_dirs/stgcn++/stgcn++_ntu60_xsub_hrnet/j-hrnet-cliplen100-gpu2-time5-bs96-lr01-gcn2tcn-stride12-stage5-24'
dist_params = dict(backend='nccl')
gpu_ids = range(0, 2)
