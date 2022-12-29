pipeline=[
    dict(type='LoadMedicalDataFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(512, 512), ratio_range=(0.5, 2.0)),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type='MSDDataset',
        pipeline=pipeline,
        data_root='/home/akiyo/cached_dataset/MSD/Task01_BrainTumour',
        img_dir='imagesTr',
        ann_dir='labelsTr',
        img_suffix='.nii.gz',
        seg_map_suffix='.nii.gz',
        split='dataset.json',
    ),
    val=dict(
        type='MSDDataset',
        pipeline=pipeline,
        img_dir="/home/akiyo/cached_dataset/MSD/Task01_BrainTumour/imagesTr"
    ),
)