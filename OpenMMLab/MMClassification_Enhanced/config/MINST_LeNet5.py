_base_ = '/home/akiyo/nfs/zhang/github/mmclassification/configs/lenet/lenet5_mnist.py'

train = False

gpu_ids=[0]
device='cuda'

work_dir = '/home/akiyo/sandbox/work_dirs/MINST'
data_root = "/home/akiyo/nfs/zhang/dataset/MNIST/raw"
data = dict(
    samples_per_gpu=128,
    workers_per_gpu=4,
    train=dict(data_prefix=data_root),
    val=dict(data_prefix=data_root),
    test=dict(data_prefix=data_root)
)

if not train:
    load_from = '/home/akiyo/sandbox/work_dirs/MINST/latest.pth'