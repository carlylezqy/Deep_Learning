from mmcv.utils import Registry, build_from_cfg
MODELS = Registry('models')
@MODELS.register_module()
class ResNet:
    pass