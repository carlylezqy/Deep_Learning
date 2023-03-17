from mmengine.registry import Registry

def basic_1():
    backbones = Registry('backbone')

    #[1]: as a decorator
    @backbones.register_module()
    class ResNet:
        pass
    
    #[2]: class
    @backbones.register_module(name='ResNetFunction')
    class ResNetFunction:
        pass

    #[3]: as a normal function
    class ResNetFromNormalFunction:
        pass
    backbones.register_module(module=ResNetFromNormalFunction)

    print(backbones)

def basic_2():
    MODELS = Registry('models')

    @MODELS.register_module()
    class ResNet:
        pass
    @MODELS.register_module()
    def resnet50():
        pass
    
    resnet = MODELS.build(dict(type='ResNet'))
    resnet = MODELS.build(dict(type='resnet50'))

    # hierarchical registry
    DETECTORS = Registry('detectors', parent=MODELS, scope='det')
    @DETECTORS.register_module()
    class FasterRCNN:
        pass
    fasterrcnn = DETECTORS.build(dict(type='FasterRCNN'))

    # add locations to enable auto import
    DETECTORS = Registry(
        'detectors', parent=MODELS, scope='det', 
        locations=['det.models.detectors']
    )
    # define this class in 'det.models.detectors'
    @DETECTORS.register_module()
    class MaskRCNN:
        pass
    # The registry will auto import det.models.detectors.MaskRCNN
    fasterrcnn = DETECTORS.build(dict(type='det.MaskRCNN'))

if __name__ == '__main__':
    basic_2()