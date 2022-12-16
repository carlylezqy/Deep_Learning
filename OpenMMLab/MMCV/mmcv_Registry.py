from mmcv.utils import Registry

"""
1. A Simple Example
"""

MODELS = Registry('models')
@MODELS.register_module()
class Operations:
    def __init__(self):
        pass

    def hello(self):
        print("hello")

    def add(self, x, y):
        return x + y

operate = MODELS.build(dict(type='Operations'))
print(operate.hello())
print(operate.add(1, 2))

@MODELS.register_module()
def add(x, y):
    return f"x={x}, y={y}, x+y={x+y}"

dic = dict(type='add', x=1, y=2)
result = MODELS.build(dic)
print(result)

"""
2. Customize Build Function
"""

# create a build function
def build_converter(cfg, registry, *args, **kwargs):
    print("HELLO<PPP")
    cfg_ = cfg.copy()
    converter_type = cfg_.pop('type')
    if converter_type not in registry:
        raise KeyError(f'Unrecognized converter type {converter_type}')
    else:
        converter_cls = registry.get(converter_type)

    converter = converter_cls(*args, **kwargs, **cfg_)
    return converter

# create a registry for converters and pass ``build_converter`` function
CONVERTERS = Registry('converter', build_func=build_converter)

@CONVERTERS.register_module()
def add(x, y):
    return f"x={x}, y={y}, x+y={x+y}"

dic = dict(type='add', x=1, y=2)
result = MODELS.build(dic)
print(result)

"""
3. Hierarchy Registry
"""
