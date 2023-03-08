custom_imports = dict(
    imports=[
        'utils.import_test',
        'models.resnet50',
    ], 
    allow_failed_imports=False
)

model = dict(
    type='MMResNet50',
    pretrained=True
)