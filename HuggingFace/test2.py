from transformers.adapters import AdapterConfig
from transformers import RobertaConfig, RobertaAdapterModel
from transformers import AutoModelForImageClassification

from transformers.adapters.layer import AdapterLayerBase

#onfig = RobertaConfig.from_pretrained("roberta-base", num_labels=2,)
config = AdapterConfig(mh_adapter=True, output_adapter=True, reduction_factor=16, non_linearity="relu")

model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k")

# for i, layer in model.vit.iter_layers():
#     for name, module in layer.named_modules():
#         if isinstance(module, AdapterLayerBase):
#             try:
#                 print(name, module.__class__, module.in_features, module.out_features)
#             except:
#                 print(name, module.__class__)
#     break

model.add_adapter("bottleneck_adapter", config=config)


