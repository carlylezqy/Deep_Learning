from transformers.adapters import AdapterConfig
from transformers import RobertaConfig, RobertaAdapterModel
from transformers import AutoModelForImageClassification

#onfig = RobertaConfig.from_pretrained("roberta-base", num_labels=2,)
config = AdapterConfig(mh_adapter=True, output_adapter=True, reduction_factor=16, non_linearity="relu")

model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k")
model.add_adapter("bottleneck_adapter", config=config)
print(model)
