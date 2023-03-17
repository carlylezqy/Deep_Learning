from mmcv.runner import HOOKS
from mmcv.runner.hooks import Hook
import loralib as lora
import time

@HOOKS.register_module()
class LoRAHook2(Hook):
    def __init__(self, apply_lora=True, lora_r=8, lora_alpha=8):
        self.apply_lora = apply_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
    
    def before_run(self, runner):
        runner.logger.info("LoRA Hook: before_run")
        model = runner.model.module

        if self.apply_lora:
            lora.mark_only_lora_as_trainable(model)
            model.head.fc.weight.requires_grad = True
            model.head.fc.bias.requires_grad = True
            # runner.model.find_unused_parameters = True
            # device = next(model.parameters()).device

            # for n in model.backbone.state_dict():
            #     if "qkv.weight" in n:
            #         module_name = n.replace(".qkv.weight", "")
            #         m = model.backbone.get_submodule(module_name).qkv
            #         in_features, out_features = m.in_features, m.out_features
            #         lora_layer = lora.Linear(in_features, out_features, r=self.lora_r, lora_alpha=self.lora_alpha, merge_weights=False)
            #         lora_layer.to(device)
            #         model.backbone.get_submodule(module_name).qkv = lora_layer
            #     elif "proj.weight" in n:
            #         module_name = n.replace(".proj.weight", "")
            #         m = model.backbone.get_submodule(module_name).proj
            #         in_features, out_features = m.in_features, m.out_features
            #         lora_layer = lora.Linear(in_features, out_features, r=self.lora_r, lora_alpha=self.lora_alpha, merge_weights=False)
            #         lora_layer.to(device)
            #         model.backbone.get_submodule(module_name).proj = lora_layer

            #model_state_dict = lora.lora_state_dict(model)

        #lora.mark_only_lora_as_trainable(model)
        #model.head.fc.weight.requires_grad = True
        #model.head.fc.bias.requires_grad = True

        runner.logger.info("===========Parameters Summary===========")
        for n, p in model.named_parameters():
            runner.logger.info(f"{n},{p.numel()},{p.requires_grad}")
        runner.logger.info("========================================")
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        runner.logger.info(f"LoRA Hook: Total: {total_num}, Trainable: {trainable_num}")

        self.time_start = time.time()

    def after_run(self, runner):
        self.time_end = time.time()
        runner.logger.info(f"[Time] {self.time_end - self.time_start}s")