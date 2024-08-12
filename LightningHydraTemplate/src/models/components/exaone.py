import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import inject_adapter_in_model, LoraConfig, AdaLoraConfig


class EXAONE(nn.Module):
    def __init__(
        self,
        name="LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct",
        lora_module=["attention.q_proj", "attention.k_proj", "attention.v_proj", "attention.out_proj"],
        lora_type="lora",
        slide_layer=-1,
        QaOutput_Version=1,
    ) -> None:
        super(EXAONE, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True)

        if slide_layer > 0:
            self.model.transformer.h = self.model.transformer.h[:slide_layer]

        if lora_module:
            lora_target = [
                name
                for name, module in self.model.named_modules()
                if ((isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d))) and any([m in name for m in lora_module])
            ]
            if lora_type == "lora":
                lora_config = LoraConfig(
                    r=16,
                    lora_alpha=8,
                    lora_dropout=0.2,
                    bias="lora_only",
                    target_modules=lora_target,
                )
            elif lora_type == "adalora":
                lora_config = AdaLoraConfig(
                    peft_type="ADALORA",
                    r=16,
                    lora_alpha=8,
                    lora_dropout=0.2,
                    bias="lora_only",
                    target_modules=lora_target,
                )
            self.model = inject_adapter_in_model(lora_config, self.model)

            if QaOutput_Version == 1:
                self.model.set_output_embeddings(QaOutput(4096))
            elif QaOutput_Version == 2:
                self.model.set_output_embeddings(QaOutputV2(4096))

    def forward(self, x) -> torch.Tensor:
        logits = self.model(**x).logits
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()  # (bs, max_query_len)
        end_logits = end_logits.squeeze(-1).contiguous()  # (bs, max_query_len)
        return {"start_logits": start_logits, "end_logits": end_logits}


class QaOutput(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.classifier = nn.Sequential(
            *[
                nn.Dropout(0.33),
                nn.Linear(hidden_size, 2, bias=False),
            ]
        )

    def forward(self, x):
        return self.classifier(x)


class QaOutputV2(nn.Module):
    def __init__(self, hidden_size, r=4):
        super().__init__()
        self.conv1 = nn.Sequential(
            *[
                nn.Conv1d(hidden_size, hidden_size // (r // 2), 1, 1),
                nn.GELU(),
            ]
        )
        self.conv2 = nn.Sequential(
            *[
                nn.Conv1d(hidden_size, hidden_size // r, 3, 1, 1),
                nn.GELU(),
            ]
        )
        self.conv3 = nn.Sequential(
            *[
                nn.Conv1d(hidden_size, hidden_size // r, 5, 1, 2),
                nn.GELU(),
            ]
        )

        self.dropout = nn.Dropout(0.33)

        self.classifier = nn.Sequential(
            *[
                nn.Dropout(0.5),
                nn.Linear(hidden_size, 2, bias=False),
            ]
        )

    def forward(self, x):
        x = self.dropout(x.permute(0, 2, 1))
        x = torch.cat([conv(x) for conv in [self.conv1, self.conv2, self.conv3]], dim=1)
        x = x.permute(0, 2, 1)

        logits = self.classifier(x)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
        return {"start_logits": start_logits, "end_logits": end_logits}


if __name__ == "__main__":
    net = lamma()
    ckpt = torch.load(
        "/shared/home/ai_math/2024MRC/LightningHydraTemplate/logs/train/runs/512_fold1_lamma/checkpoints/epoch_001.ckpt",
        map_location="cpu",
    )
    print(1)
