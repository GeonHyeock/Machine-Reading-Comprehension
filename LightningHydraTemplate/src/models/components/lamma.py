import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from peft import inject_adapter_in_model, LoraConfig


class lamma(nn.Module):
    def __init__(
        self,
        name="beomi/Llama-3-Open-Ko-8B-Instruct-preview",
        lora_module=["self_attn.q_proj", "self_attn.v_proj", "self_attn.k_proj", "self_attn.o_proj"],
        QaOutput_Version=1,
    ) -> None:
        super(lamma, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModel.from_pretrained(name)

        if QaOutput_Version == 2:
            self.qa_output = QaOutputV2(4096)
        else:
            self.qa_output = QaOutput(4096)

        if lora_module:
            lora_target = [
                name
                for name, module in self.model.named_modules()
                if ((isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d))) and any([m in name for m in lora_module])
            ]
            lora_config = LoraConfig(
                lora_alpha=8,
                lora_dropout=0.2,
                r=16,
                bias="lora_only",
                target_modules=lora_target,
            )
            self.model = inject_adapter_in_model(lora_config, self.model)

    def forward(self, x) -> torch.Tensor:
        output = self.model(**x)
        hidden_state = output.last_hidden_state  # (bs, max_query_len, dim)
        return self.qa_output(hidden_state)


class QaOutput(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.classifier = nn.Sequential(
            *[
                nn.Dropout(0.5),
                nn.Linear(hidden_size, 2, bias=False),
            ]
        )

    def forward(self, hidden_states):
        logits = self.classifier(hidden_states)  # (bs, max_query_len, 2)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()  # (bs, max_query_len)
        end_logits = end_logits.squeeze(-1).contiguous()  # (bs, max_query_len)
        return {"start_logits": start_logits, "end_logits": end_logits}


class QaOutputV2(nn.Module):
    def __init__(self, hidden_size, bottlenect=16):
        super().__init__()
        self.conv1 = nn.Sequential(
            *[
                nn.Dropout(0.2),
                nn.Conv1d(hidden_size, hidden_size // bottlenect, 3, 1, 1, 1),
                nn.GELU(),
                nn.Conv1d(hidden_size // bottlenect, hidden_size, 1, 1, 0, 1),
            ]
        )
        self.conv2 = nn.Sequential(
            *[
                nn.Dropout(0.2),
                nn.Conv1d(hidden_size, hidden_size // bottlenect, 3, 1, 1, 2),
                nn.GELU(),
                nn.Conv1d(hidden_size // bottlenect, hidden_size, 1, 1, 0, 1),
            ]
        )
        self.conv3 = nn.Sequential(
            *[
                nn.Dropout(0.2),
                nn.Conv1d(hidden_size, hidden_size // bottlenect, 3, 1, 1, 4),
                nn.GELU(),
                nn.Conv1d(hidden_size // bottlenect, hidden_size, 1, 1, 0, 1),
            ]
        )

        self.active = nn.GELU()

        self.classifier1 = nn.Sequential(
            *[
                nn.Dropout(0.5),
                nn.Linear(hidden_size, 2, bias=False),
            ]
        )

        self.classifier2 = nn.Sequential(
            *[
                nn.Dropout(0.5),
                nn.Linear(hidden_size, 2, bias=False),
            ]
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, a=0.7):
        x_ = x.permute(0, 2, 1)
        for conv in [self.conv1, self.conv1, self.conv1]:
            x_ = self.active(conv(x_) + x_)
        x_ = x_.permute(0, 2, 1)

        logits = self.softmax(self.classifier1(x)) * a + self.softmax(self.classifier2(x_)) * (1 - a)
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
