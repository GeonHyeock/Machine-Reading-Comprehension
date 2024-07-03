import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from peft import inject_adapter_in_model, LoraConfig


class SOLAR(nn.Module):
    def __init__(
        self,
        name="upstage/SOLAR-10.7B-Instruct-v1.0",
        lora_module=["self_attn.q_proj", "self_attn.v_proj", "self_attn.k_proj", "self_attn.o_proj"],
    ) -> None:
        super(SOLAR, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModel.from_pretrained(name)
        self.qa_output = QaOutput(4096)

        if lora_module:
            lora_target = [
                name
                for name, module in self.model.named_modules()
                if ((isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d))) and any([m in name for m in lora_module])
            ]
            lora_config = LoraConfig(
                lora_alpha=2,
                lora_dropout=0.1,
                r=2,
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
                nn.Dropout(0.1),
                nn.Linear(hidden_size, hidden_size // 8),
                nn.SiLU(),
                nn.Linear(hidden_size // 8, 2),
            ]
        )

    def forward(self, hidden_states):
        logits = self.classifier(hidden_states)  # (bs, max_query_len, 2)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()  # (bs, max_query_len)
        end_logits = end_logits.squeeze(-1).contiguous()  # (bs, max_query_len)
        return {"start_logits": start_logits, "end_logits": end_logits}


if __name__ == "__main__":
    net = SOLAR()
    print("hi")
