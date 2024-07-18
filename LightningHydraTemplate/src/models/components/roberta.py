import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from peft import inject_adapter_in_model, LoraConfig


class Roberta(nn.Module):
    def __init__(
        self,
        name: str = "klue/roberta-large",
        hidden_state: int = 1024,
        is_ensemble: bool = False,
        lora_module: list = [],
    ) -> None:
        super(Roberta, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.is_ensemble = is_ensemble

        if is_ensemble:
            self.model = Ensemble()
        else:
            self.model = AutoModel.from_pretrained(name, add_pooling_layer=False)

        if lora_module:
            lora_target = [
                name
                for name, module in self.model.named_modules()
                if ((isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d))) and any([m in name for m in lora_module])
            ]
            lora_config = LoraConfig(
                lora_alpha=8,
                lora_dropout=0.1,
                r=4,
                bias="lora_only",
                target_modules=lora_target,
            )
            self.model = inject_adapter_in_model(lora_config, self.model)
        else:
            for name, param in self.model.named_parameters():
                param.requires_grad = False

        self.qa_output = QaOutput(hidden_state, is_ensemble)

    def forward(self, x) -> torch.Tensor:
        output = self.model(**x)
        if not self.is_ensemble:
            output = output.last_hidden_state
        return self.qa_output(output)


class Ensemble(nn.Module):
    def __init__(self):
        super().__init__()
        self.small = AutoModel.from_pretrained("klue/roberta-small", add_pooling_layer=False)
        self.base = AutoModel.from_pretrained("klue/roberta-base", add_pooling_layer=False)
        self.large = AutoModel.from_pretrained("klue/roberta-large", add_pooling_layer=False)

    def forward(self, **x) -> torch.Tensor:
        output = [model(**x).last_hidden_state for model in [self.small, self.base, self.large]]
        return torch.concat(output, dim=-1)


class QaOutput(nn.Module):
    def __init__(self, hidden_size, is_ensemble):
        super().__init__()
        if is_ensemble:
            self.classifier = nn.Sequential(
                *[
                    nn.Linear(hidden_size, hidden_size),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.SiLU(),
                    nn.Linear(hidden_size // 2, 2, bias=False),
                ]
            )
        else:
            self.classifier = nn.Sequential(
                *[
                    nn.Linear(hidden_size, 2, bias=False),
                ]
            )
        for name, param in self.classifier.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, hidden_states):
        logits = self.classifier(hidden_states)  # (bs, max_query_len, 2)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()  # (bs, max_query_len)
        end_logits = end_logits.squeeze(-1).contiguous()  # (bs, max_query_len)
        return {"start_logits": start_logits, "end_logits": end_logits}


if __name__ == "__main__":
    net = Roberta()
