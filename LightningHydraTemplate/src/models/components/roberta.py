import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel


class Roberta(nn.Module):
    def __init__(
        self,
        name: str = "klue/roberta-large",
    ) -> None:
        super(Roberta, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name, add_pooling_layer=False)
        for name, param in self.model.named_parameters():
            param.requires_grad = False
        self.qa_output = QaOutput(1024)

    def forward(self, x) -> torch.Tensor:
        output = self.model(**x)
        hidden_state = output.last_hidden_state  # (bs, max_query_len, dim)
        return self.qa_output(hidden_state)


class QaOutput(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.classifier = nn.Sequential(
            *[
                nn.Linear(hidden_size, 2, bias=False),
            ]
        )

    def forward(self, hidden_states):
        logits = self.classifier(hidden_states)  # (bs, max_query_len, 2)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()  # (bs, max_query_len)
        end_logits = end_logits.squeeze(-1).contiguous()  # (bs, max_query_len)
        return {"start_logits": start_logits, "end_logits": end_logits}


if __name__ == "__main__":
    net = Roberta()
    print("hi")
