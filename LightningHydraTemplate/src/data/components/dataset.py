from torch.utils.data import Dataset
import pandas as pd
import torch
import os


class MyDataset(Dataset):
    def __init__(self, df, data_folder_path, dtype):
        self.df = df
        self.path = os.path.join(data_folder_path, dtype)
        self.data_folder_path = data_folder_path
        self.dtype = dtype

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df.iloc[idx]


class Collate_fn:
    def __init__(self, tokenizer, dtype):
        self.tokenizer = tokenizer
        self.dtype = dtype

    def __call__(self, batch):
        data = pd.DataFrame(batch)
        if self.dtype != "test":
            data = {
                "id": data.id.to_list(),
                "context": data.context.to_list(),
                "question": data.question.to_list(),
                "answers": [
                    {"text": [text], "answer_start": [answer_start]}
                    for text, answer_start in zip(
                        data.answer.to_list(), data.apply(lambda x: x.context.find(x.answer), axis=1).to_list()
                    )
                ],
            }
            data.update(self.preprocess_function(data))
            return data
        else:
            batch = pd.DataFrame(batch)
            output = self.tokenizer([str(i) for i in batch.X.to_list()], padding="max_length", max_length=512, truncation=True)
            output.update({"Y": batch["Y"].values, "label": batch["label"].values})
            return {k: torch.tensor(v) for k, v in output.items()}

    def preprocess_function(self, raw_text):
        inputs = self.tokenizer(
            raw_text["question"],
            raw_text["context"],
            max_length=1536,
            truncation="only_second",
            return_offsets_mapping=True,
            padding="max_length",
            return_tensors="pt",
        )

        offset_mapping = inputs.pop("offset_mapping")
        answers = raw_text["answers"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            answer = answers[i]
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label it (0, 0)
            if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = torch.tensor(start_positions)
        inputs["end_positions"] = torch.tensor(end_positions)
        return inputs
