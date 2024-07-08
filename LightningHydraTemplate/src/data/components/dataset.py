from torch.utils.data import Dataset
import pandas as pd
import torch
import json
import os


class MyDataset(Dataset):
    def __init__(self, df, tokenizer, data_folder_path, dtype):
        self.df = df
        self.tokenizer = tokenizer
        self.path = os.path.join(data_folder_path, dtype)
        self.data_folder_path = data_folder_path
        self.dtype = dtype

        if not os.path.isdir(self.path):
            os.makedirs(self.path)
            idx = 0
            for _, data in self.df.iterrows():
                data = {
                    "id": [data.id],
                    "context": [data.context],
                    "question": ["<|start_header_id|>user<|end_header_id|>" + data.question],
                    "answers": [{"text": [data.answer], "answer_start": [data.context.find(data.answer)]}],
                }
                preprocess_data = self.preprocess_function(data)
                for i in range(len(preprocess_data["start_positions"])):
                    sub_data = {k: v[i].tolist() for k, v in preprocess_data.items()}
                    sub_data.update(data)
                    if not sub_data["start_positions"] == sub_data["end_positions"]:
                        if sub_data["answers"][0]["text"][0] == tokenizer.decode(
                            sub_data["input_ids"][sub_data["start_positions"] : sub_data["end_positions"] + 1]
                        ):
                            with open(os.path.join(data_folder_path, dtype, f"{idx}.json"), "w") as f:
                                json.dump(sub_data, f, indent=4)
                                idx += 1

    def __len__(self):
        return len(os.listdir(self.path))

    def __getitem__(self, idx):
        return json.load(open(os.path.join(self.path, f"{idx//10}.json"), "r"))

    def preprocess_function(self, raw_text, max_length=384, stride=128):
        inputs = self.tokenizer(
            raw_text["question"],
            raw_text["context"],
            max_length=max_length,
            stride=stride,
            truncation="only_second",
            return_offsets_mapping=True,
            return_overflowing_tokens=True,
            padding="max_length",
            return_tensors="pt",
        )

        offset_mapping = inputs.pop("offset_mapping")
        overflow_to_sample_mapping = inputs["overflow_to_sample_mapping"]
        answers = raw_text["answers"]
        start_positions = []
        end_positions = []
        context_position = []

        for i, offset in enumerate(offset_mapping):
            example_index = overflow_to_sample_mapping[i]
            answer = answers[example_index]
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i) + [None]

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            context_position += [(context_start, context_end)]

            # If the answer is not fully inside the context, label it (0, 0)
            if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
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
        inputs["context_position"] = torch.tensor(context_position)
        return inputs


class Collate_fn:
    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, batch):
        data = pd.DataFrame(batch)

        return {
            "input_ids": torch.tensor(data["input_ids"]),
            "attention_mask": torch.tensor(data["attention_mask"]),
            "overflow_to_sample_mapping": data["overflow_to_sample_mapping"],
            "start_positions": torch.tensor(data["start_positions"]),
            "end_positions": torch.tensor(data["end_positions"]),
            "context_position": data["context_position"],
            "id": data["id"],
            "context": data["context"],
            "question": data["question"],
            "answers": data["answers"],
        }
