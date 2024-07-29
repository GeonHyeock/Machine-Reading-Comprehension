from torch.utils.data import Dataset
from src.models.roberta_module import normalize_answer
from tqdm import tqdm
import pandas as pd
import torch
import json
import os


class MyDataset(Dataset):
    def __init__(self, df, tokenizer, data_folder_path, dtype, fold, max_length=384, stride=128):
        self.df = df
        self.tokenizer = tokenizer
        self.path = os.path.join(data_folder_path, dtype, fold)
        self.data_folder_path = data_folder_path
        self.dtype = dtype

        remove_id, use_id = [], []
        if not os.path.isdir(self.path):
            os.makedirs(self.path)
            idx = 0
            for _, data in tqdm(self.df.iterrows(), total=len(self.df), desc=f"{self.dtype} data preprocess"):
                data = {
                    "id": [data.id],
                    "context": [data.context],
                    "question": [data.question],
                    "answers": ["" if self.dtype == "test" else data.answer],
                }
                preprocess_data = self.preprocess_function(data, max_length, stride)
                for i in range(len(preprocess_data["start_positions"])):
                    sub_data = {k: v[i].tolist() for k, v in preprocess_data.items()}
                    sub_data.update(data)
                    if self.dtype == "test":
                        with open(os.path.join(self.path, f"{idx}.json"), "w", encoding="utf-8") as f:
                            json.dump(sub_data, f, indent=4)
                            idx += 1

                    else:
                        if sub_data["start_positions"] != sub_data["end_positions"]:
                            text = sub_data["answers"][0]
                            decode_text = sub_data["input_ids"][sub_data["start_positions"] : sub_data["end_positions"] + 1]
                            decode_text = self.tokenizer.decode(decode_text)
                            if normalize_answer(text) == normalize_answer(decode_text):
                                with open(os.path.join(self.path, f"{idx}.json"), "w", encoding="utf-8") as f:
                                    json.dump(sub_data, f, indent=4)
                                    idx += 1
                                    use_id += sub_data["id"]
                        else:
                            remove_id += sub_data["id"]
            remove_id = list(set(remove_id) - set(use_id))
            pd.DataFrame(remove_id).to_csv(os.path.join(data_folder_path, f"{dtype}_{fold}_remove.csv"), index=False)

    def __len__(self):
        return len(os.listdir(self.path))

    def __getitem__(self, idx):
        return json.load(open(os.path.join(self.path, f"{idx}.json"), "r"))

    def preprocess_function(self, raw_text, max_length=384, stride=128):
        inputs = self.tokenizer(
            raw_text["question"],
            raw_text["context"],
            max_length=max_length,
            stride=stride,
            truncation="only_second",
            return_overflowing_tokens=True,
            padding="max_length",
            return_tensors="pt",
        )

        start_positions, end_positions, context_position = [], [], []
        for i, input_ids in enumerate(inputs["input_ids"]):
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

            start, end = self.start_end(raw_text["answers"][0], input_ids, context_start, context_end)
            if start == 0 and end == 0:
                start, end = self.start_end(" " + raw_text["answers"][0], input_ids, context_start, context_end)

            start_positions.append(start)
            end_positions.append(end)

        inputs["start_positions"] = torch.tensor(start_positions)
        inputs["end_positions"] = torch.tensor(end_positions)
        inputs["context_position"] = torch.tensor(context_position)
        return inputs

    def start_end(self, text, input_ids, context_start, context_end):
        start, end = 0, 0
        answer = torch.tensor(self.tokenizer(text)["input_ids"][1:])
        answer_length = len(answer)
        for idx in range(context_start, context_end - answer_length):
            if (input_ids[idx : idx + answer_length] == answer).all():
                start = idx
                end = idx + answer_length - 1
        return start, end


class Collate_fn:
    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, batch):
        data = pd.DataFrame(batch)
        return {
            "input_ids": torch.tensor(data["input_ids"]),
            "attention_mask": torch.tensor(data["attention_mask"]),
            "start_positions": torch.tensor(data["start_positions"]),
            "end_positions": torch.tensor(data["end_positions"]),
            "context_position": data["context_position"],
            "id": data["id"],
            "context": data["context"],
            "question": data["question"],
            "answers": data["answers"],
        }
