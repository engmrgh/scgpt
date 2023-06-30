import json
from tqdm import tqdm
from torch.utils.data import Dataset


class SCGPTDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        dataset_path,
        max_in_seq_length,
        max_out_seq_length
    ):
        self.tokenizer = tokenizer
        self.max_in_seq_length = max_in_seq_length
        self.max_out_seq_length = max_out_seq_length
        self.dataset = self.read_dataset(dataset_path)
        self.dataset = self.prepare_dataset()

    def read_dataset(self, dataset_path):
        with open(dataset_path) as f:
            data = json.load(f)
        return data

    def prepare_dataset(self):
        dataset = list()
        for _, dialogue in tqdm(self.dataset.items()):
            for turn in dialogue['log']:
                sys_act = turn['sys_act']
                resp_delex = turn['resp_delex']

                text = sys_act + self.tokenizer.bos_token + resp_delex
                text_tokenized = self.tokenizer(text, padding='max_length', return_tensors='pt',
                                                max_length=self.max_in_seq_length)

                input_ids = text_tokenized['input_ids'].clone().detach().squeeze()
                attention_mask = text_tokenized['attention_mask'].clone().detach().squeeze()
                lm_labels = text_tokenized['input_ids'].clone().detach().squeeze()

                start_of_resp_index = (input_ids == self.tokenizer.bos_token_id).nonzero(as_tuple=True)[0].item() + 1
                start_of_pad_index = (input_ids == self.tokenizer.pad_token_id).nonzero(as_tuple=True)[0][0].item()

                lm_labels[:start_of_resp_index] = -100
                input_ids[start_of_pad_index] = self.tokenizer.eos_token_id
                lm_labels[start_of_pad_index] = self.tokenizer.eos_token_id
                attention_mask[start_of_pad_index] = 1

                dataset.append((input_ids,
                                attention_mask,
                                lm_labels))

        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        input_ids, attention_mask, lm_label = self.dataset[index]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": lm_label
        }


class SCGPTDatasetForTest(Dataset):
    def __init__(
        self,
        tokenizer,
        dataset_path
    ):
        self.tokenizer = tokenizer
        self.dataset = self.read_dataset(dataset_path)
        self.dataset = self.prepare_dataset()

    def read_dataset(self, dataset_path):
        with open(dataset_path) as f:
            data = json.load(f)
        return data

    def prepare_dataset(self):
        prompts = list()
        labels = list()
        for _, dialogue in tqdm(self.dataset.items()):
            for turn in dialogue['log']:
                sys_act = turn['sys_act']
                resp_delex = turn['resp_delex']

                prompt = sys_act + self.tokenizer.bos_token
                prompts.append(prompt)
                labels.append(resp_delex)

        prompts_enc = self.tokenizer(prompts, padding=True, return_tensors='pt')
        prompts_enc['labels'] = labels
        prompts_enc['prompts'] = prompts
        return prompts_enc

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        input_ids = self.dataset['input_ids']
        attention_mask = self.dataset['attention_mask']
        prompt = self.dataset['prompts']
        label = self.dataset['labels']

        return {
            "input_ids": input_ids[index],
            "attention_mask": attention_mask[index],
            "prompt": prompt[index],
            "label": label[index]
        }
