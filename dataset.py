import json
from tqdm import tqdm
from torch.utils.data import Dataset


class SCGPTDatasetForTrain(Dataset):
    def __init__(
        self,
        tokenizer,
        dataset_path,
        dataset_type,
        max_length
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dataset_type = dataset_type
        self.dataset = self.read_dataset(dataset_path)
        self.dataset = self.prepare_dataset()

    def read_dataset(self, dataset_path):
        with open(dataset_path) as f:
            data = json.load(f)
        return data

    def prepare_dataset(self):
        dataset = list()
        if self.dataset_type == 'damd':
            for _, dialogue in tqdm(self.dataset.items()):
                for turn in dialogue['log']:
                    dataset.append((
                        turn['sys_act'],
                        turn['resp_delex']
                    ))
        elif self.dataset_type == 'raw':
            for entry in tqdm(self.dataset):
                dataset.append((
                    entry['sys_act'],
                    entry['resp']
                ))

        tokenized_dataset = list()
        for sys_act, resp in dataset:
            text = sys_act + self.tokenizer.bos_token + resp + self.tokenizer.eos_token
            text_tokenized = self.tokenizer(text, padding='max_length', return_tensors='pt',
                                            max_length=self.max_length)

            input_ids = text_tokenized['input_ids'].clone().detach().squeeze()
            attention_mask = text_tokenized['attention_mask'].clone().detach().squeeze()
            labels = text_tokenized['input_ids'].clone().detach().squeeze()

            start_of_resp_index = (input_ids == self.tokenizer.bos_token_id).nonzero(as_tuple=True)[0].item() + 1
            labels[:start_of_resp_index] = -100

            tokenized_dataset.append((input_ids, attention_mask, labels))
        return tokenized_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        input_ids, attention_mask, label = self.dataset[index]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label
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
        return len(self.dataset['labels'])

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
