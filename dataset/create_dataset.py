from torch.utils.data import Dataset,DataLoader
from datasets import load_dataset
import torch
import pandas as pd
import lightning.pytorch as pl
import copy
import json
from sklearn.model_selection import train_test_split


class MetaTTP(torch.utils.data.Dataset):
    def __init__(self, dataset, platform: str = 'vllm', pad_token: str = "eos", padding_side: str = "left", max_length: int = 4096,tokenizer=None, system_prompt:str="You are a helpful assistant."):
        self.dataset = dataset 
        self.platform = platform
        self.pad_token = pad_token
        self.padding_side = padding_side
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt

        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        tokenized_input=0
        if self.platform == 'huggingface':
            print("Tokenizing input for huggingface platform")
            messages = []
            if self.system_prompt != "pre_defined":
                messages.append({
                    "role": "system",
                    "content": (
                        self.system_prompt
                        if self.system_prompt != "default"
                        else "You are a helpful assistant."
                    )
                })
            
            messages.append({"role": "user", "content": item["mutated_prompt"]})
            print("messages: ", messages)

            text = self.tokenizer.apply_chat_template(
                                messages,
                                tokenize=False,
                                add_generation_prompt=True,
                            )
            if self.pad_token == "eos":
                self.tokenizer.pad_token = self.tokenizer.eos_token
                print("pad_token set to :", self.tokenizer.pad_token)
            template_tokenized = self.tokenizer(text, max_length=self.max_length, padding='max_length', padding_side="left", return_tensors='pt')        
            tokenized_input=template_tokenized
            # print("input id shape: ", tokenized_input['input_ids'].shape)
            # print("attention_mask shape: ", tokenized_input['attention_mask'].shape)

        return {
            'tokenized_input': tokenized_input,
            'prompt': item["mutated_prompt"],
            'base_prompt': item["base_prompt"],
            "mitre_category": item["mitre_category"],
            "ttp_id_name_mapping": item["ttp_id_name_mapping"]
        }

class MetaTTP_eval(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset 
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        # print("item in eval dataset: ", item)
        try:
            return {
                'prompt': item["prompt"],
                'response': item["response"],
                "mitre_category": item["mitre_category"],
                "ttp_name": item["ttp_name"],
                "ttp_id": item["ttp_id"]
            }
        except:
            return {
                'prompt': item["prompt"],
                'response': item["response"]
            }

class MetaTTP_attack(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset 
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        # print("item in eval dataset: ", item)
        return item
    

class RMCbench(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset 
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        # print("item in eval dataset: ", item)
        return {
            'prompt': item["prompt"],
            'category': item["malicious categories"],
            "language": item["language"],
        } 

class RMCbench_eval(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset 
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        # print("item in eval dataset: ", item)
        try:
            return {
                'prompt': item["prompt"],
                'response': item["response"],
                "category": item["category"],
                "language": item["language"]
            }                
        except:
            return {
                'prompt': item["prompt"],
                'response': item["response"]
            } 

class Redcode(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset 
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        # print("item in eval dataset: ", item)
        return {
            'prompt': item["prompt"],
            'category': item["Category"]
        }


class Redcode_eval(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset 
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        # print("item in eval dataset: ", item)
        try:
            return {
                'prompt': item["prompt"],
                'response': item["response"],
                "category": item["category"],
            }
        except:
            return {
                'prompt': item["prompt"],
                'response': item["response"]
            } 

 

class LitDataModule(pl.LightningDataModule):
    def __init__(self,batch_size: int = 1, 
                 num_worker: int = 4, 
                 platform: str = 'vllm', 
                 pad_token: str = "eos", 
                 padding_side: str = "left", 
                 max_length: int = 4096, 
                 tokenizer=None,
                 system_prompt:str="You are a helpful assistant."):
        super().__init__()
        self.batch_size = batch_size
        self.num_worker = num_worker
        self.platform = platform
        self.pad_token = pad_token
        self.padding_side = padding_side
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        

        
    def setup(self, dirctory:str, dataset:str = 'MetaTTP', stage: str = 'test', start_point:int = 0):
            if dataset == "MetaTTP":
                if stage == 'test':
                    with open(dirctory, 'r') as f:
                        test_data = json.load(f)  
                    test_data = test_data[start_point:] 
                    self.test_dataset = MetaTTP(test_data,self.platform,self.pad_token,self.padding_side,self.max_length,tokenizer=self.tokenizer, system_prompt=self.system_prompt)
                elif stage == 'eval':
                    eval_data = []
                    with open(dirctory, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:  # skip blank lines
                                continue
                            obj = json.loads(line)
                            eval_data.append(obj)
                    eval_data = eval_data[start_point:]
                    print("length of dataset: ", len(eval_data))
                    print('start_point: ', start_point)
                    self.eval_dataset = MetaTTP_eval(eval_data)
            elif dataset == "attack":
                
                if stage == 'fit':
                    print("prepaing train data")
                    train_data = []
                    with open("dataset/attack_train.jsonl", "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:  # skip blank lines
                                continue
                            obj = json.loads(line)
                            train_data.append(obj)
                    self.train_dataset = MetaTTP_attack(train_data)

                elif stage == 'test':
                    test_data = []
                    with open("dataset/attack_test.jsonl", "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:  # skip blank lines
                                continue
                            obj = json.loads(line)
                            test_data.append(obj)
                    self.test_dataset = MetaTTP_attack(test_data)

            elif dataset == 'infill':
                if stage == 'test':
                    with open(dirctory, 'r') as f:
                        test_data = json.load(f)  
                    test_data = test_data[start_point:] 
                    self.test_dataset = RMCbench(test_data)
                elif stage == 'eval':
                    eval_data = []
                    with open(dirctory, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:  # skip blank lines
                                continue
                            obj = json.loads(line)
                            eval_data.append(obj)
                    eval_data = eval_data[start_point:]
                    print("length of dataset: ", len(eval_data))
                    print('start_point: ', start_point)
                    self.eval_dataset = RMCbench_eval(eval_data)
            
            elif dataset == "translate":
                if stage == 'test':
                    with open(dirctory, 'r') as f:
                        test_data = json.load(f)  
                    test_data = test_data[start_point:] 
                    self.test_dataset = RMCbench(test_data)
                elif stage == 'eval':
                    eval_data = []
                    with open(dirctory, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:  # skip blank lines
                                continue
                            obj = json.loads(line)
                            eval_data.append(obj)
                    eval_data = eval_data[start_point:]
                    print("length of dataset: ", len(eval_data))
                    print('start_point: ', start_point)
                    self.eval_dataset = RMCbench_eval(eval_data)
            
            elif dataset == "complete":
                if stage == 'test':
                    with open(dirctory, 'r') as f:
                        test_data = json.load(f)  
                    test_data = test_data[start_point:] 
                    self.test_dataset = Redcode(test_data)
                elif stage == 'eval':
                    eval_data = []
                    with open(dirctory, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:  # skip blank lines
                                continue
                            obj = json.loads(line)
                            eval_data.append(obj)
                    eval_data = eval_data[start_point:]
                    print("length of dataset: ", len(eval_data))
                    print('start_point: ', start_point)
                    self.eval_dataset = Redcode_eval(eval_data)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_worker,drop_last=False,pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_worker,drop_last=False,pin_memory=True)
    
    def eval_dataloader(self):
        return DataLoader(self.eval_dataset, batch_size=self.batch_size, num_workers=self.num_worker,drop_last=False,pin_memory=True)