import tiktoken
import torch
from torch.utils.data import Dataset,DataLoader

class GPTDataset(Dataset):
    def __init__(self, text, tokenizer, window,  stride):
        self.text = text
        self.tokenizer = tokenizer
        self.window = window
        self.stride = stride
        self.x = []
        self.y = []
        self.data = self.tokenizer.encode(self.text, allowed_special={"<|endoftext|>"})
        for i in range(0, len(self.data) - self.window, self.stride):
            input_chunk = self.data[i:i+self.window]
            target_chunk = self.data[i+1:i+self.window+1]
            self.x.append(torch.tensor(input_chunk))
            self.y.append(torch.tensor(target_chunk))
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def create_dataloader(text, batch_size=2, window=256, stride=128, drop_last=True, shuffle=True):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDataset(text, tokenizer, window, stride)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,num_workers=0)
    return data_loader

