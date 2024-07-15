import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
import json
from tqdm import tqdm

tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")
tokenizer.enable_padding(length=256)

class MyDataSet(Dataset):
    def __init__(self, file: str):
        self.data = []
        self.label = []
        with open(file, "r", encoding='utf-8') as fin:
            print(f"load data {file}")
            for line in tqdm(list(fin)):
                tmp_dict = json.loads(line)
                self.data.append(torch.tensor(tokenizer.encode(tmp_dict["content"]).ids[:256]))
                self.label.append(torch.tensor([1-tmp_dict["label"], tmp_dict['label']], dtype=torch.float32))
                
    def __getitem__(self, index):
        return self.data[index], self.label[index]
    
    def __len__(self):
        return len(self.data)

train_set = MyDataSet(file="dataset/train.jsonl")
test_set = MyDataSet(file="dataset/test.jsonl")


train_loader = DataLoader(dataset=train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=32, shuffle=True)

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings=50000, embedding_dim=64)
        self.layer1 = nn.Linear(256*64, 64*32)
        self.ac1 = nn.ReLU()
        self.layer2 = nn.Linear(64*32, 16*16)
        self.ac2 = nn.ReLU()
        self.out = nn.Linear(16*16, 2)
        
    def forward(self, data):
        hidden = self.emb(data).view(-1, 64*256)
        return self.out(self.ac2(self.layer2(self.ac1(self.layer1(hidden)))))
    
    
model = MyModel().cuda()

from torch.optim import Adam

loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=1e-3)

import wandb
import numpy as np

wandb.init(
    project="summer_guide",
    config={
        "learning_rate": 1e-3,
        "architecture": "MLP",
        "dataset": "amazon-plarity",
        "epochs": 3,
    },
)

for i in tqdm(range(3)):
    for batch, (X, y) in enumerate(train_loader):
        pred = model(X.cuda())
        loss = loss_fn(pred, y.cuda())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 50 == 0:
            total = 0
            with torch.no_grad():
                for batch, (X, y_t) in enumerate(test_loader):
                    pred_t = model(X.cuda())
                    total += np.sum((torch.argmax(pred_t.cpu(), 1) == torch.argmax(y_t.cpu(), 1)).numpy())
            wandb.log(
                {
                    "loss": loss,
                    "train_acc": np.mean((torch.argmax(pred.cpu(), 1) == torch.argmax(y.cpu(), 1)).numpy()),
                    "test_acc": total/20000
                }
            )
    torch.save(model, "results/model.pt")
    
wandb.finish()