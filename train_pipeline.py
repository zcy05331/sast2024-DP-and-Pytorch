import json
import math
import threading

import numpy as np
import torch
import torch.nn as nn
import wandb
from tokenizers import Tokenizer
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")
tokenizer.enable_padding(length=256)

device = "cuda" if torch.cuda.is_available() else "cpu"


def deal_segment(inputlist, savelist):
    for item in inputlist:
        tmp_dict = json.loads(item)
        savelist.append(
            (
                torch.tensor(tokenizer.encode(tmp_dict["content"]).ids[:256]),
                torch.tensor(
                    [1 - tmp_dict["label"], tmp_dict["label"]], dtype=torch.float32
                ),
            )
        )


class MyDataSet(Dataset):
    def __init__(self, file: str):
        self.data = []
        with open(file, "r", encoding="utf-8") as fin:
            inputlist = list(fin)
            tlist = [
                threading.Thread(
                    target=deal_segment,
                    args=(
                        inputlist[1000 * i : 1000 * (i + 1)],
                        self.data,
                    ),
                )
                for i in range(math.ceil(len(inputlist) / 1000))
            ]
            for t in tqdm(tlist):
                t.start()
            for t in tlist:
                t.join()
        fin.close()

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

train_set = MyDataSet(file="dataset/train.jsonl")
test_set = MyDataSet(file="dataset/test.jsonl")
train_loader = DataLoader(dataset=train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=32, shuffle=True)


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

model = MyModel().to(device)


loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=1e-3)


wandb.init(
    project="summer_guide",
    config={
        "learning_rate": 1e-3,
        "architecture": "MLP",
        "dataset": "amazon-plarity",
        "epochs": 3,
    },
)

for i in range(3):
    print(f"--------------iteration {i+1}----------------")
    for batch, (X, y) in enumerate(tqdm(list(train_loader))):
        pred = model(X.to(device))
        loss = loss_fn(pred, y.to(device))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 50 == 0:
            total = 0
            with torch.no_grad():
                for batch, (X, y_t) in enumerate(test_loader):
                    pred_t = model(X.to(device))
                    total += np.sum((torch.argmax(pred_t.cpu(), 1) == torch.argmax(y_t.cpu(), 1)).numpy())
            wandb.log(
                {
                    "loss": loss,
                    "train_acc": np.mean((torch.argmax(pred.cpu(), 1) == torch.argmax(y.cpu(), 1)).numpy()),
                    "test_acc": total/20000
                }
            )
    torch.save(model.state_dict(), "results/model.pt")

wandb.finish()
