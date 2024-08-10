import json
import math
import threading
import numpy as np
import torch
import torch.nn as nn
import wandb
from tokenizers import Tokenizer
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Initialize tokenizer
torch.manual_seed(42)
tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")
tokenizer.enable_padding(length=256)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Function to deal with the segments of the dataset
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

# Custom Dataset class
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

# Create datasets and dataloaders
train_set = MyDataSet(file="dataset/train.jsonl")
test_set = MyDataSet(file="dataset/test.jsonl")
train_loader = DataLoader(dataset=train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=32, shuffle=True)

# Define LSTM-based Model
class MyLSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings=50000, embedding_dim=64)
        self.lstm = nn.LSTM(input_size=64, hidden_size=32, batch_first=True)
        # self.rnn = nn.RNN(input_size=64, hidden_size=8, num_layers=3, batch_first=True)
        self.fc = nn.Linear(32, 2)

    def forward(self, data):
        emb_out = self.emb(data)
        lstm_out, _ = self.lstm(emb_out)
        lstm_out = lstm_out[:, -1, :]  # Taking the output of the last time step
        # print(lstm_out.shape)
        # rnn_out, _ = self.rnn(emb_out)
        # rnn_out = rnn_out[:, -1, :]
        out = self.fc(lstm_out)  # Taking the output of the last time step
        return out
    
    def _initialize_weights(self):
        # Xavier initialization for embedding layer
        nn.init.xavier_uniform_(self.emb.weight)
        
        # Orthogonal initialization for LSTM layer
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:  # input-hidden weights
                nn.init.orthogonal_(param.data)
            elif 'weight_hh' in name:  # hidden-hidden weights
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:  # biases
                nn.init.constant_(param.data, 0)
        
        # Xavier initialization for fully connected layer
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
    # def __init__(self):
    #     super().__init__()
    #     self.emb = nn.Embedding(num_embeddings=50000, embedding_dim=64)
    #     self.layer1 = nn.Linear(256*64, 64*32)
    #     self.ac1 = nn.ReLU()
    #     self.layer2 = nn.Linear(64*32, 16*16)
    #     self.ac2 = nn.ReLU()
    #     self.out = nn.Linear(16*16, 2)

    # def forward(self, data):
    #     hidden = self.emb(data).view(-1, 64*256)
    #     return self.out(self.ac2(self.layer2(self.ac1(self.layer1(hidden)))))

model = MyLSTMModel().to(device)
model._initialize_weights()

# Loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=5e-3)
# optimizer = SGD(model.parameters(), lr=1e-2, momentum=0.9, nesterov=True)

# Initialize wandb
wandb.init(
    project="text_sentiment_classification",
    config={
        "learning_rate": 3e-3,
        "architecture": "LSTM",
        "dataset": "amazon-plarity",
        "epochs": 10,
    },
)
    
model.train()

# Training loop
def train():
    for epoch in range(10):
        print(f"--------------Epoch {epoch+1}----------------")
        model.train()
        for batch, (X, y) in enumerate(tqdm(train_loader)):
            pred = model(X.to(device))
            loss = loss_fn(pred, y.to(device))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 50 == 0:
                total_correct = 0
                with torch.no_grad():
                    for batch, (X, y_t) in enumerate(test_loader):
                        pred_t = model(X.to(device))
                        total_correct += np.sum((torch.argmax(pred_t.cpu(), 1) == torch.argmax(y_t.cpu(), 1)).numpy())
                test_accuracy = total_correct / len(test_set)
                train_accuracy = np.mean((torch.argmax(pred.cpu(), 1) == torch.argmax(y.cpu(), 1)).numpy())
                wandb.log(
                    {
                        "loss": loss.item(),
                        "train_acc": train_accuracy,
                        "test_acc": test_accuracy
                    }
                )
                # print(f"Train Acc: {train_accuracy:.4f}, Test Acc: {test_accuracy:.4f}")

        torch.save(model.state_dict(), f"result/model_epoch_{epoch+1}.pt")
    wandb.finish()

train()
# Inference after loading the saved model
model.load_state_dict(torch.load("result/model_epoch_10.pt"))

def predict(text):
    encoded = tokenizer.encode(text).ids[:256]
    encoded = torch.tensor(encoded).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = model(encoded)
        prediction = torch.argmax(output, 1).item()
    return "Positive" if prediction == 1 else "Negative"

# Example usage:
print(predict("This movie was absolutely fantastic!"))
