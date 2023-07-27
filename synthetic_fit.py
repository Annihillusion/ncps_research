import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from ncps.torch import LTC
from ncps.wirings import AutoNCP
from utils.DataLoader import NeuronDataset


def train_one_epoch(model, trainloader, criterion, optimizer):
    running_loss = 0.0
    total = len(trainloader.dataset)
    pbar = tqdm(total)
    model.train()
    device = next(model.parameters()).device  # get device the model is located on
    for i, (inputs, labels) in enumerate(trainloader):
        inputs = inputs.to(device)  # move data to same device as the model
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs, hx = model(inputs)
        labels = labels.view(-1, *labels.shape[2:])  # flatten
        outputs = outputs.reshape(-1, *outputs.shape[2:])  # flatten
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        pbar.set_description(f"loss={running_loss / ((i+1)*trainloader.batch_size):0.4g}")
        pbar.update(1)
    pbar.close()
    return running_loss / total

def eval(model, valloader, criterion):
    losses = []
    model.eval()
    device = next(model.parameters()).device  # get device the model is located on
    with torch.no_grad():
        for inputs, labels in valloader:
            inputs = inputs.to(device)  # move data to same device as the model
            labels = labels.to(device)

            outputs, _ = model(inputs)
            outputs = outputs.reshape(-1, *outputs.shape[2:])  # flatten
            labels = labels.view(-1, *labels.shape[2:])  # flatten
            loss = criterion(outputs, labels)
            losses.append(loss.item())
    return np.mean(losses)


if __name__ == '__main__':
    EPOCHS = 10
    LR = 1e-3
    BATCH_SIZE = 4
    NUM_WORKERS = 4
    DATA_DIR = "E:/Celegans-ForwardCrawling-RNNs/Dataset1"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LTC(4, AutoNCP(8, 4), batch_first=True).to(device)

    dataset = NeuronDataset(root_dir=DATA_DIR)
    train_num = int(len(dataset) * 0.8)
    train_set, val_set = random_split(dataset, [train_num, len(dataset)-train_num])
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)

    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)

        # Evaluate
        val_loss = eval(model, val_loader, criterion)
        print(f"Epoch {epoch+1}, train_loss={train_loss:0.4g}, val_loss={val_loss:0.4g}")

        # if np.mean(returns) > max_return:
        #     max_return = np.mean(returns)
        #     torch.save(model.state_dict(), MODEL_NAME)
