import copy
import torch
import torch.nn.functional as F
from tqdm import tqdm


def finetune(model, optim, train_loader, test_loader, n_epochs, early_stop=-1):
    best_epoch = 0
    best_acc = 0.0
    best_model = None
    early_stop_epochs = 0

    for epoch in range(n_epochs):
        print(f'\nEpoch {epoch}')
        print('-' * 80)
        model.train()
        predicts, labels = [], []
        for batch_input, batch_labels in tqdm(train_loader, ncols=80, desc='Train'):
            batch_input, batch_labels = batch_input.to('cuda'), batch_labels.to('cuda')
            optim.zero_grad()

            batch_output = model(batch_input)
            loss = F.cross_entropy(batch_output, batch_labels)
            loss.backward()
            optim.step()

            predicts.append(torch.argmax(batch_output, dim=1).detach())
            labels.append(batch_labels.detach())

        predicts = torch.cat(predicts, dim=0)
        labels = torch.cat(labels, dim=0)
        acc = torch.sum(predicts == labels) / labels.shape[0]
        print(f'Train Acc: {acc:.2%}')

        model.eval()
        with torch.no_grad():
            predicts, labels = [], []
            for batch_input, batch_labels in tqdm(test_loader, ncols=80, desc='Val  '):
                batch_input, batch_labels = batch_input.to('cuda'), batch_labels.to('cuda')
                batch_output = model(batch_input)
                predicts.append(torch.argmax(batch_output, dim=1).detach())
                labels.append(batch_labels.detach())

            predicts = torch.cat(predicts, dim=0)
            labels = torch.cat(labels, dim=0)
            acc = torch.sum(predicts == labels) / labels.shape[0]
            print(f'Val   Acc: {acc:.2%}')

        if acc >= best_acc:
            best_acc = acc
            best_epoch = epoch
            best_model = copy.deepcopy(model)
            early_stop_epochs = 0
        else:
            early_stop_epochs += 1
            if early_stop_epochs == early_stop:
                print(f'Early Stop.\n\n')
                break
    return best_model, best_acc, best_epoch
