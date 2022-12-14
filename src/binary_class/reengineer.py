import copy
import torch
import torch.nn.functional as F
from tqdm import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Reengineer:
    def __init__(self, pretrained_model, train_dataset, val_dataset, acc_pre_model):
        self.pt_model = pretrained_model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.acc_pre_model = acc_pre_model
        assert 0 < self.acc_pre_model <= 1

    def alter(self, lr_mask, lr_head, n_epochs, alpha, early_stop=-1):
        reengineered_model = copy.deepcopy(self.pt_model)

        mask_params = []
        for name, p in reengineered_model.named_parameters():
            if 'weight_mask' in name or 'bias_mask' in name:
                p.requires_grad = True
                mask_params.append(p)
            else:
                p.requires_grad = False

        module_head = reengineered_model.module_head
        for p in module_head.parameters():
            p.requires_grad = True

        # just optimize reengineered_model head
        optim_h = torch.optim.Adam(
            module_head.parameters(),
            lr_head
        )

        # jointly optimize masks and reengineered_model head
        optim_hm = torch.optim.Adam(
            [{'params': mask_params, 'lr': lr_mask},
             {'params': module_head.parameters(), 'lr': lr_head}]
        )

        strategy = [0, 1, 1, 1, 1] * (int(n_epochs / 5) + 1)

        best_acc, best_epoch = 0, 0
        best_loss_pred, best_loss_weight_ratio = 0.0, 1.0
        best_acc_with_ratio = -10000
        best_reengineered_model = None
        early_stop_epochs = 0
        enable_early_stop = False

        for epoch in range(n_epochs):
            if strategy[epoch] == 0:
                print(f'\nEpoch {epoch}: Just   Head')
                print('-' * 80)
                optim = optim_h
            else:
                print(f'\nEpoch {epoch}: Head & Mask')
                print('-' * 80)
                optim = optim_hm

            reengineered_model = self._train(reengineered_model, optim, alpha)
            acc, loss, loss_pred, loss_weight_ratio = self._test(reengineered_model, alpha)

            acc_with_ratio = acc - loss_weight_ratio  # old early stopping strategy. used to another way for now.
            if acc_with_ratio > best_acc_with_ratio:
                best_acc_with_ratio = acc_with_ratio
                bak_loss_weight_ratio = loss_weight_ratio
                bak_acc = acc
                bak_epoch = epoch
                bak_reengineered_model = copy.deepcopy(reengineered_model)

            # new early stopping strategy. NOTE: the early_stop should not too small.
            if acc >= self.acc_pre_model:
                enable_early_stop = True

            if acc >= self.acc_pre_model:
                if int(loss_weight_ratio * 1000) < int(best_loss_weight_ratio * 1000):
                    early_stop_epochs = 0
                elif loss_weight_ratio > best_loss_weight_ratio:
                    continue
                elif acc <= best_acc:
                    continue
                best_loss_weight_ratio = loss_weight_ratio
                best_acc = acc
                best_epoch = epoch
                best_reengineered_model = copy.deepcopy(reengineered_model)
            elif enable_early_stop:
                early_stop_epochs += 1
                if early_stop_epochs == early_stop:
                    print(f'Early Stop.\n\n')
                    break

        if best_reengineered_model is None:
            best_reengineered_model = bak_reengineered_model
            best_acc = bak_acc
            best_loss_weight_ratio = bak_loss_weight_ratio
            best_epoch = bak_epoch

        print(f"\n{'Epoch':^6}  {'Acc':^8}  {'Weight_ratio':^12}")
        print(f'{best_epoch:^6}  {best_acc:^8.2%}  {best_loss_weight_ratio:^12.2%}')
        return best_reengineered_model

    def fix_bn(self, m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.eval()

    def _train(self, reengineered_model, optim, alpha):
        reengineered_model.train()
        reengineered_model.apply(self.fix_bn)  # IMPORTANT!
        loss_pred, loss_weight_ratio, loss = [], [], []
        n_corrects = 0
        n_samples = 0

        for batch_inputs, batch_labels in tqdm(self.train_dataset, ncols=80, desc=f'Train'):
            batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
            optim.zero_grad()

            batch_outputs = reengineered_model(batch_inputs)
            batch_pred_loss = F.cross_entropy(batch_outputs, batch_labels)
            batch_weight_ratio_loss = reengineered_model.count_weight_ratio()
            # batch_loss = (1 - alpha) * batch_pred_loss + alpha * batch_weight_ratio_loss
            batch_loss = batch_pred_loss + alpha * batch_weight_ratio_loss
            batch_loss.backward()
            optim.step()

            # batch log
            n_samples += batch_labels.shape[0]
            batch_preds = torch.argmax(batch_outputs, dim=1)
            n_corrects += torch.sum(batch_preds == batch_labels).item()
            loss_pred.append(batch_pred_loss.item())
            loss_weight_ratio.append(batch_weight_ratio_loss.item())
            loss.append(batch_loss.item())


        # epoch log
        acc = float(n_corrects) / n_samples
        loss_pred = sum(loss_pred) / len(loss_pred)
        loss_weight_ratio = sum(loss_weight_ratio) / len(loss_weight_ratio)
        loss = sum(loss) / len(loss)
        print(f"{'Phase':<8}  {'Acc':<8}  ||  {'Loss':<5} ({'pred':<5} | {'weight':<6})")
        print(f"{'Train':<8}  {acc:<8.2%}  ||  {loss:<5.2f} ({loss_pred:<5.2f} | {loss_weight_ratio:<6.2%})")
        return reengineered_model

    @torch.no_grad()
    def _test(self, reengineered_model, alpha):
        reengineered_model.eval()
        loss_pred, loss_weight_ratio, loss = 0.0, 0.0, 0.0
        n_corrects = 0
        n_samples = 0

        # forward
        for batch_inputs, batch_labels in tqdm(self.val_dataset, ncols=80, desc=f'Val  '):
            batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
            batch_outputs = reengineered_model(batch_inputs)
            batch_pred_loss = F.cross_entropy(batch_outputs, batch_labels)
            batch_weight_ratio_loss = reengineered_model.count_weight_ratio()
            # batch_loss = (1 - alpha) * batch_pred_loss + alpha * batch_weight_ratio_loss
            batch_loss = batch_pred_loss + alpha * batch_weight_ratio_loss

            # batch log
            n_samples += batch_labels.shape[0]
            batch_preds = torch.argmax(batch_outputs, dim=1)
            n_corrects += torch.sum(batch_preds == batch_labels).item()
            loss_pred += batch_pred_loss.item() * batch_inputs.shape[0]
            loss_weight_ratio += batch_weight_ratio_loss.item() * batch_inputs.shape[0]
            loss += batch_loss.item() * batch_inputs.shape[0]

        # epoch log
        acc = float(n_corrects) / n_samples
        loss_pred = loss_pred / n_samples
        loss_weight_ratio = loss_weight_ratio / n_samples
        loss = loss / n_samples
        print(f"{'Val':<8}  {acc:<8.2%}  ||  {loss:<5.2f} ({loss_pred:<5.2f} | {loss_weight_ratio:<6.2%})")
        return acc, loss, loss_pred, loss_weight_ratio
