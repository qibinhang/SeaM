import argparse
import torch
import sys
sys.path.append('..')
sys.path.append('../..')
from torch.utils.data import DataLoader
from tqdm import tqdm
from advertorch.attacks import LinfPGDAttack
from utils.dataset_loader import load_dataset
from defect_inherit.models.resnet import resnet18, resnet18_nofc, resnet50, resnet50_nofc
from defect_inherit.config import load_config

"""Copy from ICSE'22 ReMos. Thanks the authors!"""

def adversary_test(model, loader, adversary):
    model.eval()
    total = 0
    top1_clean = 0
    top1_adv = 0
    adv_success = 0
    adv_trial = 0
    for batch, label in tqdm(loader, ncols=80, desc='adv_test'):
        batch, label = batch.to('cuda'), label.to('cuda')

        total += batch.size(0)
        out_clean = model(batch)
        y = torch.zeros(batch.shape[0], model.fc.in_features).cuda()  # for ResNet
        y[:, 0] = 1000
        batch_adv = adversary.perturb(batch, y)

        out_adv = model(batch_adv)

        pred_clean = torch.argmax(out_clean, dim=1)
        pred_adv = torch.argmax(out_adv, dim=1)

        clean_correct = pred_clean.eq(label)
        adv_trial += int(clean_correct.sum().item())
        adv_success += int(pred_adv[clean_correct].eq(label[clean_correct]).sum().detach().item())
        top1_clean += int(pred_clean.eq(label).sum().detach().item())
        top1_adv += int(pred_adv.eq(label).sum().detach().item())

    return float(top1_clean) / total * 100, float(top1_adv) / total * 100, float(
        adv_trial - adv_success) / adv_trial * 100


def myloss(yhat, y):
    return -((yhat[:, 0] - y[:, 0]) ** 2 + 0.1 * ((yhat[:, 1:] - y[:, 1:]) ** 2).mean(1)).mean()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['resnet18', 'resnet50'], required=True)
    parser.add_argument('--dataset', type=str,
                        choices=['cub200', 'dog120', 'flower102', 'mit67', 'action40'], required=True)

    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--eval_method', type=str, choices=['seam', 'remos', 'standard', 'retrain'], required=True)

    # just for seam_finetune
    parser.add_argument('--lr_mask', type=float, default=0.0,
                        help='learning rate for optimizing the mask in step 2.')
    parser.add_argument('--alpha', type=float, default=0.0,
                        help='the weight for the weighted sum of two losses in re-engineering.')
    parser.add_argument('--prune_threshold', type=float, default=1.0)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    configs = load_config()
    num_workers = 8
    pin_memory = True

    eval_method = args.eval_method

    if eval_method == 'seam':
        model_path = f'{configs.seam_finetune_dir}/{args.model}_{args.dataset}_dropout_{args.dropout}/' \
                     f'lr_mask_{args.lr_mask}_alpha_{args.alpha}_thres_{args.prune_threshold}/step_3_seam_ft.pth'
    elif eval_method == 'standard':
        model_path = f'{configs.standard_finetune_dir}/{args.model}_{args.dataset}_dropout_{args.dropout}/model_ft.pth'
    elif eval_method == 'retrain':
        model_path = f'{configs.retrain_dir}/{args.model}_{args.dataset}_dropout_{args.dropout}/model_rt.pth'
    else:
        raise ValueError

    dataset_test = load_dataset(args.dataset, is_train=False)
    num_classes = dataset_test.num_classes

    test_loader = DataLoader(dataset_test, batch_size=64, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    model_ft = eval(args.model)(pretrained=False, num_classes=num_classes).to('cuda').eval()
    model_ft.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))

    model_pt = eval(f'{args.model}_nofc')(pretrained=True, num_classes=num_classes).to('cuda').eval()

    attacker = LinfPGDAttack(
        model_pt, loss_fn=myloss, eps=0.1,
        nb_iter=40, eps_iter=0.01,
        rand_init=True, clip_min=-2.2, clip_max=2.2,
        targeted=False)

    clean_top1, _, adv_sr = adversary_test(model_ft, test_loader, attacker)

    print('Clean Top-1: {:.2f} | Attack Success Rate: {:.2f}'.format(clean_top1, adv_sr))
