import argparse
import numpy as np
import re
import torch
import torch.nn as nn
import sys
sys.path.append('../..')
sys.path.append('..')
from numbers import Number
from fvcore.nn import FlopCountAnalysis, flop_count_table
from typing import Any, Callable, List, Optional, Union
from collections import OrderedDict
from tqdm import tqdm
from torch.utils.data import DataLoader

from multi_class.datasets.dataset_loader import load_dataset
from multi_class.config import load_config
from multi_class.models.resnet20 import cifar100_resnet20 as resnet20
from multi_class.models.resnet50 import resnet50



class CalculateFLOP:
    def __init__(self, model):
        self.model = model
        self.model_modules = dict(list(model.named_modules()))
        self.calculated_convs = []

    @torch.no_grad()
    def calculate_flop_fvcore(self, img_size, is_sparse, show_details=False):
        x = torch.randn(1, 3, img_size, img_size)
        fca = FlopCountAnalysis(self.model, x)
        if is_sparse:
            handlers = {'aten::_convolution': self.masked_conv_flop_jit,
                        'aten::addmm': self.masked_fc_flop_jit}
            fca.set_op_handle(**handlers)

        flops = fca.total()
        if show_details:
            print(flop_count_table(fca))

        return flops

    def masked_conv_flop_jit(self, inputs: List[Any], outputs: List[Any]) -> Number:
        x = inputs[0]
        x_shape = self.get_shape(x)
        batch_size = x_shape[0]
        assert batch_size == 1

        # For vgg16 and resnet20 model. other models may use different inputs[x].
        conv_name = re.match(r'.+scope: /(.+)', str(inputs[3]).strip()).groups(1)[0].strip()
        conv_name = conv_name.split('/')[-1]

        self.calculated_convs.append(conv_name)

        weight = self.model_modules[conv_name].weight.numpy()
        vector_length = np.count_nonzero(weight)

        out_shape = self.get_shape(outputs[0])
        output_H, output_W = out_shape[2], out_shape[3]

        flop_mults = vector_length * output_H * output_W

        # Note: Same as fvcore, we only calculate multiplication, without calculating addition.
        flop_adds = 0
        # flop_adds = (vector_length-1) * output_H * output_W
        flop = flop_mults + flop_adds
        return flop

    def masked_fc_flop_jit(self, inputs: List[Any], outputs: List[Any]) -> Number:
        """
        Count flops for fully connected layers.
        """
        # Count flop for nn.Linear
        # inputs is a list of length 3.
        input_shapes = [self.get_shape(v) for v in inputs[1:3]]
        # input_shapes[0]: [batch size, input feature dimension]
        # input_shapes[1]: [batch size, output feature dimension]
        assert len(input_shapes[0]) == 2, input_shapes[0]
        assert len(input_shapes[1]) == 2, input_shapes[1]
        batch_size, input_dim = input_shapes[0]
        assert batch_size == 1


        # For vgg16 and resnet20 model. other models may use different inputs[x].
        fc_name = re.match(r'.+scope: /(.+) #.*', str(inputs[3]).strip()).groups(1)[0].strip()
        fc_name = fc_name.split('/')[-1]

        weight =  self.model_modules[fc_name].weight.numpy()
        flop = np.count_nonzero(weight)

        return flop

    @staticmethod
    def get_shape(val: Any) -> Optional[List[int]]:
        if val.isCompleteTensor():
            return val.type().sizes()
        else:
            return None


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['resnet20', 'resnet50'], required=True)
    parser.add_argument('--dataset', type=str, choices=['cifar100', 'imagenet'], required=True)
    parser.add_argument('--superclass_type', type=str, choices=['predefined'], default='predefined')
    parser.add_argument('--target_superclass_idx', type=int, default=-1)
    parser.add_argument('--n_classes', type=int, default=-1,
                        help='When randomly construct superclasses, how many classes dose a reengineered model recognize.')

    parser.add_argument('--lr_head', type=float, default=0.1)
    parser.add_argument('--lr_mask', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=1,
                        help='the weight for the weighted sum of two losses in re-engineering.')

    parser.add_argument('--shots', default=-1, type=int, help='how many samples for each classes.')
    parser.add_argument('--seed', type=int, default=0,
                        help='the random seed for sampling ``num_superclasses'' classes as target classes.')

    args = parser.parse_args()
    return args


@torch.no_grad()
def eval_reengineered_model(reengineered_model, test_loader):
    reengineered_model.eval()
    n_corrects = 0
    n_samples = 0

    for batch_inputs, batch_labels in tqdm(test_loader, ncols=80, desc=f'Eval '):
        batch_inputs, batch_labels = batch_inputs.to('cpu'), batch_labels.to('cpu')
        batch_outputs = reengineered_model(batch_inputs)
        n_samples += batch_labels.shape[0]
        batch_preds = torch.argmax(batch_outputs, dim=1)
        n_corrects += torch.sum(batch_preds == batch_labels).item()

    acc = float(n_corrects) / n_samples
    return acc


def eval_pretrained_model():
    model = eval(args.model)(pretrained=True).to('cpu')
    dataset_test = load_dataset(dataset_name=args.dataset, is_train=False, shots=args.shots,
                                superclass_type=args.superclass_type, target_superclass_idx=args.target_superclass_idx,
                                n_classes=args.n_classes, seed=args.seed, reorganize=False)
    test_loader = DataLoader(dataset_test, batch_size=64, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    model.eval()
    n_corrects = 0
    n_samples = 0

    for batch_inputs, batch_labels in tqdm(test_loader, ncols=80, desc=f'Eval '):
        batch_inputs, batch_labels = batch_inputs.to('cpu'), batch_labels.to('cpu')
        batch_outputs = model(batch_inputs)
        n_samples += batch_labels.shape[0]
        batch_preds = torch.argmax(batch_outputs, dim=1)
        n_corrects += torch.sum(batch_preds == batch_labels).item()

    acc = float(n_corrects) / n_samples
    return acc


def main():
    dataset_test = load_dataset(dataset_name=args.dataset, is_train=False, shots=args.shots,
                                superclass_type=args.superclass_type, target_superclass_idx=args.target_superclass_idx,
                                n_classes=args.n_classes, seed=args.seed, reorganize=True)

    test_loader = DataLoader(dataset_test, batch_size=64, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    print(f'INFO: Superclass {args.target_superclass_idx} contains {len(dataset_test.classes)} classes: {dataset_test.classes}.\n')
    if len(dataset_test.classes) == 1:
        sys.exit(0)

    # prepare reengineered_model saved path.
    save_dir = f'{config.project_data_save_dir}/{args.model}_{args.dataset}/{args.superclass_type}/tsc_{args.target_superclass_idx}'
    if args.superclass_type == 'predefined':
        save_path = f'{save_dir}/lr_head_mask_{args.lr_head}_{args.lr_mask}_alpha_{args.alpha}.pth'
    elif args.superclass_type == 'random':
        # save_path = f'{save_dir}/n_classes_{args.n_classes}.pth'
        raise ValueError('Not support for now.')
    else:
        raise ValueError


    model = eval(f'{args.model}')(pretrained=True, is_reengineering=False).to('cpu')
    module_head = nn.Sequential(
        nn.ReLU(inplace=True),
        nn.Linear(100 if args.dataset == 'cifar100' else 1000, len(dataset_test.classes))
    ).to('cpu')
    setattr(model, 'module_head', module_head)

    masks = torch.load(save_path, map_location=torch.device('cpu'))

    # remove irrelevant weights using masks
    model_params = model.state_dict()
    masked_params = OrderedDict()
    check_flag = 0
    for name, weight in model_params.items():
        if f'{name}_mask' in masks:
            mask = masks[f'{name}_mask']
            bin_mask = (mask > 0).float()
            masked_weight = weight * bin_mask
            masked_params[name] = masked_weight
            check_flag += 1
        elif f'module_head' in name:
            module_head_param = masks[name]
            masked_params[name] = module_head_param
            check_flag += 1
        else:
            masked_params[name] = weight
    assert check_flag == len(masks)
    model.load_state_dict(masked_params)

    acc = eval_reengineered_model(model, test_loader)  # check the reengineered_model's acc
    print(f'Reengineered model ACC: {acc:.2%}')

    cal_flop = CalculateFLOP(model)
    img_size = 32 if args.dataset == 'cifar100' else 224
    total_flop_dense = cal_flop.calculate_flop_fvcore(img_size=img_size, is_sparse=False)
    total_flop_sparse = cal_flop.calculate_flop_fvcore(img_size=img_size, is_sparse=True)
    print(f'FLOPs  Dense: {total_flop_dense/1e6:.2f}M')
    print(f'FLOPs Sparse: {total_flop_sparse/1e6:.2f}M')
    print(f'FLOP% (Sparse / Dense): {total_flop_sparse / total_flop_dense:.2%}')


if __name__ == '__main__':
    args = get_args()

    if args.model == 'resnet20':
        assert args.dataset == 'cifar100'
    elif args.model == 'resnet50':
        assert args.dataset == 'imagenet'
    else:
        raise ValueError
    print(args)
    config = load_config()
    num_workers = 0
    pin_memory = False
    acc = eval_pretrained_model()  # check the model's acc
    print(f'Model ACC: {acc:.2%}')

    main()