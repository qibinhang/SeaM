import argparse
from deepsparse import compile_model
from deepsparse.utils import generate_random_inputs
import torch.onnx
import sys
sys.path.append('../..')
sys.path.append('..')
import numpy as np
from sparseml.pytorch.utils import ModuleExporter

from multi_class.utils.reengineered_model_loader import load_reengineered_model
from multi_class.datasets.dataset_loader import load_dataset
from multi_class.config import load_config
from multi_class.models.resnet20 import cifar100_resnet20 as resnet20
from multi_class.models.resnet50 import resnet50


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['resnet20', 'resnet50'], required=True)
    parser.add_argument('--dataset', type=str, choices=['cifar100', 'imagenet'], required=True)
    parser.add_argument('--superclass_type', type=str, choices=['predefined', 'random'], default='predefined')
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


def compile_sparse_model(sparse_model_path, batch_size):
    batch_size = batch_size

    # Generate random sample input
    inputs = generate_random_inputs(sparse_model_path, batch_size)

    # Compile and run
    engine = compile_model(sparse_model_path, batch_size)
    return engine


def main():
    # prepare reengineered_model saved path.
    save_dir = f'{config.project_data_save_dir}/{args.model}_{args.dataset}/{args.superclass_type}/tsc_{args.target_superclass_idx}'
    if args.superclass_type == 'predefined':
        save_path = f'{save_dir}/lr_head_mask_{args.lr_head}_{args.lr_mask}_alpha_{args.alpha}.pth'
    elif args.superclass_type == 'random':
        raise ValueError('Not support for now.')
    else:
        raise ValueError
    reengineered_model = save_path

    img_size = 32 if args.dataset == 'cifar100' else 224

    # convert model and reengineered_model from pytorch to onnx.
    model = eval(f'{args.model}')(pretrained=True)
    exporter = ModuleExporter(model, './')
    exporter.export_onnx(torch.randn(16, 3, img_size, img_size), name='init_model.onnx')

    reengineered_model = load_reengineered_model(model, reengineered_model)
    exporter = ModuleExporter(reengineered_model, './')
    exporter.export_onnx(torch.randn(16, 3, img_size, img_size), name='reengineered_model.onnx')

    # use deepsaprse to compile and run the onnx model and onnx reengineered_model.
    np.random.seed(0)
    inputs = [np.random.rand(16, 3, img_size, img_size).astype(np.float32)]

    model = compile_sparse_model('./init_model.onnx', batch_size=16)
    model_costs = []
    for i in range(2):
        result = model.benchmark(inputs, num_iterations=200, num_warmup_iterations=100)
        ms_per_batch = result.ms_per_batch
        model_costs.append(ms_per_batch)

    reengineered_model = compile_sparse_model('./reengineered_model.onnx', batch_size=16)
    reengineered_model_costs = []
    for i in range(2):
        result = reengineered_model.benchmark(inputs, num_iterations=200, num_warmup_iterations=100)
        ms_per_batch = result.ms_per_batch
        reengineered_model_costs.append(ms_per_batch)

    model_cost = sum(model_costs) / len(model_costs)
    reengineered_model_costs = sum(reengineered_model_costs) / len(reengineered_model_costs)
    print(f'Model  Time Cost: {model_cost:.2f}ms')
    print(f'Reengineered Model Time Cost: {reengineered_model_costs:.2f}ms')
    print(f'Reduction (original - reengineered) / model: {(model_cost - reengineered_model_costs) / model_cost:.2%}')


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

    main()
