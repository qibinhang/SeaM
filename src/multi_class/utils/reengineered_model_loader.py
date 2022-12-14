import torch
import torch.nn as nn
from collections import OrderedDict



def load_reengineered_model(original_model, mask_info_path):
    masks = torch.load(mask_info_path, map_location=torch.device('cpu'))

    ones, total = 0, 0
    original_non_zeros = 0

    # remove irrelevant weights using masks
    model_params = original_model.state_dict()
    masked_params = OrderedDict()
    for name, weight in model_params.items():
        if f'{name}_mask' in masks:
            mask = masks[f'{name}_mask']
            bin_mask = (mask > 0).int()

            masked_weight = weight * bin_mask
            masked_params[name] = masked_weight
            # print(f'{1 - torch.sum(bin_mask) / bin_mask.numel():.2%}')

            ones += torch.sum(bin_mask)
            total += bin_mask.numel()
            original_non_zeros += torch.count_nonzero(weight)
        else:
            masked_params[name] = weight
    original_model.load_state_dict(masked_params)
    print(f'Model  weights: {total/1e6:.2f}M')
    print(f'Reengineered Model weights: {ones/1e6:.2f}M')
    print(f'Pruned   ratio: {1 - ones / total:.2%}')
    return original_model