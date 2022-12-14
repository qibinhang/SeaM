# Reusing Deep Neural Network Models through Model Re-engineering
## Abstract
Training deep neural network (DNN) models, which has become an important task in today's software development, is often costly in terms of computational resources and time. With the inspiration of software reuse, building DNN models through reusing existing ones has gained increasing attention recently.  Prior approaches to DNN model reuse have two main limitations: 1) reusing the entire model, while only a small part of the model's functionalities (labels) are required, would cause much overhead (e.g., computational and time costs for inference), and 2) model reuse would inherit the defects and weaknesses of the reused model, and hence put the new system under threats of security attack. To solve the above problem, we propose SeaM, a tool that re-engineers a trained DNN model to improve its reusability. Specifically, given a target problem and a trained model, SeaM utilizes a gradient-based search method to search for the model's weights that are relevant to the target problem. The re-engineered model that only retains the relevant weights is then reused to solve the target problem. Evaluation results on widely-used models show that the re-engineered models produced by SeaM only contain 10.11% weights of the original models, resulting 42.41% reduction in terms of inference time. For the target problem, the re-engineered models even outperform the original models in classification accuracy by 5.85%. Moreover, reusing the re-engineered models inherits an average of 57% fewer defects than reusing the entire model. We believe our approach to reducing reuse overhead and defect inheritance is one important step forward for practical model reuse.


## Requirements
+ advertorch 0.2.3<br>
+ fvcore 0.1.5.post20220512<br>
+ matplotlib 3.4.2<br>
+ numpy 1.19.2<br>
+ python 3.8.10<br>
+ pytorch 1.8.1<br>
+ torchvision 0.9.0<br>
+ tqdm 4.61.0<br>
+ GPU with CUDA support is also needed

<br>

## Structure of the directories

```powershell
  |--- README.md                        :  user guidance
  |--- data/                            :  experimental data
  |--- src/                             :  source code of our work
  |------ global_config.py              :  setting the path
  |------ binary_class/                 :  direct reuse on binary classification problems
  |--------- model_reengineering.py     :  re-engineering a trained model and then reuse the re-engineered model
  |--------- calculate_flop.py          :  calculating the number of FLOPs required by reusing the re-engineered and original models
  |--------- calculate_time_cost.py     :  calculating the inference time required by reusing the re-engineered and original models
  |--------- ......                 
  |------ multi_class/                  :  direct reuse on multi-class classification problems
  |--------- ...... 
  |------ defect_inherit/               :  indirect reuse 
  |--------- reengineering_finetune.py  :  re-engineering a trained model and then fine-tuning the re-engineered model
  |--------- standard_finetune.py       :  using standard fine-tuning approach to fine-tune a trained model
  |--------- eval_robustness.py         :  calculating the defect inheritance rate
  |--------- ......
```


<br>

The following sections describe how to reproduce the experimental results in our paper. 

## Downloading experimental data
1. We provide the trained models and datasets used in the experiments, as well as the corresponding re-engineered models.<br>
One can download `data/` from [here](https://mega.nz/file/tX91ACpR#CSbQ2Xariha7_HLavE_6pKg4FoO5axOPemlv5J0JYwY) and then move it to `SeaM/`.<br>
The trained models will be downloaded automatically by PyTorch when running our project. If the download fails, please move our provided trained models to the folder according to the failure information given by PyTorch.<br>
Due to the huge size of ImageNet, please download it from its [webpage](https://www.image-net.org/).
2. Modify `self.root_dir` in `src/global_config.py`.

## Direct model reuse  
### Re-engineering on binary classification problems
1. Go to the directory of experiments related to the binary classification problems.
```commandline
cd src/binary_class
```
2. Re-engineer VGG16-CIFAR10 on a binary classification problem.
```commandline
python model_reengineering.py --model vgg16 --dataset cifar10 --target_class 0 --lr_mask 0.01 --alpha 1
```
3. Compute the number of FLOPs required by the original and re-engineered VGG16-CIFAR10, respectively. This command also presents the accuracy of models.
```commandline
python calculate_flop.py --model vgg16 --dataset cifar10 --target_class 0 --lr_mask 0.01 --alpha 1
```
4. Compute the time cost for inference required by the original and re-engineered VGG16-CIFAR10, respectively. This command also presents the number of a model's weights.
```commandline
python calculate_time_cost.py --model vgg16 --dataset cifar10 --target_class 0 --lr_mask 0.01 --alpha 1
```

### Re-engineering on multi-class classification problems
1. Go to the directory of experiments related to the multi-class classification problems.
```commandline
cd src/multi_class
```
2. Re-engineer ResNet20-CIFAR100 on a multi-class classification problem.
```commandline
python model_reengineering.py --model resnet20 --dataset cifar100 --target_superclass_idx 0 --lr_mask 0.1 --alpha 2
```
3. Compute the number of FLOPs required by the original and re-engineered ResNet20-CIFAR100, respectively. This command also presents the accuracy of models. 
```commandline
python calculate_flop.py --model resnet20 --dataset cifar100 --target_superclass_idx 0 --lr_mask 0.1 --alpha 2
```
4. Compute the time cost for inference required by the original and re-engineered ResNet20-CIFAR100, respectively. This command also presents the number of a model's weights. 
```commandline
python calculate_time_cost.py --model resnet20 --dataset cifar100 --target_superclass 0 --lr_mask 0.1 --alpha 2
```

***NOTE***: When computing the time cost for inference, DeepSparse runs a model on several CPUs.
The inference process would be interfered with other active processes, leading to fluctuations in inference time cost.
In our experiments, we manually kill as many other processes as possible and enable the inference process to occupy the CPUs exclusively.

## Indirect model reuse
1. Go to the directory of experiments related to the defect inheritance. 
```commandline
cd src/defect_inherit
```
2. Re-engineer ResNet18-ImageNet and then fine-tune the re-engineered model on the target dataset Scenes.
```commandline
python reengineering_finetune.py --model resnet18 --dataset mit67 --lr_mask 0.05 --alpha 0.5 --prune_threshold 0.6
```
3. Compute the defect inheritance rate of fine-tuned re-engineered ResNet18-Scenes. 
```commandline
python eval_robustness.py --model resnet18 --dataset mit67 --eval seam --lr_mask 0.05 --alpha 0.5 --prune_threshold 0.6
```
4. Fine-tune the original ResNet18-ImageNet on the target dataset Scenes. 
```commandline
python standard_finetune.py --model resnet18 --dataset mit67
```
5. Compute the defect inheritance rate of fine-tuned original ResNet18-Scenes. 
```commandline
python eval_robustness.py --model resnet18 --dataset mit67 --eval standard
```


## Supplementary experimental results
### The values of major parameters
The following table shows the default hyperparameters. The details of settings for re-engineering each trained model on each target problem can be obtained according to the experimental result files. <br>
For instance, the values of *learning rate* and *alpha* for the re-engineered model file `SeaM/data/multi_class_classification/resnet20_cifar100/predefined/tsc_0/lr_head_mask_0.1_0.05_alpha_1.0.pth` are 0.05 and 1.0, respectively.
<table>
<thead>
  <tr>
    <th>Target Problem</th>
    <th>Model Name</th>
    <th>Learning Rate</th>
    <th>Alpha</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="4">Binary<br>Classification</td>
    <td>VGG16-CIFAR10</td>
    <td>0.01</td>
    <td>1.00</td>
  </tr>
  <tr>
    <td>VGG16-CIFAR100</td>
    <td>0.05</td>
    <td>1.50</td>
  </tr>
  <tr>
    <td>ResNet20-CIFAR10</td>
    <td>0.05</td>
    <td>1.00</td>
  </tr>
  <tr>
    <td>ResNet20-CIFAR100</td>
    <td>0.12</td>
    <td>1.50</td>
  </tr>
  <tr>
    <td rowspan="2">Multi-class<br>Classification</td>
    <td>ResNet20-CIFAR100</td>
    <td>0.10</td>
    <td>2.00</td>
  </tr>
  <tr>
    <td>ResNet50-ImageNet</td>
    <td>0.05</td>
    <td>2.00</td>
  </tr>
</tbody>
</table>

<br>


### The impact of reducing the number of weights on ACC and DIR. (for RQ3)
We investigate the impact of the reduction in the number of weights on the ACC and DIR.
A threshold $t$ is used to early stop model re-engineering when the rate of removed weights reaches the threshold.
The following figure shows the ACC and DIR of the fine-tuned ResNet18 with different values of $t$, where $t=0.3, 0.4, 0.5, 0.6, 0.7$.
We find that, as the number of weights reduces, the DIR reduces significantly, while the ACC is stable overall. 
The results demonstrate the effectiveness of model re-engineering in reducing the DIR and the robustness of SeaM.

![img](https://github.com/qibinhang/SeaM/blob/main/src/defect_inherit/prune_ratio.png)
