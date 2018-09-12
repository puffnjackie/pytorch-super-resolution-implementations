# Pytorch super-resolution implementations

A collections of Super-resolution models

## Requirements
- python 3.6
- numpy
- pytorch >= 0.4.0


## Models

- [RDN] https://arxiv.org/abs/1802.08797


# Usage
- training
  
```bash

python3 train.py --upscale_factor 2 --datapath /model/path/to/folder

```

- super-resolve

```bash
python3 inference.py --input_image /path/to/img --output_filename /path/to/img --model /path/to/pretrained/model
```

# Reference 
- https://github.com/pytorch/examples
- https://github.com/andreasveit/densenet-pytorch
- https://github.com/xternalz/WideResNet-pytorch