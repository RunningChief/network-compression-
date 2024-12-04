import os
import torch
import argparse

def setup_args():
    """Setup and return command line arguments for quantization
    
    Key arguments:
        --dataset: Dataset to use (default: cifar10)
        --fp8: Enable 8-bit floating point quantization
        --arch: Model architecture to use (default: vgg)
        --depth: Depth of the network (default: 19)
    """
    parser = argparse.ArgumentParser(description='PyTorch VGG CIFAR quantization')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--test-batch-size', type=int, default=1)
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--save', default='./logs', type=str)
    parser.add_argument('--arch', default='vgg', type=str)
    parser.add_argument('--depth', default=19, type=int)
    parser.add_argument('--fp8', action='store_true', default=False)
    
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    if not os.path.exists(args.save):
        os.makedirs(args.save)
        
    return args

def load_model(args):
    """Load and prepare model for quantization
    
    Loads a pre-trained model that has been pruned and finetuned.
    The model configuration is loaded from pruned_logs/pruned.pth.tar
    and weights are loaded from finetuned_logs/model_best.pth.tar
    
    Args:
        args: Command line arguments containing model configuration
    
    Returns:
        model: Loaded PyTorch model ready for quantization
    """
    checkpoint = torch.load('pruned_logs/pruned.pth.tar')
    model = models.__dict__[args.arch](
        dataset=args.dataset,
        depth=args.depth,
        cfg=checkpoint['cfg']
    )
    
    model.load_state_dict(
        torch.load('finetuned_logs/model_best.pth.tar')['state_dict']
    )
    
    model.eval()
    if args.cuda:
        model.cuda()
        
    return model 

def setup_prune_args():
    """Setup and return command line arguments for network pruning
    
    Key arguments:
        --dataset: Dataset to use (default: cifar100)
        --percent: Percentage of channels to prune (default: 0.5)
        --model: Path to pre-trained model to prune
        --save: Path to save pruned model
    """
    parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR prune')
    parser.add_argument('--dataset', type=str, default='cifar100',
                        help='training dataset (default: cifar100)')
    parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                        help='input batch size for testing (default: 256)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--depth', type=int, default=19,
                        help='depth of the vgg')
    parser.add_argument('--percent', type=float, default=0.5,
                        help='scale sparse rate (default: 0.5)')
    parser.add_argument('--model', default='', type=str, metavar='PATH',
                        help='path to the model (default: none)')
    parser.add_argument('--save', default='', type=str, metavar='PATH',
                        help='path to save pruned model (default: none)')
    
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    if not os.path.exists(args.save):
        os.makedirs(args.save)
        
    return args

def setup_train_args():
    """Setup and return command line arguments for model training
    
    Key arguments:
        --dataset: Dataset to use (default: cifar100)
        --sr: Enable sparsity regularization
        --s: Sparsity regularization strength (default: 0.0001)
        --knowledge-distillation: Enable knowledge distillation
        --teacher-ckpt: Path to teacher model checkpoint for knowledge distillation
        --vgg16-as-student: Use VGG16 as student model
    """
    parser = argparse.ArgumentParser(description='PyTorch VGG Slimming CIFAR training')
    parser.add_argument('--dataset', type=str, default='cifar100')
    parser.add_argument('--sparsity-regularization', '-sr', dest='sr', action='store_true')
    parser.add_argument('--s', type=float, default=0.0001)
    parser.add_argument('--refine', default='', type=str)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--test-batch-size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=160)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', default=1e-4, type=float)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--log-interval', type=int, default=100)
    parser.add_argument('--save', default='./logs', type=str)
    parser.add_argument('--arch', default='vgg', type=str)
    parser.add_argument('--depth', default=19, type=int)
    parser.add_argument('--knowledge-distillation', action='store_true')
    parser.add_argument('--teacher-ckpt', default='baseline_logs/model_best.pth.tar', type=str)
    parser.add_argument('--vgg16-as-student', action='store_true', default=False)
    
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    if not os.path.exists(args.save):
        os.makedirs(args.save)
        
    return args