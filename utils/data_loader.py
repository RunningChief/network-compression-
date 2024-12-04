import torch
from torchvision import datasets, transforms

def get_test_loader(args):
    """Create test data loader"""
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    
    normalize = transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010)
    )
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    
    dataset = datasets.CIFAR10(
        './data.cifar10',
        train=False,
        transform=transform
    )
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=args.test_batch_size,
        shuffle=True,
        **kwargs
    ) 