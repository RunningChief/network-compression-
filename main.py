# Main training script for VGG model variants
# Supports:
# - Regular VGG-19 training
# - Sparsity-regularized training
# - Model fine-tuning after pruning
# - Matrix decomposition training
# - Knowledge distillation training

import warnings
warnings.filterwarnings('ignore')
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchsummary import summary

from utils.data_loader import get_train_loader, get_test_loader
from utils.model_utils import setup_train_args, load_checkpoint, save_checkpoint
from models.vgg import vgg

class ModelTrainer:
    """
    Main trainer class that handles:
    1. Regular VGG training
    2. Sparsity-regularized training with BatchNorm updates
    3. Fine-tuning of pruned models
    4. Knowledge distillation with teacher-student training
    5. Matrix decomposition model training
    """
    def __init__(self, args):
        self.args = args
        self.train_loader = get_train_loader(args)
        self.test_loader = get_test_loader(args)
        self.model = self._initialize_model()
        self.teacher_model = self._initialize_teacher_model() if args.knowledge_distillation else None
        self.optimizer = self._initialize_optimizer()
        self.best_prec1 = 0.

    def _initialize_model(self):
        """Initialize the main model based on arguments"""
        if self.args.refine:
            checkpoint = load_checkpoint(self.args.refine)
            model = vgg(dataset=self.args.dataset, depth=self.args.depth, cfg=checkpoint['cfg'])
            model.load_state_dict(checkpoint['state_dict'])
        elif self.args.knowledge_distillation and self.args.vgg16_as_student:
            model = vgg(dataset=self.args.dataset, depth=16)
        else:
            model = vgg(dataset=self.args.dataset, depth=self.args.depth)

        if self.args.cuda:
            print("****** summary of model ******")
            summary(model, (3, 32, 32))
            model.cuda()
        return model

    def _initialize_teacher_model(self):
        """Initialize teacher model for knowledge distillation"""
        teacher_model = vgg(dataset=self.args.dataset, depth=self.args.depth)
        checkpoint = load_checkpoint(self.args.teacher_ckpt)
        teacher_model.load_state_dict(checkpoint['state_dict'])
        
        if self.args.cuda:
            print("****** summary of teacher model ******")
            summary(teacher_model, (3, 32, 32))
            teacher_model.cuda()
        return teacher_model

    def _initialize_optimizer(self):
        """Initialize the optimizer"""
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.args.lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay
        )
        
        if self.args.resume:
            checkpoint = load_checkpoint(self.args.resume)
            self.args.start_epoch = checkpoint['epoch']
            self.best_prec1 = checkpoint['best_prec1']
            self.model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            
        return optimizer

    def update_learning_rate(self, epoch):
        """Update learning rate at specific epochs"""
        if epoch in [self.args.epochs*0.5, self.args.epochs*0.75]:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= 0.1

    def update_bn(self):
        """Update BatchNorm layers for sparsity regularization"""
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.grad.data.add_(self.args.s * torch.sign(m.weight.data))

    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        if self.args.knowledge_distillation:
            self.teacher_model.eval()

        for batch_idx, (data, target) in enumerate(self.train_loader):
            if self.args.cuda:
                data, target = data.cuda(), target.cuda()
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self._calculate_loss(output, target, data)
            
            loss.backward()
            if self.args.sr:
                self.update_bn()
            self.optimizer.step()

            if batch_idx % self.args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader), loss.item()))

    def _calculate_loss(self, output, target, data):
        """Calculate loss including knowledge distillation if enabled"""
        base_loss = F.cross_entropy(output, target)
        
        if self.args.knowledge_distillation:
            T = 10
            alpha = 0.1
            with torch.no_grad():
                teacher_output = self.teacher_model(data)
                kd_loss = F.kl_div(
                    F.log_softmax(output / T, dim=1),
                    F.softmax(teacher_output / T, dim=1),
                    reduction='batchmean'
                )
            return (1 - alpha) * base_loss + alpha * T * T * kd_loss
        return base_loss

    def test(self):
        """Evaluate model on test set"""
        self.model.eval()
        test_loss = 0
        correct = 0

        for data, target in self.test_loader:
            if self.args.cuda:
                data, target = data.cuda(), target.cuda()
            
            output = self.model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(self.test_loader.dataset)
        accuracy = correct / float(len(self.test_loader.dataset))
        
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
            test_loss, correct, len(self.test_loader.dataset),
            100. * accuracy))
        return accuracy

    def run(self):
        """Main training loop"""
        for epoch in range(self.args.start_epoch, self.args.epochs):
            self.update_learning_rate(epoch)
            self.train_epoch(epoch)
            prec1 = self.test()
            
            is_best = prec1 > self.best_prec1
            self.best_prec1 = max(prec1, self.best_prec1)
            
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'best_prec1': self.best_prec1,
                'optimizer': self.optimizer.state_dict()
            }, is_best, self.args.save)

def main():
    args = setup_train_args()
    trainer = ModelTrainer(args)
    trainer.run()

if __name__ == "__main__":
    main()