# Model pruning script for VGG architectures
# Implements channel pruning based on BatchNorm scaling factors
# Process:
# 1. Calculate pruning threshold from BatchNorm weights
# 2. Generate binary masks for pruning
# 3. Create new compact model
# 4. Transfer remaining weights to new model

import warnings
warnings.filterwarnings('ignore')
import os
import torch
import torch.nn as nn
import numpy as np
from utils.data_loader import get_test_loader
from utils.model_utils import load_model, setup_prune_args
from models.vgg import vgg

class ModelPruner:
    """
    Pruning implementation class that:
    1. Analyzes model weights to determine pruning threshold
    2. Creates pruning masks for each layer
    3. Handles weight transfer to new compact model
    4. Manages model evaluation before and after pruning
    """
    def __init__(self, args):
        self.args = args
        self.model = load_model(args)
        self.test_loader = get_test_loader(args)
        
    def test_model(self, model):
        """Test model accuracy after pruning"""
        model.eval()
        correct = 0
        
        for data, target in self.test_loader:
            if self.args.cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        accuracy = correct / float(len(self.test_loader.dataset))
        print('\nTest set: Accuracy: {}/{} ({:.1f}%)\n'.format(
            correct, len(self.test_loader.dataset), 100. * accuracy))
        return accuracy

    def get_pruning_mask(self):
        """Calculate pruning threshold and generate masks"""
        total = sum(m.weight.data.shape[0] for m in self.model.modules() 
                   if isinstance(m, nn.BatchNorm2d))
        
        # Collect all BatchNorm weights
        bn_weights = torch.zeros(total)
        index = 0
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                size = m.weight.data.shape[0]
                bn_weights[index:(index+size)] = m.weight.data.abs().clone()
                index += size

        # Calculate pruning threshold
        y, _ = torch.sort(bn_weights)
        thre_index = int(total * self.args.percent)
        thre = y[thre_index]
        
        return thre

    def prune_model(self):
        """Perform model pruning"""
        thre = self.get_pruning_mask()
        pruned = 0
        cfg = []
        cfg_mask = []
        
        # Generate pruning configuration
        for k, m in enumerate(self.model.modules()):
            if isinstance(m, nn.BatchNorm2d):
                weight_copy = m.weight.data.abs().clone()
                mask = weight_copy.gt(thre).float().cuda()
                pruned += mask.shape[0] - torch.sum(mask)
                m.weight.data.mul_(mask)
                m.bias.data.mul_(mask)
                cfg.append(int(torch.sum(mask)))
                cfg_mask.append(mask.clone())
                print(f'layer index: {k:d} \t total channel: {mask.shape[0]:d} \t '
                      f'remaining channel: {int(torch.sum(mask)):d}')
            elif isinstance(m, nn.MaxPool2d):
                cfg.append('M')
        
        pruned_ratio = pruned/total
        print(f'Pruned ratio: {pruned_ratio:.3f}')
        
        return cfg, cfg_mask

    def transfer_weights(self, model_src, model_dst, cfg_mask):
        """Transfer weights from pruned model to new model"""
        layer_id_in_cfg = 0
        start_mask = torch.ones(3)
        end_mask = cfg_mask[layer_id_in_cfg]

        for [m0, m1] in zip(model_src.modules(), model_dst.modules()):
            if isinstance(m0, nn.BatchNorm2d):
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))
                    
                m1.weight.data = m0.weight.data[idx1.tolist()].clone()
                m1.bias.data = m0.bias.data[idx1.tolist()].clone()
                m1.running_mean = m0.running_mean[idx1.tolist()].clone()
                m1.running_var = m0.running_var[idx1.tolist()].clone()
                
                layer_id_in_cfg += 1
                start_mask = end_mask.clone()
                if layer_id_in_cfg < len(cfg_mask):
                    end_mask = cfg_mask[layer_id_in_cfg]
                    
            elif isinstance(m0, nn.Conv2d):
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                print(f'In shape: {idx0.size:d}, Out shape {idx1.size:d}.')
                
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))
                    
                w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
                w1 = w1[idx1.tolist(), :, :, :].clone()
                m1.weight.data = w1.clone()
                
            elif isinstance(m0, nn.Linear):
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                m1.weight.data = m0.weight.data[:, idx0].clone()
                m1.bias.data = m0.bias.data.clone()

    def run(self):
        """Main pruning process"""
        # Test original model
        acc = self.test_model(self.model)
        
        # Prune model
        cfg, cfg_mask = self.prune_model()
        
        # Create new model with pruned configuration
        newmodel = vgg(dataset=self.args.dataset, cfg=cfg)
        if self.args.cuda:
            newmodel.cuda()
            
        # Transfer weights
        self.transfer_weights(self.model, newmodel, cfg_mask)
        
        # Save pruned model info
        num_parameters = sum(param.nelement() for param in newmodel.parameters())
        save_path = os.path.join(self.args.save, "prune.txt")
        with open(save_path, "w") as fp:
            fp.write(f"Configuration:\n{str(cfg)}\n")
            fp.write(f"Number of parameters:\n{str(num_parameters)}\n")
            fp.write(f"Test accuracy:\n{str(acc)}")
            
        # Save model
        torch.save({
            'cfg': cfg,
            'state_dict': newmodel.state_dict()
        }, os.path.join(self.args.save, 'pruned.pth.tar'))
        
        # Test pruned model
        self.test_model(newmodel)

def main():
    args = setup_prune_args()
    pruner = ModelPruner(args)
    pruner.run()

if __name__ == "__main__":
    main()
