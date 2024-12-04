import os
import torch
import torch_tensorrt
import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.utils import export_torch_mode
import joblib

from utils.data_loader import get_test_loader
from utils.model_utils import load_model, setup_args
from models.vgg import vgg

class ModelQuantizer:
    def __init__(self, args):
        self.args = args
        self.test_loader = get_test_loader(args)
        self.model = load_model(args)
        
    def test_model(self, model, fp8=False):
        """Test quantized model accuracy"""
        correct = 0
        total = 0
        
        for data, target in self.test_loader:
            if self.args.cuda:
                data = data.cuda() if fp8 else data.cuda().half()
                target = target.cuda()
            
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).cpu().sum()
            total += target.size(0)
            
        accuracy = correct / float(total)
        print(f"{'FP8' if fp8 else 'FP16'} Quantized accuracy: {accuracy}")
        return accuracy

    def quantize_fp16(self):
        """Perform FP16 quantization"""
        inputs = [torch.randn((1, 3, 32, 32)).cuda().half()]
        trt_model = torch_tensorrt.compile(
            self.model.half(), 
            inputs=inputs,
            enabled_precisions={torch.half},
            workspace_size=1 << 22
        )
        
        # Save model
        save_path = os.path.join(self.args.save, "trt_model_fp16.ep")
        torch_tensorrt.save(trt_model, save_path, inputs=inputs)
        return trt_model

    def quantize_fp8(self):
        """Perform FP8 quantization"""
        def calibrate_loop(model):
            for data, _ in self.test_loader:
                data = data.cuda()
                _ = model(data)

        # Apply FP8 quantization
        quant_cfg = mtq.FP8_DEFAULT_CFG
        mtq.quantize(self.model, quant_cfg, forward_loop=calibrate_loop)

        # Export and compile model
        input_tensor = torch.randn((1, 3, 32, 32)).cuda()
        with torch.no_grad(), export_torch_mode():
            exp_program = torch.export.export(self.model, (input_tensor,))
            trt_model = torch_tensorrt.dynamo.compile(
                exp_program,
                inputs=[input_tensor],
                enabled_precisions={torch.float8_e4m3fn},
                min_block_size=1,
                debug=False
            )
            
            # Save model
            save_path = os.path.join(self.args.save, "trt_model_fp8.pkl")
            joblib.dump(trt_model, save_path)
            return trt_model

    def run(self):
        """Main quantization process"""
        if not self.args.fp8:
            model = self.quantize_fp16()
            self.test_model(model, fp8=False)
        else:
            model = self.quantize_fp8()
            self.test_model(model, fp8=True)

def main():
    args = setup_args()
    quantizer = ModelQuantizer(args)
    quantizer.run()

if __name__ == "__main__":
    main()
