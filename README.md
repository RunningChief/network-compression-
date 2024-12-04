# VGG Model Training and Optimization Pipeline

## Requirements:

### Core Dependencies
- torch>=1.8.0
- torchvision>=0.9.0
- numpy>=1.19.2
- torchsummary>=1.5.1
- tensorboard>=2.4.0

### Model Optimization
- torch-tensorrt>=1.4.0
- modelopt>=0.3.0
- joblib>=1.0.1

### Development Tools
- pytest>=6.2.5
- pylint>=2.8.0
- black>=21.5b2

### CUDA Support   
- cudatoolkit>=11.3

## Pipeline Steps

1. **Train VGG-19 Model**
    ```bash
    python main.py --dataset cifar10 --arch vgg --depth 19 --save baseline_logs
    ```

2. **Train VGG-19 with Sparsity**
    ```bash
    python main.py -sr --s 0.0001 --dataset cifar10 --arch vgg --depth 19 --save sparsity_logs
    ```

3. **Prune the Model**
    ```bash
    python prune.py --dataset cifar10 --depth 19 --percent 0.7 --model sparsity_logs/model_best.pth.tar --save pruned_logs
    ```

4. **Fine-tune**
    ```bash
    python main.py --refine pruned_logs/pruned.pth.tar --dataset cifar10 --arch vgg --depth 19 --save finetuned_logs
    ```

5. **Quantization**
    - **FP16 Quantization**
        ```bash
        python quantize.py --dataset cifar10 --arch vgg --depth 19 --save quantized_logs
        ```
    - **INT8 Quantization**
        ```bash
        python quantize.py --dataset cifar10 --arch vgg --depth 19 --fp8 --save quantized_logs_fp8
        ```

6. **Train VGG-19 with Matrix Decomposition**
    ```bash
    python main.py --dataset cifar10 --arch vgg_matrix_decomposition --depth 19 --save matrix_decomposition_logs
    ```

7. **Knowledge Distillation for Pruned Model**
    ```bash
    python main.py --refine pruned_logs/pruned.pth.tar --dataset cifar10 --arch vgg --depth 19 --knowledge-distillation --teacher-ckpt baseline_logs/model_best.pth.tar --save knowledge_distillation_logs
    ```

8. **Knowledge Distillation for VGG16**
    ```bash
    python main.py --dataset cifar10 --arch vgg --depth 19 --knowledge-distillation --vgg16-as-student --teacher-ckpt baseline_logs/model_best.pth.tar --save knowledge_distillation_vgg16_logs
    ```

9. **Inference**
    ```bash
    python inference.py
    ```
