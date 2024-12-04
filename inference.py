import warnings
warnings.filterwarnings('ignore')
import torch
import torch_tensorrt
from PIL import Image
from torchvision import transforms
import joblib

def predict(path):
    image = Image.open(path)

    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    #model = torch.export.load("quantized_logs/trt_model_fp16.ep").module()
    model = joblib.load('quantized_logs_fp8/trt_model_fp8.pkl')
    transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # image = transform(image).cuda().half().unsqueeze(0)
    image = transform(image).cuda().unsqueeze(0)
    with torch_tensorrt.logging.errors():
        label = model(image).argmax()

    return classes[label]

result = predict("dog.jpg")
print('result:', result)