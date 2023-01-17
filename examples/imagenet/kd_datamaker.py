import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
import os
from PIL import Image
from efficientnet_pytorch import EfficientNet
import torch.nn as nn

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))
class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)
class teacher_Headmodel(nn.Module):
    def __init__(self,student_efficientnet=EfficientNet.from_pretrained('efficientnet-b7')):
        super().__init__()
        self._conv_stem = student_efficientnet._conv_stem
        self._conv_stem.requires_grad = False
        self._bn0 = student_efficientnet._bn0
        self._swish = MemoryEfficientSwish()
        self._blocks = student_efficientnet._blocks[:11]   
    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export).

        Args:
            memory_efficient (bool): Whether to use memory-efficient version of swish.
        """
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
        for block in self._blocks:
            block.set_swish(memory_efficient)
            
    def extract_features(self, inputs):
        """use convolution layer to extract feature .

        Args:
            inputs (tensor): Input tensor.

        Returns:
            Output of the final convolution
            layer in the efficientnet model.
        """
        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = 0.2
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)

        # print(x.shape) #torch.Size([1, 48, 150, 150]
        return x

t = teacher_Headmodel()
t.eval()
image_size = 600
tfms = transforms.Compose([transforms.Resize(image_size), transforms.CenterCrop(image_size), 
                           transforms.ToTensor(),
                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])


ori_dir = '/home/yenhsiu/dataset/ilsvrc2012'
dst_dir = ori_dir + "_KD"
if os.path.exists(dst_dir):
    print("target file exists")
else:
    os.mkdir(dst_dir)
    ori_dir_list = os.listdir(ori_dir)
    for f in ori_dir_list:
        if os.path.isdir(ori_dir+"/"+f):
            os.mkdir(dst_dir+"/"+f)
            ori_dir_label_list = os.listdir(ori_dir+"/"+f)
            print(f)
            for label in ori_dir_label_list:
                os.mkdir(dst_dir+"/"+f+"/"+label)
                ori_dir_img_list = os.listdir(ori_dir+"/"+f+"/"+label)
                for imgname in ori_dir_img_list:
                    img = Image.open(ori_dir+"/"+f+"/"+label+"/"+imgname)
                    try:
                        img = tfms(img).unsqueeze(0)
                        torch.save(t.extract_features(img), dst_dir+"/"+f+"/"+label+"/"+imgname[:-5]+'.pt')
                    except:
                        continue