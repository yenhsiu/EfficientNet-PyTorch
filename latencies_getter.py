import json
from PIL import Image
import time
import torch
from torchvision import transforms
from efficientnet_pytorch import EfficientNet

# Open image
model_names = ['efficientnet-b0','efficientnet-b1','efficientnet-b2','efficientnet-b3','efficientnet-b4','efficientnet-b5','efficientnet-b6','efficientnet-b7']

model_latencies_dict = {}
for m in model_names:
    model_name = m
    model = EfficientNet.from_pretrained(m)
    image_size = EfficientNet.get_image_size(model_name)
    # Preprocess image
    tfms = transforms.Compose([transforms.Resize(image_size), transforms.CenterCrop(image_size), 
                           transforms.ToTensor(),
                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
    img = Image.open('img.jpg')
    img = tfms(img).unsqueeze(0)
    model.eval()
    
    start = time.process_time()
    for i in range(50):
        with torch.no_grad():
            logits = model(img)
    end = time.process_time()
    print(m," Execution time: %f s" % (end - start))
    model_latencies_dict[m] = (end - start)/50

    print(model_latencies_dict)

    import matplotlib.pyplot as plt 
    names = ["input layer"]
    plt.plot(model_latencies_dict.keys(), model_latencies_dict.values(), color='red')
    plt.xticks(rotation=45)
    plt.grid()
    # plt.show()
    plt.savefig('model_latencies.png')

    import json

    # as requested in comment

    with open('vodka_latenies.txt', 'w') as file:
        file.write(json.dumps(model_latencies_dict))