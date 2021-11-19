# Import libraries
import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision.transforms import transforms


class ClassifyWear:
    def __init__(self, model_path):
        self.path = model_path

    def predict(self, img_pth):
        # Load Model
        model = torch.load(self.path)
        model.eval()

        # Load and transform Data
        image = Image.open(img_pth).convert('L')
        im = np.asarray(image)
        im = Image.fromarray(im)
        im = ImageOps.invert(im)
        t = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((28, 28)),
                                transforms.Normalize((0.5), (0.5))
                                ])
        im = torch.autograd.Variable(t(im).unsqueeze(0))
        image.close()

        # Predict
        model.eval()
        im = im.view(im.shape[0], -1)
        output = model(im)

        ps = torch.exp(output)
        top_p, top_class = ps.topk(1, dim=1)
        print(top_class[0][0].item())

        classes = {0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat', 5: 'Sandal', \
                   6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'
                   }
        pred = classes[top_class[0][0].item()]

        return int(top_p[0][0].item() * 100), pred

