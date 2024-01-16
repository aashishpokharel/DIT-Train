import io

import torch
from PIL import Image
from torchvision import transforms

from .main import Net

class DeepMnist(object):

    def __init__(self):
        self._model = Net()
        self._model.load_state_dict(
            torch.load("/storage/model.pt", map_location=torch.device("cpu"))
        )
        self._model.eval()


    def predict(self, X, features_names):
        data = transforms.ToTensor()(Image.open(io.BytesIO(X)))
        return self._model(data[None, ...]).detach().numpy()
    