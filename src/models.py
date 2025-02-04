import uuid
import torch
import torch.nn as nn
import torch.nn.functional as F


"""
    Implementation of the sender module
"""
class Sender(nn.Module):
    """
    Sender module for the image dataset
    """
    def __init__(self, vision, n_hidden, nodes = [12, 12]):
        """
        :param vision: vision module
        :param n_hidden: number of output features
        :param nodes: number of nodes in the hidden layers
        """
        super(Sender, self).__init__()
        self.uuid = uuid.uuid4()
        self.nodes = nodes

        self.vision = vision
        self.n_hidden = n_hidden

        self.fc1 = nn.Linear(vision.get_n_hidden(), nodes[0])

        for idx in range(len(nodes)):
            if idx == len(nodes) - 1:
                setattr(self, f"fc{idx+2}", nn.Linear(nodes[idx], self.n_hidden))
                break
            setattr(self, f"fc{idx+2}", nn.Linear(nodes[idx], nodes[idx+1]))


    def forward(self, x, input=None, aux_input=None):

        with torch.no_grad():
            x = self.vision(x)

        for idx in range(len(self.nodes)):
            x = F.tanh(getattr(self, f"fc{idx+1}")(x))

        x = getattr(self, f"fc{len(self.nodes)+1}")(x)

        return x


"""
    Implementation of the receiver module
"""
class DiscriReceiver(nn.Module):
    """
    Receiver module for the image dataset
    """
    def __init__(self, n_hidden, image_size):
        """
        :param n_hidden: number of output features.
        :param image_size: tuple of image size (width, height)
        """
        super(DiscriReceiver, self).__init__()
        self.uuid = uuid.uuid4()

        self.image_size = image_size
        self.fc1 = nn.Linear(self.image_size[0]*self.image_size[1]*3, n_hidden)


    def forward(self, x, _input, _aux_input):
        """
        :param x: message from sender
        :param _input: full set of images
        :param _aux_input:
        :return:
        """

        embedded_input = self.embed(_input)
        dots = torch.matmul(embedded_input, torch.unsqueeze(x, dim=-1))
        return dots.squeeze()

    def embed(self, x):
        split_tensors = torch.unbind(x, dim=1)
        return torch.stack([F.tanh(self.fc1(tensor.view(tensor.size(0), -1))) for tensor in split_tensors], dim=1)


"""
    Implementation of the vision module
"""
class Vision(nn.Module):
    """
    Vision module for the image dataset
    """
    def __init__(self, image_size, n_hidden, n_classes):
        """
        :param image_size: tuple of image size (width, height)
        :param n_hidden: number of output features
        :param n_classes: number of classes in the dataset
        """
        super(Vision, self).__init__()
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.conv1 = nn.Conv2d(3, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, n_hidden)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        return x

    def get_n_hidden(self):
        return self.n_hidden

    def get_n_features(self):
        return self.n_features

    def get_n_classes(self):
        return self.n_classes

class PretrainNet(nn.Module):
    def __init__(self, vision_module):
        super(PretrainNet, self).__init__()
        self.vision_module = vision_module
        self.fc = nn.Linear(self.vision_module.get_n_hidden(), self.vision_module.get_n_classes())  # Map to num_classes

    def forward(self, x):
        x = self.vision_module(x)
        x = self.fc(F.leaky_relu(x))
        return x