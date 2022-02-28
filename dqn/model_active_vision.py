from gym import spaces
import torch.nn as nn
import torch.nn.functional as F

# Class structure loosely inspired by https://towardsdatascience.com/beating-video-games-with-deep-q-networks-7f73320b9592

class ConvTorso(nn.Module):
    """
    convolutional torso
    """

    def __init__(self,
                 observation_space: spaces.Box):
        """
        Initialise the convolutional layers
        :param observation_space: the state space of the environment
        """
        super().__init__()
             
        # convolution layer
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=observation_space.shape[0], out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU()
        )
   
    def forward(self, x):
        return self.conv(x).view(x.size()[0],-1)
      
class NaturalHead(nn.Module):
    

    def __init__(self,
                 action_space: spaces.Discrete):
        """
        Initialise the fully connected layers for the natural actions
        :param action_space: the action space of the environment
        """
        super().__init__()
     
        # fully connected layer for natural actions  
        self.fc = nn.Sequential(
            nn.Linear(in_features=64*7*7 , out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=action_space.n)
        )
        
    # compute value for natural actions
    def forward(self, conv):
        return self.fc(conv)


class VisionHead(nn.Module):

    def __init__(self,n: 5):
        """
        Initialise the fully connected layers for the visual actions
        :param n: number of visual actions  of the environment
        """
        super().__init__()

        # fully connected layer for natural actions  
        self.fc = nn.Sequential(
            nn.Linear(in_features=64*7*7 , out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=n)
        )
        
    # compute value for natural actions
    def forward(self, conv):
        
        return self.fc(conv)


class ActiveDQN(nn.Module):
    """
    Active vision version of a Deep Q-Network. The architecture is the same as that described in the
    Activision DQN paper.
    """

    def __init__(self,
                 observation_space: spaces.Box,
                 action_space: spaces.Discrete, n_visual_actions:int):
        """
        Initialise the DQN
        :param observation_space: the state space of the environment
        :param action_space: the action space of the environment
        """
        super().__init__()
        assert type(
            observation_space) == spaces.Box, 'observation_space must be of type Box'
        assert len(
            observation_space.shape) == 3, 'observation space must have the form channels x width x height'
        assert type(
            action_space) == spaces.Discrete, 'action_space must be of type Discrete'
        # convolution layer
        self.conv = ConvTorso(observation_space)
        # fully connected layer for natural actions  
        self.fcn = NaturalHead(action_space)
        
        # fully connected layer for visual actions  
        self.fcv = VisionHead(n_visual_actions)
        
        self._visual_forward = None
    # compute value 
    def forward(self, x):
        conv_out = self.conv(x)
        self._visual_forward  = self.fcv(conv_out)
        return self.fcn(conv_out)
    
    def visual_forward(self, x):
        conv_out = self.conv(x)
        self._visual_forward  = self.fcv(conv_out)
        return self._visual_forward
        
class Qv(nn.Module):
    
    def __init__(self, active_dqn: ActiveDQN):

        super().__init__()
       
        # convolution layer sub module
        self.conv = active_dqn.conv
        
        # fully connected layer submodule
        self.fc = active_dqn.fcv
    # compute value 
    def forward(self, x):
        conv_out = self.conv.forward(x)
        return self.fc.forward(conv_out)
        
