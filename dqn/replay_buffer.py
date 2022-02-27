import numpy as np
import cv2
from array2gif import write_gif
from PIL import Image, ImageDraw


# class ReplayBuffer:
    # """
    # Simple storage for transitions from an environment.
    # """

    # def __init__(self, size):
        # """
        # Initialise a buffer of a given size for storing transitions
        # :param size: the maximum number of transitions that can be stored
        # """
        # self._storage = []
        # self._maxsize = size
        # self._next_idx = 0
        # self.old_data = (0, 0, 0, 0, 0, 0) #np.zeros((3,84,84))
        # self.old_reward = 0
        
    # def __len__(self):
        # return len(self._storage)

    # def add(self, state, action, visual_action, reward, next_state, done):
        # """
        # Add a transition to the buffer. Old transitions will be overwritten if the buffer is full.
        # :param state: the agent's initial state
        # :param action: the action taken by the agent
        # :param reward: the reward the agent received
        # :param next_state: the subsequent state
        # :param done: whether the episode terminated
        # """
        # # old incomplete transition 
        # old_state, old_action, old_visual_action, old_reward, old_next_state, old_done = self.old_data
        
        # # add transition to memory
        # data = (old_state, old_action, old_visual_action, old_reward, old_next_state, reward, old_done)
        
        # # temporily store incomplete data
        # self.old_data = (state, action, visual_action, reward, next_state, done)
        # # save to check
        # #print(state._frames[0][0].shape )
        # #print(len(state._frames ))
        # # dataset =list()
        # # for x in state._frames:
            # # rgb = np.array([x[0],x[0],x[0]])
            # # print(rgb.shape)
            # # dataset.append(rgb)
        
        
        # #write_gif(dataset, 'state.gif', fps = 2)
        # #im.save('out.gif', save_all=True, append_images=dataset)
        # cv2.imwrite("state.png", state._frames[0][0])
        
        # if self._next_idx >= len(self._storage):
            # self._storage.append(data)
        # else:
            # self._storage[self._next_idx] = data
        # self._next_idx = (self._next_idx + 1) % self._maxsize

    # def _encode_sample(self, indices):
        # states, actions, visual_actions, rewards, next_states, next_rewards, dones = [], [], [], [], [], [], []
        # for i in indices:
            # data = self._storage[i]
            
            # state, action, visual_action, reward, next_state, next_reward, done = data
            
            # states.append(np.array(state, copy=False))
            # actions.append(action)
            # visual_actions.append(visual_action)
            # rewards.append(reward)
            # next_states.append(np.array(next_state, copy=False))
            # next_rewards.append(np.array(next_reward, copy=False))
            # dones.append(done)
            
        # return np.array(states), np.array(actions), np.array(visual_actions),np.array(rewards), np.array(next_states), np.array(next_rewards), np.array(dones)

    # def sample(self, batch_size):
        # """
        # Randomly sample a batch of transitions from the buffer.
        # :param batch_size: the number of transitions to sample
        # :return: a mini-batch of sampled transitions
        # """
        # indices = np.random.randint(0, len(self._storage) - 1, size=batch_size)
        # return self._encode_sample(indices)


class ReplayBuffer:
    """
    Simple storage for transitions from an environment.
    """

    def __init__(self, size):
        """
        Initialise a buffer of a given size for storing transitions
        :param size: the maximum number of transitions that can be stored
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0
        self.old_data = (0, 0, 0, 0, 0, 0) #np.zeros((3,84,84))
        self.old_reward = 0
        
    def __len__(self):
        return len(self._storage)

    def add(self, state, action, visual_action, reward, next_state, done):
        """
        Add a transition to the buffer. Old transitions will be overwritten if the buffer is full.
        :param state: the agent's initial state
        :param action: the action taken by the agent
        :param reward: the reward the agent received
        :param next_state: the subsequent state
        :param done: whether the episode terminated
        """
        # old incomplete transition 
        old_state, old_action, old_visual_action, old_reward, old_next_state, old_done = self.old_data
        

        # add transition to memory
        data = (old_state, old_action, old_visual_action, old_reward, old_next_state, reward, old_done)
        
        # temporily store incomplete data
        self.old_data = (state, action, visual_action, reward, next_state, done)
        
        # add transition to memory
        #data = (state, action, visual_action, reward, next_state, done)
        if old_state ==0 :
            return


        # save to check
        #print(state._frames[0][0].shape )
        #print(len(state._frames ))
        # dataset =list()
        # for x in state._frames:
            # rgb = np.array([x[0],x[0],x[0]])
            # print(rgb.shape)
            # dataset.append(rgb)
        
        
        #write_gif(dataset, 'state.gif', fps = 2)
        #im.save('out.gif', save_all=True, append_images=dataset)
        cv2.imwrite("state.png", state._frames[0][0])
        
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, indices):
        states, actions, visual_actions, rewards, next_states, next_rewards, dones = [], [], [], [], [], [],[]
        for i in indices:
            data = self._storage[i]
            
            state, action, visual_action, reward, next_state, next_reward, done = data
            
            states.append(np.array(state, copy=False))
            actions.append(action)
            visual_actions.append(visual_action)
            rewards.append(reward)
            next_states.append(np.array(next_state, copy=False))
            next_rewards.append(np.array(next_reward, copy=False))
            dones.append(done)
            
        return np.array(states), np.array(actions), np.array(visual_actions),np.array(rewards), np.array(next_states), np.array(next_rewards), np.array(dones)

    def sample(self, batch_size):
        """
        Randomly sample a batch of transitions from the buffer.
        :param batch_size: the number of transitions to sample
        :return: a mini-batch of sampled transitions
        """
        indices = np.random.randint(0, len(self._storage) - 1, size=batch_size)
        return self._encode_sample(indices)
