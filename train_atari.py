# Inspired from https://github.com/raillab/dqn
import random
import numpy as np
import gym

from dqn.agent import DQNAgent
from dqn.replay_buffer import ReplayBuffer
from dqn.wrappers import *
import torch
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='DQN Atari')
    parser.add_argument('--load-checkpoint-file', type=str, default=None, 
                        help='Where checkpoint file should be loaded from (usually results/checkpoint.pth)')

    args = parser.parse_args()
 # If you have a checkpoint file, spend less time exploring
    if(args.load_checkpoint_file):
        eps_start= 0.01
    else:
        eps_start= 1
 # init hyper-parameters dict
    
    hyper_params = {
        "seed": 42,  # which seed to use
        "env": "PongNoFrameskip-v4",  # name of the game
        "replay-buffer-size": int(5e3),  # replay buffer size
        "learning-rate": 1e-4,  # learning rate for Adam optimizer
        "discount-factor": 0.99,  # discount factor
        "dqn_type":"nature", # dnn archictecture
        "num-steps": int(1e6),# total number of steps to run the environment for
        "batch-size": 32,  # number of transitions to optimize at the same time
        "learning-starts": 1,# 10000,  # number of steps before learning starts
        "learning-freq": 1,  # number of iterations between every optimization step
        "use-double-dqn": True,  # use double deep Q-learning
        "target-update-freq": 1000,  # number of iterations between every target network update
        "eps-start": eps_start,  # e-greedy start threshold
        "eps-end": 0.01,  # e-greedy end threshold
        "eps-fraction": 0.1,  # fraction of num-steps
        "print-freq": 10
    }
 # init random libs with seed
  
    np.random.seed(hyper_params["seed"])
    random.seed(hyper_params["seed"])
    
 # init game environment with gym
    
    av = ActiveVision()
    
    assert "NoFrameskip" in hyper_params["env"], "Require environment with no frameskip"
    env = gym.make(hyper_params["env"])
    env.seed(hyper_params["seed"])

    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    env = FireResetEnv(env)
    env = WarpFrame(env, av)
    #env = ActiveVision(env, WarpFrame) # active vision wrapper
    env = PyTorchFrame(env)
    env = ClipRewardEnv(env)
    env = FrameStack(env, 4)
    env = gym.wrappers.Monitor(
        env, './video/', video_callable=lambda episode_id: episode_id % 50 == 0, force=True)
    
# init Replay buffer 

    replay_buffer = ReplayBuffer(hyper_params["replay-buffer-size"])
    
# init  DQN Agent

    agent = DQNAgent(
        env.observation_space,
        env.action_space,
        replay_buffer,
        use_double_dqn=hyper_params["use-double-dqn"],
        lr=hyper_params['learning-rate'],
        batch_size=hyper_params['batch-size'],
        gamma=hyper_params['discount-factor'],
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        dqn_type=hyper_params["dqn_type"],
        n_visual_actions = len(av.visual_actions)
    )

    
# loading a pre-trained network parameters

    if(args.load_checkpoint_file):
        print(f"Loading a policy - { args.load_checkpoint_file } ")
        agent.policy_network.load_state_dict(
            torch.load(args.load_checkpoint_file, map_location=torch.device('cpu') )) 
        #print(f"policy { args.load_checkpoint_file } succesfully loaded'")
      

    eps_timesteps = hyper_params["eps-fraction"] * \
        float(hyper_params["num-steps"])
        
 # init rewards list
    episode_rewards = [0.0]

# start environment
    
    state = env.reset()
    print("main training loop") 
    
# main training loop

    for t in range(hyper_params["num-steps"]):
    
        # exploit vs explore
        fraction = min(1.0, float(t) / eps_timesteps)
        eps_threshold = hyper_params["eps-start"] + fraction * \
            (hyper_params["eps-end"] - hyper_params["eps-start"])
        sample = random.random()

        if(sample > eps_threshold):
            # Exploit : pick optimal action using network
            natural_action = agent.act(state)
            visual_action = agent.vision_act(state)
        else:
            # Explore
            natural_action = env.action_space.sample()
            visual_action =  random.sample(av.visual_actions,1)[0]
        
        # pick visual action 
        visual_action = 4
    
        # env simulate action; get new state/reward
        
        #next_state, reward, done, info = env.step(natural_action)
        next_state, reward, done, info =  av.step(env, natural_action, visual_action)
        # update memory with new transition/experience
        agent.memory.add(state, natural_action, visual_action, reward, next_state, float(done))
        
        # update state
        
        state = next_state

        # increment episode reward 
        episode_rewards[-1] += reward
        
        # reset env if terminal state
        if done:
            state = env.reset()
            episode_rewards.append(0.0)

        # online network optimize
        if t > hyper_params["learning-starts"] and t % hyper_params["learning-freq"] == 0:
            agent.optimise_td_loss()

        # target network optimize
        if t > hyper_params["learning-starts"] and t % hyper_params["target-update-freq"] == 0:
            agent.update_target_network()

        num_episodes = len(episode_rewards)
        
        # print useful info : nb of  steps and episodes, time spent exploring, avg reward/score over 100 last episodes
        # save network as checkpoint
        #save rewards_per_episode
        if done and hyper_params["print-freq"] is not None and len(episode_rewards) % hyper_params[
                "print-freq"] == 0:
            mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
            print("********************************************************")
            print("steps: {}".format(t))
            print("episodes: {}".format(num_episodes))
            print("mean 100 episode reward: {}".format(mean_100ep_reward))
            print("% time spent exploring: {}".format(int(100 * eps_threshold)))
            print("********************************************************")
            torch.save(agent.policy_network.state_dict(), f'checkpoint.pth')
            np.savetxt('rewards_per_episode.csv', episode_rewards,
                       delimiter=',', fmt='%1.3f')
