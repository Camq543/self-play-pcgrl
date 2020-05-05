import multiprocessing
import multiprocessing.connection
import time
from collections import deque
from typing import Dict, List

import cv2
import gym
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.distributions import Categorical
from torch.nn import functional as F

from gym_pcgrl import wrappers


if torch.cuda.is_available():
    print('gpu found')
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")




class Orthogonal(object):

    def __init__(self, scale=1.):
        self.scale = scale


    def __call__(self, shape, dtype=None, partition_info=None):
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4:  # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v  # pick the one with the correct shape
        q = q.reshape(shape)
        return (self.scale * q[:shape[0], :shape[1]]).astype(np.float32)


    def get_config(self):
        return {
            'scale': self.scale
        }


class Game:


    def __init__(self, seed: int):


        self.env = gym.make('BreakoutNoFrameskip-v4')
        self.env.seed(seed)


        self.obs_2_max = np.zeros((2, 84, 84, 1), np.uint8)

        self.obs_4 = np.zeros((84, 84, 4))

        self.rewards = []

        self.lives = 0


    def step(self, action):

        reward = 0.
        done = None

        for i in range(4):
            obs, r, done, info = self.env.step(action)

            if i >= 2:
                self.obs_2_max[i % 2] = self._process_obs(obs)

            reward += r

            lives = self.env.unwrapped.ale.lives()

            if lives < self.lives:
                done = True
            self.lives = lives


            if done:
                break

        self.rewards.append(reward)

        if done:
            episode_info = {"reward": sum(self.rewards),
                        "length": len(self.rewards)}
            self.reset()

        else:
            episode_info = None

            obs = self.obs_2_max.max(axis=0)

            self.obs_4 = np.roll(self.obs_4, shift=-1, axis=-1)
            self.obs_4[..., -1:] = obs

        return self.obs_4, reward, done, episode_info            

    def reset(self):

        obs = self.env.reset()
        obs = self._process_obs(obs)

        self.obs_4[..., 0:] = obs
        self.obs_4[..., 1:] = obs
        self.obs_4[..., 2:] = obs
        self.obs_4[..., 3:] = obs
        self.rewards = []

        self.lives = self.env.unwrapped.ale.lives()

        return self.obs_4

    @staticmethod
    def _process_obs(obs):

        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
        return obs[:, :, None]  # Shape (84, 84, 1)


def worker_process(remote: multiprocessing.connection.Connection, env_name: str,crop_size: int,n_agents:int,kwargs:Dict):

    game = wrappers.CroppedImagePCGRLWrapper(env_name, crop_size, n_agents,**kwargs)

    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            # print('stepping')
            temp = game.step(data)
            # print(temp)
            remote.send(temp)
        elif cmd == "reset":
            # print('resetting')
            temp = game.reset()
            # print(temp)
            remote.send(temp)
        elif cmd == "close":
            remote.close()
            break
        else:
            raise NotImplementedError




class Worker:

    child: multiprocessing.connection.Connection
    process: multiprocessing.Process

    def __init__(self, env_name,crop_size,n_agents,kwargs):

        self.child, parent = multiprocessing.Pipe()
        self.process = multiprocessing.Process(target=worker_process, args=(parent, env_name,crop_size,n_agents,kwargs))
        self.process.start()




class Model(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=32,
                               kernel_size=1,
                               stride=1,
                               padding=0)
        nn.init.orthogonal_(self.conv1.weight, np.sqrt(2))

        self.conv2 = nn.Conv2d(in_channels=32,
                               out_channels=64,
                               kernel_size=1,
                               stride=1,
                               padding=0)
        nn.init.orthogonal_(self.conv2.weight, np.sqrt(2))

        self.conv3 = nn.Conv2d(in_channels=64,
                               out_channels=64,
                               kernel_size=1,
                               stride=1,
                               padding=0)
        nn.init.orthogonal_(self.conv3.weight, np.sqrt(2))


        self.lin = nn.Linear(in_features=28 * 28 * 64,
                             out_features=512)
        nn.init.orthogonal_(self.lin.weight, np.sqrt(2))

        self.pi_logits = nn.Linear(in_features=512,
                                   out_features=3)
        nn.init.orthogonal_(self.pi_logits.weight, np.sqrt(2))

        self.value = nn.Linear(in_features=512,
                                 out_features=1)
        nn.init.orthogonal_(self.value.weight, 1)


    def forward(self, obs: np.ndarray):
        # print('runnin forward')
        h: torch.Tensor

        h = F.relu(self.conv1(obs))
        # print(h.shape)
        h = F.relu(self.conv2(h))
        # print(h.shape)
        h = F.relu(self.conv3(h))
        # print(h.shape)
        h = h.reshape((-1, 28 * 28 * 64))
        # print(h.shape)

        h = F.relu(self.lin(h))
        # print(h.shape)

        # print(h)
        # print('logits',self.pi_logits(h))
        pi = Categorical(logits=self.pi_logits(h))
        value = self.value(h).reshape(-1)

        return pi, value



def obs_to_torch(obs: np.ndarray) -> torch.Tensor:
    # print("before",obs.shape)
    obs = np.swapaxes(obs, 1, 3)
    # print("after first",obs.shape)
    obs = np.swapaxes(obs, 3, 2)
    # print("after second",obs.shape)



    return torch.tensor(obs, dtype=torch.float32, device=device)



class Trainer:

    def __init__(self, model: Model):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=2.5e-4)

    def train(self,
              samples: Dict[str, np.ndarray],
              learning_rate: float,
              clip_range: float):

        sampled_obs = samples['obs']

        sampled_action = samples['actions']

        sampled_return = samples['values'] + samples['advantages']

        sampled_normalized_advantage = Trainer._normalize(samples['advantages'])

        sampled_neg_log_pi = samples['neg_log_pis']

        sampled_value = samples['values']

        pi, value = self.model(sampled_obs)

        neg_log_pi = -pi.log_prob(sampled_action)

        ratio: torch.Tensor = torch.exp(sampled_neg_log_pi - neg_log_pi)

        clipped_ratio = ratio.clamp(min=1.0 - clip_range,
                                    max=1.0 + clip_range)
        policy_reward = torch.min(ratio * sampled_normalized_advantage,
                                  clipped_ratio * sampled_normalized_advantage)
        policy_reward = policy_reward.mean()

        entropy_bonus = pi.entropy()
        entropy_bonus = entropy_bonus.mean()

        clipped_value = sampled_value + (value - sampled_value).clamp(min=-clip_range,
                                                                      max=clip_range)
        vf_loss = torch.max((value - sampled_return) ** 2, (clipped_value - sampled_return) ** 2)
        vf_loss = 0.5 * vf_loss.mean()

        loss: torch.Tensor = -(policy_reward - 0.5 * vf_loss + 0.01 * entropy_bonus)

        for pg in self.optimizer.param_groups:
            pg['lr'] = learning_rate
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        self.optimizer.step()


        approx_kl_divergence = .5 * ((neg_log_pi - sampled_neg_log_pi) ** 2).mean()
        clip_fraction = (abs((ratio - 1.0)) > clip_range).type(torch.FloatTensor).mean()

        return [policy_reward,
                vf_loss,
                entropy_bonus,
                approx_kl_divergence,
                clip_fraction]



    @staticmethod
    def _normalize(adv: np.ndarray):

#

        return (adv - adv.mean()) / (adv.std() + 1e-8)


class MultiTrainer:

    def __init__(self, models: List[Model], learning_rate=2.5e-4):
        self.n_agents = len(models)
        self.models = models
        self.optimizers = [optim.Adam(model.parameters(), lr=learning_rate) for model in self.models]

    def train(self,
              samples: List[Dict[str, np.ndarray]],
              learning_rate: float,
              clip_range: float):

        to_return = []

        for i in range(self.n_agents):
            sampled_obs = samples[i]['obs']

            sampled_action = samples[i]['actions']

            sampled_return = samples[i]['values'] + samples[i]['advantages']

            sampled_normalized_advantage = Trainer._normalize(samples[i]['advantages'])

            sampled_neg_log_pi = samples[i]['neg_log_pis']

            sampled_value = samples[i]['values']

            pi, value = self.models[i](sampled_obs)

            neg_log_pi = -pi.log_prob(sampled_action)

            ratio: torch.Tensor = torch.exp(sampled_neg_log_pi - neg_log_pi)

            clipped_ratio = ratio.clamp(min=1.0 - clip_range,
                                        max=1.0 + clip_range)
            policy_reward = torch.min(ratio * sampled_normalized_advantage,
                                      clipped_ratio * sampled_normalized_advantage)
            policy_reward = policy_reward.mean()

            entropy_bonus = pi.entropy()
            entropy_bonus = entropy_bonus.mean()

            clipped_value = sampled_value + (value - sampled_value).clamp(min=-clip_range,
                                                                          max=clip_range)
            vf_loss = torch.max((value - sampled_return) ** 2, (clipped_value - sampled_return) ** 2)
            vf_loss = 0.5 * vf_loss.mean()

            loss: torch.Tensor = -(policy_reward - 0.5 * vf_loss + 0.01 * entropy_bonus)

            for pg in self.optimizers[i].param_groups:
                pg['lr'] = learning_rate
            self.optimizers[i].zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.models[i].parameters(), max_norm=0.5)
            self.optimizers[i].step()


            approx_kl_divergence = .5 * ((neg_log_pi - sampled_neg_log_pi) ** 2).mean()
            clip_fraction = (abs((ratio - 1.0)) > clip_range).type(torch.FloatTensor).mean()
            

            to_return.append([policy_reward,
                vf_loss,
                entropy_bonus,
                approx_kl_divergence,
                clip_fraction])


        return to_return



    @staticmethod
    def _normalize(adv: np.ndarray):

#

        return (adv - adv.mean()) / (adv.std() + 1e-8)


class Main(object):

    def __init__(self):

        self.gamma = 0.99
        self.lamda = 0.95

        self.updates = 10000

        self.epochs = 4

        self.n_workers = 8

        self.n_agents = 2

        self.worker_steps = 128

        self.n_mini_batch = 4

        self.batch_size = self.n_workers * self.worker_steps

        self.mini_batch_size = self.batch_size // self.n_mini_batch

        self.models = []

        assert (self.batch_size % self.n_mini_batch == 0)

        game = 'binary'
        representation = 'narrow'
        self.n_agents = 2

        kwargs = {
            'change_percentage': 0.4,
            'verbose': True
        }


        self.env_name = '{}-{}-v0'.format(game, representation)
        if game == "binary":
            kwargs['cropped_size'] = 28
        elif game == "zelda":
            kwargs['cropped_size'] = 22
        elif game == "sokoban":
            kwargs['cropped_size'] = 10

        self.crop_size = kwargs.get('cropped_size', 28)

        self.workers = [Worker(self.env_name, self.crop_size, self.n_agents, kwargs) for i in range(self.n_workers)]

        self.obs = np.zeros((self.n_agents,self.n_workers, 28, 28, 1), dtype=np.uint8)

        for worker in self.workers:
            worker.child.send(("reset", None))
        for i, worker in enumerate(self.workers):
            self.obs[:,i] = worker.child.recv()

        for i in range(self.n_agents):
            model = Model()
            model.to(device)
            self.models.append(model)

        self.trainer = MultiTrainer(self.models)



    def sample(self) -> (Dict[str, np.ndarray], List):

        rewards = np.zeros((self.n_agents, self.n_workers, self.worker_steps), dtype=np.float32) 
        actions = np.zeros((self.n_agents, self.n_workers, self.worker_steps), dtype=np.int32) 
        dones = np.zeros((self.n_agents, self.n_workers, self.worker_steps), dtype=np.bool) 
        obs = np.zeros((self.n_agents, self.n_workers, self.worker_steps, 28, 28, 1), dtype=np.uint8) 
        neg_log_pis = np.zeros((self.n_agents, self.n_workers, self.worker_steps), dtype=np.float32) 
        values = np.zeros((self.n_agents, self.n_workers, self.worker_steps), dtype=np.float32) 
        episode_infos = []

        samples_return = []
        info_return = []

        
        for t in range(self.worker_steps):
            for i in range(self.n_agents):

                obs[i,:, t] = self.obs[i]

                temp = obs_to_torch(self.obs[i])
                # print(temp.shape)
                pi, v = self.models[i](temp)
                # print(v)
                values[i,:, t] = v.cpu().data.numpy()
                a = pi.sample()
                # print(a)
                actions[i,:, t] = a.cpu().data.numpy()
                neg_log_pis[i,:, t] = -pi.log_prob(a).cpu().data.numpy()

            # print("actions",actions)


            for w, worker in enumerate(self.workers):
                worker.child.send(("step", actions[:,w, t]))

            for w, worker in enumerate(self.workers):

                self.obs[:,w], rewards[:,w, t], dones[:,w, t], info = worker.child.recv()

                if info:
                    info['obs'] = obs[:,w, t, :, :, 0]
                    episode_infos.append(info)

        for i in range(self.n_agents):
            advantages = self._calc_advantages(dones[i], rewards[i], values[i],i)
            samples = {
                'obs': obs[i],
                'actions': actions[i],
                'values': values[i],
                'neg_log_pis': neg_log_pis[i],
                'advantages': advantages
            }

            samples_flat = {}
            for k, v in samples.items():
                v = v.reshape(v.shape[0] * v.shape[1], *v.shape[2:])
                if k == 'obs':
                    samples_flat[k] = obs_to_torch(v)
                else:
                    samples_flat[k] = torch.tensor(v, device=device)

            samples_return.append(samples_flat)

        return samples_flat, episode_infos

    def _calc_advantages(self, dones: np.ndarray, rewards: np.ndarray,
                         values: np.ndarray, ind:int) -> np.ndarray:

        advantages = np.zeros((self.n_workers, self.worker_steps), dtype=np.float32)
        last_advantage = 0

        _, last_value = self.models[ind](obs_to_torch(self.obs[ind]))
        last_value = last_value.cpu().data.numpy()

        for t in reversed(range(self.worker_steps)):

            mask = 1.0 - dones[:, t]
            last_value = last_value * mask
            last_advantage = last_advantage * mask

            delta = rewards[:, t] + self.gamma * last_value - values[:, t]

            last_advantage = delta + self.gamma * self.lamda * last_advantage

            advantages[:, t] = last_advantage

            last_value = values[:, t]

        return advantages




    def train(self, samples: List[Dict[str, np.ndarray]], learning_rate: float, clip_range: float):

        train_info = [[],[]]

        for _ in range(self.epochs):

            indexes = torch.randperm(self.batch_size)
            mini_batches = []

            for start in range(0, self.batch_size, self.mini_batch_size):
                for i in range(self.n_agents):
                    end = start + self.mini_batch_size
                    mini_batch_indexes = indexes[start: end]
                    mini_batch = {}
                    for k, v in samples.items():
                        mini_batch[k] = v[mini_batch_indexes]
                    mini_batches.append(mini_batch)

                res = self.trainer.train(learning_rate=learning_rate,
                                         clip_range=clip_range,
                                         samples=mini_batches)
                for i in range(self.n_agents):  
                    train_info[i].append(res[i])

            returns = []
            for i in range(self.n_agents):
                returns.append(np.mean(train_info[i], axis=0))

        return returns




    def run_training_loop(self):

        episode_info = deque(maxlen=100)

        for update in range(self.updates):
            time_start = time.time()
            progress = update / self.updates

            learning_rate = 2.5e-4 * (1 - progress)
            clip_range = 0.1 * (1 - progress)

            samples, sample_episode_info = self.sample()

            self.train(samples, learning_rate, clip_range)

            time_end = time.time()

            fps = int(self.batch_size / (time_end - time_start))

            episode_info.extend(sample_episode_info)

            reward_mean, length_mean = Main._get_mean_episode_info(self.n_agents,episode_info)

            agent1 = reward_mean[0]
            agent2 = reward_mean[1]

            print(f"{update:4}: fps={fps:3} agent_1_reward={agent1:.2f} agent_2_reward={agent2:.2f} length={length_mean:.3f}")


    @staticmethod
    def _get_mean_episode_info(n_agents,episode_info):

        if len(episode_info) > 0:
            toreturn = [[]]
            for i in range(n_agents):
                toreturn[0].append(np.mean([info["reward"][i] for info in episode_info]))
            toreturn.append(np.mean([info["length"] for info in episode_info]))
            return toreturn
        else:
            return (np.nan,np.nan), np.nan



    def destroy(self):

        for worker in self.workers:
            worker.child.send(("close", None))


if __name__ == "__main__":
    m = Main()
    m.run_training_loop()
    m.destroy()