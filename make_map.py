from multiPCGRL import load_models, Model, obs_to_torch
from gym_pcgrl import wrappers
import matplotlib.pyplot as plt
import torch
import gym
import numpy as np
import time


if torch.cuda.is_available():
    print('gpu found')
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

def reshape_obs(obs):
    return obs[:,np.newaxis,:,:,:]


def build(game, representation, model_path, n_agents, make_gif, gif_name, **kwargs):

    env_name = '{}-{}-v0'.format(game, representation)  

    if game == "binary":
        kwargs['cropped_size'] = 28
    elif game == "zelda":
        kwargs['cropped_size'] = 22
    elif game == "sokoban":
        kwargs['cropped_size'] = 10
    kwargs['render'] = True

    crop_size = kwargs.get('cropped_size',28)

    temp_env = wrappers.CroppedImagePCGRLWrapper(self.env_name, self.crop_size, self.n_agents,**kwargs)

    if kwargs['restrict_map']:  
        map_restrictions = [{'x': (0,(temp_env.pcgrl_env._prob._width - 1)//2 - 1),
                            'y': (0,temp_env.pcgrl_env._prob._height - 1)},
                            {'x':((temp_env.pcgrl_env._prob._width - 1)//2,(temp_env.pcgrl_env._prob._width - 1)),
                            'y':(0,temp_env.pcgrl_env._prob._height - 1)}]
        kwargs['map_restrictions'] = map_restrictions

    env = wrappers.CroppedImagePCGRLWrapper(env_name, crop_size, n_agents,**kwargs)

    n_actions = env.action_space.n
    obs = env.reset()
    # print(obs.shape)


    models, optimizers, _,_ = load_models(device, model_path, n_agents,obs.shape[-1],crop_size,n_actions)

    obs = reshape_obs(obs)
    # print(obs.shape)

    frames = []

    done = False
    while not done:
        if make_gif:
            frames.append(env.render(mode='rgb_array'))
        env.render()
        actions = []
        for i in range(n_agents):
            pi, _ = models[i](obs_to_torch(obs[i]))
            a = pi.sample()
            actions.append(a)
            # print(actions) 
        obs, rewards, dones, info = env.step(actions)
        obs = reshape_obs(obs)
        if True in dones:
            done = True
    print(info)

    if make_gif:
        frames[0].save(gif_name,save_all=True,append_images = frames[1:])

    time.sleep(10)



################################## MAIN ########################################
game = 'binary'
representation = 'narrow'
n_agents = 2
make_gif = True
kwargs = {
            'change_percentage': 0.4,
            'verbose': True,
            'negative_switch': False,
            'render': False
            'restrict map':False
}

model_path = 'models/{}/{}/{}{}'.format(game,representation,'negative_switch_' if kwargs['negative_switch'] else '','map_restricted_' if kwargs['restrict_map'] else '')
gif_name = 'gifs/{}-{}{}{}.gif'.format(game, representation,'_negative_switch' if kwargs['negative_switch'] else '','map_restricted_' if kwargs['_restrict_map'] else '')


if __name__ == '__main__':
    build(game, representation, model_path, n_agents, make_gif, gif_name, **kwargs)