import argparse
import sys
import os
import re
import multiprocessing
import yaml
from baselines import logger
from baselines.common.misc_util import (
    boolean_flag,
)
import rainbow_ddpg.distributed_train as training
from rainbow_ddpg.models import Actor, Critic
from rainbow_ddpg.prioritized_memory import PrioritizedMemory
from rainbow_ddpg.noise import *
import gym
import tensorflow as tf
from mpi4py import MPI
from micoenv import demo_policies as demo
from common.vec_env import VecFrameStack, VecNormalize, VecEnv
from common.vec_env.vec_video_recorder import VecVideoRecorder
from common.cmd_util import common_arg_parser, parse_unknown_args, make_vec_env, make_env
from common.tf_util import get_session



def build_env(cloth_cfg_path=None, render_path=None, start_state_path=None, num_env=1, seed=1, alg='ddpg'):
    """Daniel: actually construct the env, using 'vector envs' for parallelism.
    For now our cloth env can follow the non-atari and non-retro stuff, because
    I don't think we need a similar kind of 'wrapping' that they do. Note that
    `VecFrameStack` is needed to stack frames, e.g., in Atari we do 4 frame
    stacking. Without that, the states would be size (84,84,1).
    The non-`args` parameters here are for the cloth env.
    """
    
    #Adi: Need to modify the next section because no 'args' parameter
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    #nenv = args.num_env or ncpu
    #alg = args.alg
    #seed = args.seed
    #env_type, env_id = get_env_type(args)
    env_type = 'cloth'
    env_id = 'cloth'
    

    if env_type in {'atari', 'retro'}:
        if alg == 'deepq':
            env = make_env(env_id, env_type, seed=seed, wrapper_kwargs={'frame_stack': True})
        elif alg == 'trpo_mpi':
            env = make_env(env_id, env_type, seed=seed)
        else:
            frame_stack_size = 4
            env = make_vec_env(env_id, env_type, nenv, seed,
                               gamestate=args.gamestate,
                               reward_scale=args.reward_scale)
            env = VecFrameStack(env, frame_stack_size)
    else:
        config = tf.ConfigProto(allow_soft_placement=True,
                                intra_op_parallelism_threads=1,
                                inter_op_parallelism_threads=1)
        config.gpu_options.allow_growth = True
        get_session(config=config)
        flatten_dict_observations = alg not in {'her'}
        env = make_vec_env(env_id, env_type, num_env or 1, seed,
                           reward_scale=1,
                           flatten_dict_observations=flatten_dict_observations,
                           cloth_cfg_path=cloth_cfg_path,
                           render_path=render_path,
                           start_state_path=start_state_path)
        if env_type == 'mujoco':
            env = VecNormalize(env)

    return env


def run(env_id, eval_env_id, noise_type, evaluation, demo_policy, num_dense_layers, dense_layer_size, layer_norm,
        demo_epsilon, replay_alpha, conv_size, **kwargs):
   
    #Need to fill these in! 
    cloth_cfg_path = '/Users/adivganapathi/Documents/UC Berkeley/Rainbow_ddpg-fork/cfg/demo_spaces.yaml'
    render_path = ''
    init_state_path = '/Users/adivganapathi/Documents/UC Berkeley/Rainbow_ddpg-fork/init_states/state_init_easy_81_coverage.pkl'
    
    #Adi: Commenting out the following 7 lines to replace with next section 
    # Create envs.
    #env = gym.make(env_id)
    #if evaluation:
        #eval_env = gym.make(eval_env_id)
    #else:
        #eval_env = None
    #demo_env = gym.make(env_id)

    #Adi: In order to support cloth envs, we can't do a normal gym.make() calls, so we do the following instead:
    seed = None
    #if env_type == 'cloth':
    with open(cloth_cfg_path, 'r') as fh:
        cloth_config = yaml.load(fh)
        #if args.seed is None:
            #args.seed = cloth_config['seed']
        seed = cloth_config['seed'] 
	#Commented out following if statement because I think they already clip the action space to (-1, 1) in the DDPG file
        #if 'clip_act_space' in cloth_config['env']:
            #extra_args['limit_act_range'] = cloth_config['env']['clip_act_space']
    env = build_env(cloth_cfg_path=cloth_cfg_path, render_path=render_path, start_state_path=init_state_path, num_env=1, seed=1, alg='ddpg')
    eval_env = build_env(cloth_cfg_path=cloth_cfg_path, render_path=render_path, start_state_path=init_state_path, num_env=1, seed=1, alg='ddpg')  
    demo_env = build_env(cloth_cfg_path=cloth_cfg_path, render_path=render_path, start_state_path=init_state_path, num_env=1, seed=1, alg='ddpg')  
     

    # Parse noise_type
    action_noise = None
    nb_actions = env.action_space.shape[-1]
    for current_noise_type in noise_type.split(','):
        current_noise_type = current_noise_type.strip()
        if current_noise_type == 'none':
            pass
        elif 'normal' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            action_noise = NormalActionNoise(
                mu=np.zeros(nb_actions),
                sigma=float(stddev) * np.ones(nb_actions))
        elif 'ou' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            action_noise = OrnsteinUhlenbeckActionNoise(
                mu=np.zeros(nb_actions),
                sigma=float(stddev) * np.ones(nb_actions))
        else:
            raise RuntimeError(
                'unknown noise type "{}"'.format(current_noise_type))

    # Initialize prioritized memory buffer
    memory = PrioritizedMemory(
        limit=int(1e4 * 5), alpha=replay_alpha, demo_epsilon=demo_epsilon)

    # Initialize neural nets.
    critic = Critic(num_dense_layers, dense_layer_size, layer_norm)
    actor = Actor(
        nb_actions,
        dense_layer_size,
        layer_norm,
        conv_size=conv_size)
    tf.reset_default_graph()

    # Instantiate demo policy object
    demo_policy_object = None
    if demo.policies[demo_policy]:
        demo_policy_object = demo.policies[demo_policy]()

    # Kick of the training
    eval_avg = training.train(
        env=env,
        env_id=env_id,
        eval_env=eval_env,
        action_noise=action_noise,
        actor=actor,
        critic=critic,
        memory=memory,
        demo_policy=demo_policy_object,
        demo_env=demo_env,
        **kwargs
    )

    # Clean up
    env.close()
    if eval_env is not None:
        eval_env.close()
    demo_env.close()

    return eval_avg


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--env-id',
        type=str,
        default='Pusher-v1')
    parser.add_argument('--eval-env-id', type=str, default='')
    boolean_flag(parser, 'render-eval', default=True)
    boolean_flag(parser, 'render-demo', default=True)
    boolean_flag(parser, 'layer-norm', default=False)
    boolean_flag(parser, 'render', default=False)
    boolean_flag(parser, 'normalize-observations', default=True)
    boolean_flag(parser, 'normalize-state', default=True)
    boolean_flag(parser, 'normalize-aux', default=True)
    parser.add_argument('--critic-l2-reg', type=float, default=1e-2)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--actor-lr', type=float, default=1e-4)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    boolean_flag(parser, 'popart', default=False)
    parser.add_argument('--gamma', type=float, default=0.999)
    parser.add_argument('--reward-scale', type=float, default=1.)
    parser.add_argument('--clip-norm', type=float, default=None)
    parser.add_argument('--nb-epochs', type=int, default=500)
    parser.add_argument('--nb-epoch-cycles', type=int, default=20)
    parser.add_argument('--nb-train-steps', type=int, default=50)
    parser.add_argument('--nb-eval-steps', type=int, default=2)
    parser.add_argument('--nb-rollout-steps', type=int, default=100)
    parser.add_argument('--noise-type', type=str, default='normal_0.2')
    parser.add_argument('--load-file', type=str, default='')
    parser.add_argument('--save-folder', type=str, default='')
    parser.add_argument('--conv-size', type=str, default='small')
    parser.add_argument('--num-timesteps', type=int, default=None)
    parser.add_argument('--num-demo-steps', type=int, default=20)
    parser.add_argument('--num-pretrain-steps', type=int, default=2000)
    parser.add_argument('--run-name', type=str, default='ignore')
    parser.add_argument('--demo-policy', type=str, default='corners')
    parser.add_argument('--lambda-pretrain', type=float, default=5.0)
    parser.add_argument('--lambda-nstep', type=float, default=0.5)
    parser.add_argument('--lambda-1step', type=float, default=1.0)
    parser.add_argument('--replay-beta', type=float, default=0.4)
    parser.add_argument('--reset-to-demo-rate', type=float, default=0.9)
    parser.add_argument('--tau', type=float, default=0.01)
    parser.add_argument('--target-policy-noise', type=float, default=0.0)
    parser.add_argument('--target-policy-noise-clip', type=float, default=0.0)
    parser.add_argument(
        '--policy-and-target-update-period', type=int, default=2)
    parser.add_argument('--dense-layer-size', type=int, default=256)
    parser.add_argument('--num-dense-layers', type=int, default=4)
    parser.add_argument('--num-critics', type=int, default=2)
    parser.add_argument('--nsteps', type=int, default=10)
    parser.add_argument('--demo-terminality', type=int, default=5)
    parser.add_argument('--replay-alpha', type=float, default=0.8)
    parser.add_argument('--demo-epsilon', type=float, default=0.2)
    parser.add_argument(
        '--lambda-obj-conf-predict', type=float, default=500000.0)
    parser.add_argument(
        '--lambda-target-predict', type=float, default=500000.0)
    parser.add_argument(
        '--lambda-gripper-predict', type=float, default=500000.0)

    boolean_flag(parser, 'positive-reward', default=True)
    boolean_flag(parser, 'only-eval', default=False)

    boolean_flag(parser, 'evaluation', default=True)

    kwargs = parser.parse_args()

    # we don't directly specify timesteps for this script, so make sure that if we do specify them
    # they agree with the other parameters
    if kwargs.num_timesteps is not None:
        assert (kwargs.num_timesteps == kwargs.nb_epochs * kwargs.nb_epoch_cycles *
                kwargs.nb_rollout_steps)
    dict_args = vars(kwargs)
    del dict_args['num_timesteps']
    return dict_args


if __name__ == '__main__':
    args = parse_args()
    if not args["eval_env_id"]:
        args["eval_env_id"] = args["env_id"]
    if MPI.COMM_WORLD.Get_rank() == 0:
        logger.configure()
    # Run actual script.
    run(**args)
