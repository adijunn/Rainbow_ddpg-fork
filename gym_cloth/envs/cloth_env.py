"""
An OpenAI Gym-style environment for the cloth smoothing experiments. It's not
exactly their interface because we pass in a configuration file. See README.md
document in this directory for details.
"""
import pyximport; pyximport.install() #Adi: Added this so that we can import .pyx files (cython)
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import os
from os.path import join
import subprocess
import sys
import time
import datetime
import logging
import json
import yaml
import subprocess
import trimesh
import cv2
import datetime
import pickle
import copy
from gym_cloth.physics.cloth import Cloth
from gym_cloth.physics.point import Point
from gym_cloth.physics.gripper import Gripper
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy
from scipy.spatial import ConvexHull

_logging_setup_table = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
}

# Listed clockwise. These have norm one. Tune `reduce_factor` from config.
K = np.sqrt(2)/2

# @deprecated
_ACT_TO_DIRECTION = {
    0 : ( 0,  1),   # North
    1 : ( K,  K),   # North East
    2 : ( 1,  0),   # East
    3 : ( K, -K),   # South East
    4 : ( 0, -1),   # South
    5 : (-K, -K),   # South West
    6 : (-1,  0),   # West
    7 : (-K,  K),   # North West
}

# @deprecated
_ACT_TO_NAME = {
    0 : 'North',
    1 : 'NorthEast',
    2 : 'East',
    3 : 'SouthEast',
    4 : 'South',
    5 : 'SouthWest',
    6 : 'West',
    7 : 'North West',
}

# Thresholds for successful episode completion.
# Keys are possible values for reward_type in config.
_REWARD_THRESHOLDS = {
    'coverage': 0.85,
    'coverage-delta': 0.85,
    'height': 0.85,
    'height-delta': 0.85,
    'variance': 2,
    'variance-delta': 2,
    'sparse1': 900,
    'folding-number': 0 # TODO
}

_EPS = 1e-5


class ClothEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, cfg_file, subrank=None, start_state_path=None):
        """Various initialization for the environment.

        Not to be confused with the initialization for the _cloth_.

        See the following for how to use proper seeding in gym:
          https://github.com/openai/gym/blob/master/gym/utils/seeding.py
          https://github.com/openai/gym/blob/master/gym/envs/toy_text/discrete.py
          https://stackoverflow.com/questions/5836335/
          https://stackoverflow.com/questions/22994423/

        The bounds of (1,1,1) are for convenience and should be the same bounds
        as the cloth has internally.

        RL algorithms refer to observation_space and action_space when building
        neural networks.  Also, we may need to sample from our action space for
        a random policy.  For the actions, we should enforce clipping, but it's
        a subtle issue. See how others do it for MuJoCo.

        Optional:
        - `subrank` if we have multiple envs in parallel, to make it easy to
          tell which env corresponds to certain loggers.
        - `start_state_path` if we want to force cloth to start at a specific
          state.  Represents the path to the state file.
        """
        with open(cfg_file, 'r') as fh:
            cfg = yaml.safe_load(fh)
        self.cfg              = cfg
        self.cfg_file         = cfg_file
        self.max_actions      = cfg['env']['max_actions']
        self.max_z_threshold  = cfg['env']['max_z_threshold']
        self.iters_up         = cfg['env']['iters_up']
        self.iters_up_rest    = cfg['env']['iters_up_rest']
        self.iters_pull_max   = cfg['env']['iters_pull_max']
        self.iters_grip_rest  = cfg['env']['iters_grip_rest']
        self.iters_rest       = cfg['env']['iters_rest']
        self.updates_per_move = cfg['env']['updates_per_move']
        self.reduce_factor    = cfg['env']['reduce_factor']
        self.grip_radius      = cfg['env']['grip_radius']
        self.start_x          = cfg['init']['start_grip_x']
        self.start_y          = cfg['init']['start_grip_y']
        self.render_gl        = cfg['init']['render_opengl']
        self.use_noise_init   = cfg['init']['use_noise']
        self._clip_act_space  = cfg['env']['clip_act_space']
        self._delta_actions   = cfg['env']['delta_actions']
        self._obs_type        = cfg['env']['obs_type']
        self.bounds = bounds  = (1, 1, 1)
        self.render_proc      = None
        self.render_port      = 5556
        self._logger_idx      = subrank

        if start_state_path is not None:
            with open(start_state_path, 'rb') as fh:
                self._start_state = pickle.load(fh)
        else:
            self._start_state = None

        # Reward design. Very tricky ... FOW NOW assume coverage.
        self.reward_type = cfg['env']['reward_type']
        assert 'coverage' in self.reward_type
        self._prev_reward = 0
        self._neg_living_rew = -0.05
        self._nogrip_penalty = 10 * self._neg_living_rew
        self._tear_penalty = -10
        self._oob_penalty = -10
        self._cover_success = 100.
        self._act_bound_factor = 1.0
        self._act_pen_limit = 3.0
        self._current_coverage = 0.0

        # Create observation ('1d', '3d') and action spaces. Possibly make the
        # obs_type and other stuff user-specified parameters.
        self._slack = 0.25
        self.num_w = num_w = cfg['cloth']['num_width_points']
        self.num_h = num_h = cfg['cloth']['num_height_points']
        self.num_points = num_w * num_h
        lim = 100
        if self._obs_type == '1d':
            #Added the observation space line
            self.obslow  = np.ones((3 * self.num_points,)) * -lim
            self.obshigh = np.ones((3 * self.num_points,)) * lim
            self.observation_space = spaces.Box(self.obslow, self.obshigh)
        elif self._obs_type == 'blender':
            #Adi: Had to change to 240x320 and the observation space line as well
            self.obslow  = np.zeros((240, 320, 3))
            self.obshigh = np.ones((240, 320, 3))
            self.observation_space = spaces.Box(self.obslow, self.obshigh, dtype=np.uint8)
        else:
            raise ValueError(self._obs_type)
        self.observation_space = spaces.Box(self.obslow, self.obshigh)

        # Ideally want the gripper to grip points in (0,1) for x and y. Perhaps
        # consider slack that we use for out of bounds detection? Subtle issue.
        b0, b1 = self.bounds[0], self.bounds[1]
        if self._clip_act_space:
            # Applies regardless of 'delta actions' vs non deltas.
            self.action_space = spaces.Box(
                low= np.array([-1., -1., -1., -1.]),
                high=np.array([ 1.,  1.,  1.,  1.])
            )
        else:
            if self._delta_actions:
                self.action_space = spaces.Box(
                    low= np.array([0., 0., -1., -1.]),
                    high=np.array([1., 1.,  1.,  1.])
                )
            else:
                self.action_space = spaces.Box(
                    low= np.array([  -self._slack,   -self._slack, 0.0, -np.pi]),
                    high=np.array([b0+self._slack, b1+self._slack, 1.0,  np.pi])
                )


        # Bells and whistles
        self._setup_logger()
        self.seed()
        self.debug_viz = cfg['init']['debug_matplotlib']

        #Adi: Added following lines to support our cloth env low dim state
        self.reset()
        self.state_dim = self.state_vector().shape
        high = np.inf * np.ones(self.state_dim)
        low = -high
        aux_high = np.ones((16,)) * 10
        self.state_space = spaces.Box(low, high)
        #Doing this to pass in hardcoded shape of state_space
        print("STATE SPACE SHAPE: ")
        print(self.state_space.shape)
        self.aux_space = spaces.Box(-aux_high, aux_high)
        print("AUX SPACE SHAPE: ")
        print(self.aux_space.shape)
   

    #Adi: Added this method to make compatible with Rainbow DDPG and to differentiate between getting observation and getting state
    def state_vector(self):
        lst = []
        for pt in self.cloth.pts:
            lst.extend([pt.x, pt.y, pt.z])
        low_dim = np.array(lst) 
        return low_dim.astype(np.float32)

    #Adi: Added this method to make compatible with Rainbow DDPG.
    def get_state(self):
        return self.state_vector()

    #Adi: Added this method to make compatible with Ranbow DDPG.  We need to think about how we want to define our auxiliary output.  The commented out portion is how they define it.  It also should match the aux_space shape I think.
    def get_aux(self):
        #grip_state = self.p.getLinkState(self.arm.armId, 6, computeLinkVelocity=1)
        #return np.concatenate([self.arm.get_joint_poses(), np.array(grip_state[0]), self.arm.goalPosition])
        #Temporary return statement
        return spaces.Box(np.ones((16,))*-10, np.ones((16,))*10)


    @property
    #This method should really be called "get_obs"
    def state(self):
        if self._obs_type == '1d':
            lst = []
            for pt in self.cloth.pts:
                lst.extend([pt.x, pt.y, pt.z])
            return np.array(lst)

        elif self._obs_type == 'blender':
            bhead = '/tmp/blender'
            if not os.path.exists(bhead):
                os.makedirs(bhead)

            # Step 1: make obj file using trimesh, and save to directory.
            # TODO wh is hard coded for now, should fix later.
            wh = 25
            cloth = np.array([[p.x, p.y, p.z] for p in self.cloth.pts])
            assert cloth.shape[1] == 3, cloth.shape
            faces = []
            for r in range(wh-1):
                for c in range(wh-1):
                    pp = r*wh + c
                    faces.append( [pp,   pp+wh, pp+1] )
                    faces.append( [pp+1, pp+wh, pp+wh+1] )
            tm = trimesh.Trimesh(vertices=cloth, faces=faces)
            # Handle file naming, with protection vs duplicate objects.
            date = '{}'.format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
            tm_path = join(bhead, 'gym-cloth-{}-rank-{}'.format(self._logger_idx, date))
            duplicates = sum([x for x in os.listdir(bhead) if tm_path in x])
            tm_path = '{}_c{}.obj'.format(tm_path, str(duplicates+1).zfill(2))
            tm.export(tm_path)

            # Step 2: call blender to get image representation.
            #subprocess.call(['blender', '--background', '--python',
            #        'blender_render/get_image_rep.py', '--', tm_path])


            #Temporary fix (giving absolute path to blender 2.8)  
            subprocess.call(['/Users/adivganapathi/Downloads/blender/blender.app/Contents/MacOS/./blender', '--background', '--python',
                    'blender_render/get_image_rep.py', '--', tm_path])


            # Step 3: load image from directory saved by blender.
            blender_path = tm_path.replace('.obj','.png')
            img = cv2.imread(blender_path)
            assert img.shape == (480, 640, 3), img.shape
            img = cv2.resize(img, dsize=(320, 240)) # x,y order is flipped :(
            # Debugging for now to see what smaller images look like.
            cv2.imwrite(tm_path.replace('.obj','_small.png'), img)

            # Step 4: remaining book-keeping, if any?
            return img

        else:
            raise ValueError(self._obs_type)

    #Adi: Added this method to make compatible with Rainbow DDPG
    def _get_obs(self):
        return self.state


    def seed(self, seed=None):
        """Apply the env seed.

        See, for example:
        https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
        We follow a similar convention by using an `np_random` object.
        """
        self.np_random, seed = seeding.np_random(seed)
        self.logger.debug("Just re-seeded env to: {}".format(seed))
        return [seed]

    #Adi: Adding this method which is the exact same as save state but with a different name so that it can be used with Rainbow DDPG.
    def store_state(self, cloth_file):
        """Save cloth.pts as a .pkl file.

        Be sure to supply a full path. Otherwise it saves under the `build/`
        directory somewhere.
        """
        with open(cloth_file, 'wb') as fh:
            pickle.dump({"pts": self.cloth.pts, "springs": self.cloth.springs}, fh)
    

    def save_state(self, cloth_file):
        """Save cloth.pts as a .pkl file.

        Be sure to supply a full path. Otherwise it saves under the `build/`
        directory somewhere.
        """
        with open(cloth_file, 'wb') as fh:
            pickle.dump({"pts": self.cloth.pts, "springs": self.cloth.springs}, fh)

    def _pull(self, i, iters_pull, x_diag_r, y_diag_r):
        """Actually perform pulling, assuming length/angle actions.

        There are two cases when the pull should be stable: after pulling up,
        and after pulling in the plane with a fixed z height.
        """
        if i < self.iters_up:
            self.gripper.adjust(x=0.0, y=0.0, z=0.0025)
        elif i < self.iters_up + self.iters_up_rest:
            pass
        elif i < self.iters_up + self.iters_up_rest + iters_pull:
            self.gripper.adjust(x=x_diag_r, y=y_diag_r, z=0.0)
        elif i < self.iters_up + self.iters_up_rest + iters_pull + self.iters_grip_rest:
            pass
        else:
            self.gripper.release()

    def step(self, action, initialize=False):
        """Execute one action.

        Currently, actions are parameterized as (grasp_point, pull fraction
        length, pull direction).  It will grasp at some target point, and then
        pull in the chosen direction for some number of cloth updates. We have
        rest periods to help with stability.

        If we clipped the action space into [-1,1], (meaning the policy or
        human would output values between [-1,1] for each component) then for
        actions with true ranges of [0,1], divide the original [-1,1] actions
        by two and add 0.5. For angle, multiply by pi.

        Parameters
        ----------
        action: tuple
            Action to be applied this step.
        initialize: bool
            Normally false. If true, that means we're in the initialization step
            from an `env.reset()` call, and so we probably don't want to count
            these 'actions' as part of various statistics we compute.

        Returns
        -------
        Usual (state, reward, done, info) from env steps. Our info contains the
        number of steps called, both for actions and `cloth.update()` calls.
        """
        info = {}
        logger = self.logger
        exit_early = False
        astr = self._act2str(action)

        # Truncate actions according to our bounds, then grip.
        low  = self.action_space.low
        high = self.action_space.high
        if self._delta_actions:
            x_coord, y_coord, delta_x, delta_y = action
            x_coord = max(min(x_coord, high[0]), low[0])
            y_coord = max(min(y_coord, high[1]), low[1])
            delta_x = max(min(delta_x, high[2]), low[2])
            delta_y = max(min(delta_y, high[3]), low[3])
        else:
            x_coord, y_coord, length, radians = action
            x_coord = max(min(x_coord, high[0]), low[0])
            y_coord = max(min(y_coord, high[1]), low[1])
            length  = max(min(length,  high[2]), low[2])
            r_trunc = max(min(radians, high[3]), low[3])

        if self._clip_act_space:
            # If we're here, then all four of these are in the range [-1,1].
            # Due to noise it might originally be out of range, but we truncated.
            x_coord = (x_coord / 2.0) + 0.5
            y_coord = (y_coord / 2.0) + 0.5
            if self._delta_actions:
                pass
            else:
                length = (length / 2.0) + 0.5
                r_trunc = r_trunc * np.pi
        # After this, we assume ranges {[0,1], [0,1],  [0,1], [-pi,pi]}.
        # Or if delta actions,         {[0,1], [0,1], [-1,1],   [-1,1]}.
        # Actually for non deltas, we have slack applied ...

        self.gripper.grab_top(x_coord, y_coord)

        # Determine direction on UNIT CIRCLE, then downscale by reduce_factor to
        # ensure we only move a limited amount each time step, might help physics?
        if self._delta_actions:
            total_length = np.sqrt( (delta_x)**2 + (delta_y)**2 )
            x_dir = delta_x / (total_length + _EPS)
            y_dir = delta_y / (total_length + _EPS)
        else:
            x_dir = np.cos(r_trunc)
            y_dir = np.sin(r_trunc)
        x_dir_r = x_dir * self.reduce_factor
        y_dir_r = y_dir * self.reduce_factor

        # Number of iterations for each stage of the action. Actually, the
        # iteration for the pull can be computed here ahead of time.
        if self._delta_actions:
            ii = 0
            current_l = 0
            while True:
                current_l += np.sqrt( (x_dir_r)**2 + (y_dir_r)**2 )
                if current_l >= total_length:
                    break
                ii += 1
            iters_pull = ii
        else:
            iters_pull = int(self.iters_pull_max * length)

        pull_up    = self.iters_up + self.iters_up_rest
        rest_start = self.iters_up + self.iters_up_rest + iters_pull
        drop_start = self.iters_up + self.iters_up_rest + iters_pull + self.iters_grip_rest
        iterations = self.iters_up + self.iters_up_rest + iters_pull + self.iters_grip_rest + self.iters_rest

        if initialize:
            logger.info("         ======== [during obs.reset()] EXECUTING ACTION: {} ========".format(astr))
        else:
            logger.info("         ======== EXECUTING ACTION: {} ========".format(astr))
        logger.debug("Gripped at ({:.2f}, {:.2f})".format(x_coord, y_coord))
        logger.debug("Grabbed points: {}".format(self.gripper.grabbed_pts))
        logger.debug("Total grabbed: {}".format(len(self.gripper.grabbed_pts)))
        logger.debug("Action maps to {:.3f}, {:.3f}".format(x_dir, y_dir))
        logger.debug("Actual magnitudes: {:.4f}, {:.4f}".format(x_dir_r, y_dir_r))
        logger.debug("itrs up / wait / pull / wait / drop+rest: {}, {}, {}, {}, {}".format(
            self.iters_up, self.iters_up_rest, iters_pull, self.iters_grip_rest, self.iters_rest))

        # Add special (but potentially common) case, if our gripper grips nothing.
        if len(self.gripper.grabbed_pts) == 0:
            logger.info("No points gripped! Exiting action ...")
            exit_early = True
            iterations = 0

        # Add noise (must be tuned!) to starting fold for more initial state diversity.
        if self.use_noise_init and self.num_steps == 0 and not exit_early:
            x_dir_r += (self.np_random.rand()-0.5)*0.002
            y_dir_r += (self.np_random.rand()-0.5)*0.002
            logger.info("Start state, w/noise, direction: {:.4f}, {:.4f})".format(
                    x_dir_r, y_dir_r))

        i = 0
        while i < iterations:
            self._pull(i, iters_pull, x_dir_r, y_dir_r)
            self.cloth.update()
            if not initialize:
                self.num_sim_steps += 1

            # Debugging -- move to separate method?
            if i == self.iters_up:
                logger.debug("i {}, now pulling".format(self.iters_up))
            elif i == drop_start:
                logger.debug("i {}, now dropping".format(drop_start))
            if self.debug_viz and i % 3 == 0:
                self._debug_viz_plots()

            # If we get any tears (pull in a bad direction, etc.), exit.
            if self.cloth.have_tear:
                logger.debug("TEAR, exiting...")
                self.have_tear = True
                break
            i += 1

        if initialize:
            return
        self.num_steps += 1
        rew  = self._reward(action, exit_early)
        term = self._terminal()
        self.logger.info("Reward: {:.4f}. Terminal: {}".format(rew, term))
        self.logger.info("Steps/SimSteps: {}, {}".format(self.num_steps, self.num_sim_steps))
        info = {
            'num_steps': self.num_steps,
            'num_sim_steps': self.num_sim_steps,
            'actual_coverage': self._current_coverage,
            'have_tear': self.have_tear,
            'out_of_bounds': self._out_of_bounds(),
        }
        return self.state, rew, term, info

    def _reward(self, action, exit_early):
        """Reward function.

        First we apply supporting and auxiliary rewards. Then we define actual
        coverage approximations. For now we are keeping the reward as deltas
        and then a large bonus for task completion.
        """
        log = self.logger

        # Keep adjusting this for our negative reward.
        rew = 0

        # Apply one of tear/oob penalities, then one for bad / wasted actions.
        if self.have_tear:
            rew += self._tear_penalty
            log.debug("Apply tear penalty, reward {:.2f}".format(rew))
        #elif self._out_of_bounds():
        #    rew += self._oob_penalty
        #    log.debug("Apply out of bounds penalty, reward {:.2f}".format(rew))

        #if exit_early:
        #    rew += self._nogrip_penalty
        #    log.debug("Apply no grip penalty, reward {:.2f}".format(rew))

        # Apply penalty if action outside the bounds.
        def penalize_action(aval, low, high):
            if low <= aval <= high:
                return 0.0
            if aval < low:
                diff = low - aval
            else:
                diff = aval - high
            pen = - min(diff**2, self._act_pen_limit) * self._act_bound_factor
            assert pen < 0
            return pen

        if not self._clip_act_space:
            x_coord, y_coord, length, radians = action
            low = self.action_space.low
            high = self.action_space.high
            pen0 = penalize_action(x_coord, low[0], high[0])
            pen1 = penalize_action(y_coord, low[1], high[1])
            pen2 = penalize_action(length,  low[2], high[2])
            pen3 = penalize_action(radians, low[3], high[3])
            rew += pen0
            rew += pen1
            rew += pen2
            rew += pen3
            log.debug("After action pen. {:.2f} {:.2f} {:.2f} {:.2f}, rew {:.2f}".
                    format(pen0, pen1, pen2, pen3, rew))

        # Define several coverage formulas. Subtle point about deltas: the
        # input is reward but not the 'auxiliary' bonuses, etc.

        def _save_bad_hull(points, eps=1e-4):
            log.warn("Bad ConvexHull hull! Note, here len(points): {}".format(len(points)))
            pth_head = 'bad_cloth_hulls'
            if not os.path.exists(pth_head):
                os.makedirs(pth_head, exist_ok=True)
            num = len([x for x in os.listdir(pth_head) if 'cloth_' in x])
            if self._logger_idx is not None:
                cloth_file = join(pth_head,
                        'cloth_{}_subrank_{}.pkl'.format(num+1, self._logger_idx))
            else:
                cloth_file = join(pth_head, 'cloth_{}.pkl'.format(num+1))
            self.save_state(cloth_file)

        def compute_height():
            threshold = self.cfg['cloth']['thickness'] / 2.0
            allpts = self.cloth.allpts_arr  # fyi, includes pinned
            z_vals = allpts[:,2]
            num_below_thresh = np.sum(z_vals < threshold)
            fraction = num_below_thresh / float(len(z_vals))
            return fraction

        def compute_variance():
            allpts = self.cloth.allpts_arr
            z_vals = allpts[:,2]
            variance = np.var(z_vals)
            if variance < 0.000001: # handle asymptotic behavior
                return 1000
            else:
                return 0.001 / variance

        def compute_coverage():
            points = np.array([[min(max(p.x,0),1), min(max(p.y,0),1)] for p in self.cloth.pts])
            try:
                # In 2D, this actually returns *AREA* (hull.area returns perimeter)
                hull = ConvexHull(points)
                coverage = hull.volume
            except scipy.spatial.qhull.QhullError as e:
                logging.exception(e)
                _save_bad_hull(points)
                coverage = 0
            return coverage

        def compute_delta(reward):
            diff = reward - self._prev_reward
            self._prev_reward = reward
            return diff

        # Huge reward if we've succeeded in coverage.  This is where we assign
        # to _current_coverage, so be careful if this is what we want, e.g., if
        # we've torn the cloth or are out of bounds, it's not updated. Will
        # need to debug check this when we get to achieving this ...
        self._current_coverage = compute_coverage()
        if self._current_coverage > _REWARD_THRESHOLDS['coverage']:
            #rew += self._cover_success
            log.debug("Success in coverage!! reward {:.2f}".format(rew))

        # Small negative living reward.
        #rew += self._neg_living_rew
        #log.debug("After small living penalty, reward {:.2f}".format(rew))

        if self.reward_type == 'coverage':
            # Actual coverage of the cloth on the XY-plane.
            rew += compute_coverage()
        elif self.reward_type == 'coverage-delta':
            # * difference * in coverage on the XY-plane.
            rew += compute_delta(compute_coverage())
        elif self.reward_type == 'height':
            # Proportion of points that are below thickness-dependent threshold.
            # This is an APPROXIMATION of coverage.
            rew += compute_height()
        elif self.reward_type == 'height-delta':
            # * difference * in the proportion of points below threshold.
            rew += compute_delta(compute_height())
        elif self.reward_type == 'variance':
            # 1/variance in Z-coordinate (to punish high variance)
            rew += compute_variance()
        elif self.reward_type == 'variance-delta':
            # * difference * in 1/variance.
            rew += compute_delta(compute_variance())
        elif self.reward_type == 'sparse1':
            # Return high reward only if we have sufficient coverage
            if compute_coverage() > _REWARD_THRESHOLDS['coverage']:
                rew += 1000
            else:
                rew += 0
        elif self.reward_type == 'folding-number':
            # TODO
            raise NotImplementedError()
        else:
            raise ValueError(self.reward_type)

        log.debug("coverage {:.2f}, reward at end {:.2f}".format(self._current_coverage, rew))
        return rew

    def _terminal(self):
        """Detect if we're done with an episode, for any reason.

        First we detect for (a) exceeding max steps, (b) tearing, (c) out of
        bounds. Then we check if we have sufficiently covered the plane.
        """
        done = False

        if self.num_steps >= self.max_actions:
            self.logger.info("num_steps {} >= max_actions {}, hence done".format(
                    self.num_steps, self.max_actions))
            done = True
        elif self.have_tear:
            self.logger.info("A \'tear\' exists, hence done")
            done = True
        elif self._out_of_bounds():
            self.logger.info("Went out of bounds, hence done")
            done = True

        # Assumes _current_coverage set in _reward() call before _terminal().
        # TODO later, fix to handle other _current_coverage cases
        #rew_thresh = _REWARD_THRESHOLDS[self.reward_type]
        #if self._current_coverage > rew_thresh:
        #    self.logger.info("Cloth is sufficiently smooth, {:.3f} exceeds "
        #        "threshold {:.3f} hence done".format(self._prev_reward, rew_thresh))
        #    done = True

        if done and self.render_gl:
            #self.render_proc.terminate()
            self.cloth.stop_render()
            #self.render_proc = None
        return done

    def reset(self):
        """Must call each time we start a new episode.

        Initializes to a new state, depending on the init 'type' in the config.

        `self.num_steps`: number of actions or timesteps in an episode.
        `self.num_sim_steps`: number of times we call `cloth.update()`.

        The above don't count any of the 'initialization' actions or steps --
        only those in the actual episode.

        Parameters
        ----------
        state: {"pts": [list of Points], "springs": [list of Springs]}
            If specified, load the cloth with this specific state and skip initialization.
        """
        reset_start = time.time()
        logger = self.logger
        cfg = self.cfg
        if self._start_state:
            self.cloth = cloth = Cloth(fname=self.cfg_file,
                                       render=self.render_gl,
                                       random_state=self.np_random,
                                       render_port=self.render_port,
                                       state=copy.deepcopy(self._start_state))
        else:
            self.cloth = cloth = Cloth(fname=self.cfg_file,
                                       render=self.render_gl,
                                       random_state=self.np_random,
                                       render_port=self.render_port)
        assert len(cloth.pts) == self.num_points, \
                "{} vs {}".format(len(cloth.pts), self.num_points)
        assert cloth.bounds[0] == self.bounds[0]
        assert cloth.bounds[1] == self.bounds[1]
        assert cloth.bounds[2] == self.bounds[2]
        self.gripper = gripper = Gripper(cloth, self.grip_radius,
                self.cfg['cloth']['height'], self.cfg['cloth']['thickness'])
        self.num_steps = 0
        self.num_sim_steps = 0
        self.have_tear = False

        if self.debug_viz:
            self.logger.info("Note: we set our config to visualize the init. We"
                    " will now play a video ...")
            nrows, ncols = 1, 2
            self.plt = plt
            self.debug_fig = plt.figure(figsize=(12*ncols,12*nrows))
            self.debug_ax1 = self.debug_fig.add_subplot(1, 2, 1)
            self.debug_ax2 = self.debug_fig.add_subplot(1, 2, 2, projection='3d')
            self.debug_ax2.view_init(elev=5., azim=-50.)
            self.plt.ion()
            self.plt.tight_layout()

        if self._start_state:
            pass
        elif cfg['init']['type'] == 'single-corner-fold':
            assert not self._delta_actions
            # Execute a long diagonal fold to get non-flat starting state.
            fraction = 0.4
            act_dir = 0.0
            action = (self.start_x, self.start_y, fraction, act_dir)
            self.step(action, initialize=True)
        elif cfg['init']['type'] == 'drop-vertical':
            for i in range(1500):
                self.cloth.update()
                if self.debug_viz and i % 3 == 0:
                    self._debug_viz_plots()
        elif cfg['init']['type'] == 'bed-like':
            assert not self._delta_actions
            # Drop, then pull towards east.
            for i in range(1500):
                self.cloth.update()
            logger.debug("Cloth settled, but now let's apply actions.")

            # Sometimes it might be preferable to avoid highest points that are
            # near blanket corners (because one long pull covers the other corner).
            # Option 1
            #highest_point = max(self.cloth.pts, key=lambda pt: pt.z)
            # Option 2
            high_points = sorted(self.cloth.pts, key=lambda pt: pt.z, reverse=True)
            highest_point = None
            for point in high_points:
                if 0.35 <= point.y <= 0.65:
                    highest_point = point
                    break

            length = 1.00
            angle = np.random.uniform(low=-0.1, high=0.1)
            action = (highest_point.x, highest_point.y, length, angle)
            self.step(action, initialize=True)
            # A little extra to get it stable for the init position.
            # Ideally we have a method that automatically detects if the cloth
            # is stable, roughly speaking.
            logger.debug("Let's continue simulating just for initialization.")
            for i in range(750):
                self.cloth.update()
        elif cfg['init']['type'] == 'easy-start':
            # Try some relatively easy starting configurations for coverage.
            for i in range(1500):
                self.cloth.update()
            logger.debug("Cloth settled, but now let's apply actions.")

            # TODO put this somewhere else? Getting cluttered...
            p0 = self.cloth.pts[-25]
            if self._delta_actions:
                dx0 = np.random.uniform(0.60, 0.70)
                dy0 = np.random.uniform(0.0, 0.2)
                action0 = (p0.x, p0.y, dx0, dy0)
                if self._clip_act_space:
                    action0 = ((p0.x-0.5)*2, (p0.y-0.5)*2, dx0, dy0)
            else:
                length0 = np.random.uniform(0.60, 0.70)
                angle0  = np.random.uniform(low=-0.1, high=0.1)
                action0 = (p0.x, p0.y, length0, angle0)
                if self._clip_act_space:
                    action0 = ((action0[0] - 0.5) * 2,
                               (action0[1] - 0.5) * 2,
                               (action0[2] - 0.5) * 2,
                               (action0[3] / np.pi))
            self.step(action0, initialize=True)

            p1 = self.cloth.pts[-1]
            if self._delta_actions:
                dx1 = np.random.uniform(0.40, 0.45)
                dy1 = np.random.uniform(-0.1, 0.1)
                action1 = (p1.x, p1.y, dx1, dy1)
                if self._clip_act_space:
                    action1 = ((p1.x-0.5)*2, (p1.y-0.5)*2, dx1, dy1)
            else:
                length1 = np.random.uniform(0.45, 0.50)
                angle1  = np.random.uniform(low=-0.1, high=0.1)
                action1 = (p1.x, p1.y, length1, angle1)
                if self._clip_act_space:
                    action1 = ((action1[0] - 0.5) * 2,
                               (action1[1] - 0.5) * 2,
                               (action1[2] - 0.5) * 2,
                               (action1[3] / np.pi))
            self.step(action1, initialize=True)

            p2 = self.cloth.pts[50]
            if self._delta_actions:
                dx2 = np.random.uniform(0.25, 0.40)
                dy2 = np.random.uniform(0.0, 0.3)
                action2 = (p2.x, p2.y, dx2, dy2)
                if self._clip_act_space:
                    action2 = ((p2.x-0.5)*2, (p2.y-0.5)*2, dx2, dy2)
            else:
                length2 = np.random.uniform(0.3, 0.5)
                angle2  = np.random.uniform(low=0.1, high=0.4)
                action2 = (p2.x, p2.y, length2, angle2)
                if self._clip_act_space:
                    action2 = ((action2[0] - 0.5) * 2,
                               (action2[1] - 0.5) * 2,
                               (action2[2] - 0.5) * 2,
                               (action2[3] / np.pi))
            self.step(action2, initialize=True)

            logger.debug("Let's continue simulating just for initialization.")
            for i in range(750):
                self.cloth.update()
            logger.debug("COVERAGE: {:.2f}".format(self._compute_coverage()))
        else:
            raise ValueError(cfg['init']['type'])

        reset_time = (time.time() - reset_start) / 60.0
        logger.debug("Done with initial state, {:.2f} minutes".format(reset_time))

        # Adding to ensure prev_reward is set correctly after init, if we are
        # using deltas. Assign here because it's after we load/init the cloth.
        # Assumes we are using the coverage and not height or variance,
        # otherwise prev_reward isn't correct.
        self._prev_reward = self._compute_coverage()

        obs = np.array(self.state)
        return obs

    def get_random_action(self, atype='over_xy_plane'):
        """Retrieves random action.

        One way is to use the usual sample method from gym action spaces. Since
        we set the cloth plane to be in the range (0,1) in the x and y
        directions by default, we will only sample points over that range. This
        may or may not be desirable; we will sometimes pick points that don't
        touch any part of the cloth, in which case we just do a 'NO-OP'.

        The other option would be to sample any point that touches the cloth, by
        randomly picking a point from the cloth mesh and then extracting its x
        and y. We thus always touch something, via the 'naive cylinder' method.
        """
        if atype == 'over_xy_plane':
            return self.action_space.sample()
        elif atype == 'touch_cloth':
            assert not self._delta_actions
            pt = self.cloth.pts[ self.np_random.randint(self.num_points) ]
            length = self.np_random.uniform(low=0, high=1)
            angle = self.np_random.uniform(low=-np.pi, high=np.pi)
            action = (pt.x, pt.y, length, angle)
            if self._clip_act_space:
                action = ((pt.x - 0.5) * 2,
                          (pt.y - 0.5) * 2,
                          (length - 0.5) * 2,
                          angle / np.pi)
            return action
        else:
            raise ValueError(atype)

    def _out_of_bounds(self):
        """Detect if we're out of bounds, e.g., to stop an action.

        Currently, bounds are [0,1]. We add some slack for x/y bounds to
        represent cloth that drapes off the edge of the bed.  We should not be
        able to grasp these points, however.
        """
        pts = self.cloth.allpts_arr
        ptsx = pts[:,0]
        ptsy = pts[:,1]
        ptsz = pts[:,2]
        cond1 = np.max(ptsx) >= self.cloth.bounds[0] + self._slack
        cond2 = np.min(ptsx) < - self._slack
        cond3 = np.max(ptsy) >= self.cloth.bounds[1] + self._slack
        cond4 = np.min(ptsy) < - self._slack
        cond5 = np.max(ptsz) >= self.cloth.bounds[2]
        cond6 = np.min(ptsz) < 0
        outb = (cond1 or cond2 or cond3 or cond4 or cond5 or cond6)
        if outb:
           self.logger.debug("np.max(ptsx): {:.4f},  cond {}".format(np.max(ptsx), cond1))
           self.logger.debug("np.min(ptsx): {:.4f},  cond {}".format(np.min(ptsx), cond2))
           self.logger.debug("np.max(ptsy): {:.4f},  cond {}".format(np.max(ptsy), cond3))
           self.logger.debug("np.min(ptsy): {:.4f},  cond {}".format(np.min(ptsy), cond4))
           self.logger.debug("np.max(ptsz): {:.4f},  cond {}".format(np.max(ptsz), cond5))
           self.logger.debug("np.min(ptsz): {:.4f},  cond {}".format(np.min(ptsz), cond6))
        return outb

    def render(self, filepath, mode='human', close=False):
        """Much subject to change.

        If mode != 'matplotlib', spawn a child process rendering the cloth.
        As a result, you only need to call this once rather than every time
        step. The process is terminated with terminal(), so you must call
        it again after each episode, before calling reset().

        You will have to pass in the renderer filepath to this program, as the
        package will be unable to find it. To get the filepath from, for example,
        gym-cloth/examples/[script_name].py, run

        >>> this_dir = os.path.dirname(os.path.realpath(__file__))
        >>> filepath = os.path.join(this_dir, "../render/build")
        """
        if mode == 'matplotlib':
            self._debug_viz_plots()
        elif self.render_gl and not self.render_proc:
            owd = os.getcwd()
            os.chdir(filepath)
            dev_null = open('/dev/null','w')
            self.render_proc = subprocess.Popen(["./clothsim"], stdout=dev_null, stderr=dev_null)
            os.chdir(owd)

    # --------------------------------------------------------------------------
    # Random helper methods, debugging, etc.
    # --------------------------------------------------------------------------

    def _compute_coverage(self):
        """I wanted a second way to refer to this method.

        We might just use this from now on.
        """
        points = np.array([[min(max(p.x,0),1), min(max(p.y,0),1)] for p in self.cloth.pts])
        try:
            # In 2D, this actually returns *AREA* (hull.area returns perimeter)
            hull = ConvexHull(points)
            coverage = hull.volume
        except scipy.spatial.qhull.QhullError as e:
            logging.exception(e)
            #_save_bad_hull(points)
            coverage = 0
        return coverage

    def _debug_viz_plots(self):
        """Use `plt.ion()` for interactive plots, requires `plt.pause(...)` later.

        This is for the debugging part of the initialization process. It's not
        currently meant for the actual rendering via `env.render()`.
        """
        plt = self.plt
        ax1 = self.debug_ax1
        ax2 = self.debug_ax2
        eps = 0.05

        ax1.cla()
        ax2.cla()
        pts  = self.cloth.noncolorpts_arr
        cpts = self.cloth.colorpts_arr
        ppts = self.cloth.pinnedpts_arr
        if len(pts) > 0:
            ax1.scatter(pts[:,0], pts[:,1], c='g')
            ax2.scatter(pts[:,0], pts[:,1], pts[:,2], c='g')
        if len(cpts) > 0:
            ax1.scatter(cpts[:,0], cpts[:,1], c='b')
            ax2.scatter(cpts[:,0], cpts[:,1], cpts[:,2], c='b')
        if len(ppts) > 0:
            ax1.scatter(ppts[:,0], ppts[:,1], c='darkred')
            ax2.scatter(ppts[:,0], ppts[:,1], ppts[:,2], c='darkred')
        ax1.set_xlim([0-eps, 1+eps])
        ax1.set_ylim([0-eps, 1+eps])
        ax2.set_xlim([0, 1])
        ax2.set_ylim([0, 1])
        ax2.set_zlim([0, 1])
        plt.pause(0.0001)

    def _save_matplotlib_img(self, target_dir=None):
        """Save matplotlib image into a target directory.
        """
        plt = self.plt
        if target_dir is None:
            target_dir = (self.fname_log).replace('.log','.png')
        print("Note: saving matplotlib img of env at {}".format(target_dir))
        plt.savefig(target_dir)

    def _setup_logger(self):
        """Set up the logger (and also save the config w/similar name).

        If you create a new instance of the environment class in the same
        program, you will get duplicate logging messages. We should figure out a
        way to fix that in case we want to scale up to multiple environments.
        """
        cfg = self.cfg
        dstr = '-{}.log'.format(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'))
        filename = (cfg['log']['file']).replace('.log',dstr)
        if self._logger_idx is not None:
            filename = filename.replace('.log',
                    '_rank_{}.log'.format(str(self._logger_idx).zfill(2)))
        logging.basicConfig(
                level=_logging_setup_table[cfg['log']['level']],
                filename=filename,
                filemode='w')

        # Define a Handler which writes messages to the sys.stderr
        console = logging.StreamHandler()
        console.setLevel(_logging_setup_table[cfg['log']['level']])

        # Set a format which is simpler for console use
        formatter = logging.Formatter('%(asctime)s %(name)-12s: %(levelname)-8s '
                                      '%(message)s', datefmt='%m-%d %H:%M:%S')

        # Tell the handler to use this format, and add handler to root logger
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)
        if self._logger_idx is not None:
            self.logger = logging.getLogger("cloth_env_{}".format(self._logger_idx))
        else:
            self.logger = logging.getLogger("cloth_env")

        # Finally, save config file so we can exactly reproduce parameters.
        json_str = filename.replace('.log','.json')
        with open(json_str, 'w') as fh:
            json.dump(cfg, fh, indent=4, sort_keys=True)
        self.fname_log = filename
        self.fname_json = json_str

    def _act2str(self, action):
        """Turn an action into something more human-readable.
        """
        if self._delta_actions:
            x, y, dx, dy = action
            astr = "({:.2f}, {:.2f}), deltax {:.2f}, deltay {:.2f}".format(
                    x, y, float(dx), float(dy))
        else:
            x, y, length, direction = action
            if self._clip_act_space:
                astr = "({:.2f}, {:.2f}), length {:.2f}, angle {:.2f}".format(
                    x, y, float(length), float(direction))
                astr += "  Re-scaled: ({:.2f}, {:.2f}), {:.2f}, {:.2f}".format(
                    (x/2)+0.5, (y/2)+0.5, (length/2)+0.5, direction*np.pi)
            else:
                astr = "({:.2f}, {:.2f}), length {:.2f}, angle {:.2f}".format(
                    x, y, float(length), float(direction))
        return astr
