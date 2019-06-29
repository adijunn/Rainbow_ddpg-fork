import numpy as np


class DemoPolicy(object):
    def choose_action(self, state):
        return np.clip(self._choose_action(state), -0.5, 0.5)

    def reset(self):
        raise Exception("Not implemented")


class Waypoints(DemoPolicy):
    def __init__(self, waypoints):
        self.waypoints = waypoints
        self.currentWaypoint = 0

    @staticmethod
    def go_to_waypoint(grip_pos, waypoint):
        return (
            np.concatenate((waypoint - grip_pos, [0.4])),
            np.linalg.norm(grip_pos - waypoint) < 0.05,
        )

    def _choose_action(self, state):
        grip_pos = state[0:3]
        action, done = self.go_to_waypoint(
            grip_pos, self.waypoints[min(self.currentWaypoint,
                                         len(self.waypoints) - 1)])
        if done:
            self.currentWaypoint += 1

        return action

    def done(self):
        return self.currentWaypoint >= len(self.waypoints)


#Adi: Adding a demo policy (corner to corner) for the cloth environment
class Corners(DemoPolicy):
    def __init__(self):
        #TODO
        self.t = 0
        #self.policy = None

    def _choose_action(self, state):
        #TODO
        #We are directly being passed the state, so no need to go through env
        #pts = env.cloth.pts
        #Need to replace line below with something else
        pts_unrolled = state (This won't work... need to do inverse of what is being done in the state function in gym_cloth/envs/cloth_env.py)
        lst = []
        lst_tmp = []
        count = 1
        for pt_u in pts_unrolled:
            lst_tmp.append(pt_u)
            if count == 3:
                lst.append(lst_temp)
                lst_tmp = []
                count = 1
        pts = np.array(lst)
                
        RAD_TO_DEG = 180 / np.pi

        # When computing angles, need to use x, y, not cx, cy.
        def _get_xy_vals(pt, targx, targy):
            #x, y = pt.x, pt.y
            #pts is defined a little differently now, so this is how we get the x and y coordinates
            x, y = pt[0], pt[1]
            cx = (x - 0.5) * 2.0
            cy = (y - 0.5) * 2.0
            dx = targx - x
            dy = targy - y
            return (x, y, cx, cy, dx, dy)

        if self.t % 4 == 0:
            # Upper right.
            x, y, cx, cy, dx, dy = _get_xy_vals(pts[-1], targx=1, targy=1)
        elif self.t % 4 == 1:
            # Lower right.
            x, y, cx, cy, dx, dy = _get_xy_vals(pts[-25], targx=1, targy=0)
        elif self.t % 4 == 2:
            # Lower left.
            x, y, cx, cy, dx, dy = _get_xy_vals(pts[0], targx=0, targy=0)
        elif self.t % 4 == 3:
            # Upper left.
            x, y, cx, cy, dx, dy = _get_xy_vals(pts[24], targx=0, targy=1)

        #if cfg['env']['clip_act_space']:
            #action = (cx, cy, dx, dy)
        #else:
            #action = (x, y, dx, dy)

        #Assume we are always clipping the action space for now (I don't want to have to pass in cfg to _choose_action)
        action = (cx, cy, dx, dy)

        
        self.t += 1
        return action
        

    def reset(self):
        #TODO (Do we need this method?)
        self.t = 0

    


class Pusher(DemoPolicy):
    def __init__(self):
        self.policy = None

    def _choose_action(self, state):
        if not self.policy:
            goal_pos = state[3:6]
            goal_pos[2] = 0.03
            object_pos = state[8:11]

            object_rel = object_pos - goal_pos
            behind_obj = object_pos + object_rel / \
                np.linalg.norm(object_rel) * 0.06
            behind_obj[2] = 0.03
            waypoints = [np.concatenate([behind_obj[:2], [0.2]]), behind_obj, goal_pos]

            self.policy = Waypoints(waypoints)
        action = self.policy._choose_action(state)

        if self.policy.done():
            self.policy = None
        # action += np.random.normal([0,0,0,0], 0.15)
        return action * 2

    def reset(self):
        self.policy = None


class ArmData(object):
    def __init__(self, data):
        assert data.shape == (13, )
        self.grip_pos, self.grip_velp, self.gripper_state, self.isGrasping, self.goalPosition, self.goalGripper = (
            data[0:3],
            data[3:6],
            data[6:8],
            data[8],
            data[9:12],
            data[12],
        )


policies = {
    "pusher": Pusher,
    "None": None,
}
