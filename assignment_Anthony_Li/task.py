import math
import numpy as np

JOINT_NAMES = ('arm_joint', 'pendulum_joint')
OBSERVATION_CONTRACT = 'v1:sin(theta),cos(theta),sin(alpha),cos(alpha),theta_dot/10,alpha_dot/10'


def observation(state):
    theta, alpha, td, ad = state
    return np.array([math.sin(theta), math.cos(theta), math.sin(alpha),
                     math.cos(alpha), td/10, ad/10], dtype=np.float32)

def action_to_current(action, p):
    a = np.asarray(action, dtype=float)
    if a.size != 1 or not np.isfinite(a).all():
        raise ValueError('Action must contain exactly one finite number')
    return float(np.clip(a.item(), -1, 1))*p.current_limit

def upright(state):
    alpha = math.atan2(math.sin(state[1]), math.cos(state[1]))
    return abs(alpha) < math.radians(10) and abs(state[2]) < 2 and abs(state[3]) < 1

def reward(state, normalized_action):
    """state = [theta, alpha, theta_dot, alpha_dot] (alpha=0 upright, pi hanging down);
    normalized_action in [-1, 1]. Called every non-terminal step; failure already gives -10.
    Return a float. upright(state) is available for a balance bonus."""
    # TODO: Implement a reward function. 
    theta, alpha, theta_dot, alpha_dot = np.asarray(state, dtype=float)

    action = float(np.asarray(normalized_action).item())
    g = 9.81  # gravity
    r = 0.125

    reward =  -0.01* (g - (r * alpha_dot**2) / 2 - g * np.cos(alpha))**2
    #reward -= 0.01 * alpha**2
    #print(f"Reward1: {reward}")
    reward += 0.001 * (-0.10 * theta**2 - 0.01 * theta_dot**2 - 0.1* alpha**2 - 0.01 * alpha_dot**2 - 0.01 * action**2)
    reward += upright(state) * 0.5  # bonus for being upright


    '''
    Initial idea: reward speed first, then reward no angle
    if abs(alpha) < 3.14/2:
        reward = 0.1 * alpha_dot ** 2
    else:
        reward = -0.1 * alpha ** 2'''


    '''
    Second idea: use one reward function that rewards being upright and penalizes speed and angle deviation
    theta_gain = 0.1
    theta_dot_gain = 0.01
    alpha_gain = 0.1
    alpha_dot_gain = 0.01
    action_gain = 0.01

    reward = (-theta_gain * theta**2 
    - theta_dot_gain * theta_dot**2 
    - alpha_gain * alpha**2 
    - alpha_dot_gain * alpha_dot**2 
    - action_gain * action**2
    ) + upright(state) * 0.5
    '''

    return reward

def failure(state, p):
    return (not np.isfinite(state).all() or abs(state[2]) > p.arm_speed_limit
            or abs(state[3]) > p.pendulum_speed_limit)


def initial_state(rng):
    # Start hanging down, with a small perturbation to break exact symmetry.
    return np.array([rng.uniform(-0.05, 0.05), math.pi+rng.uniform(-0.05, 0.05),
                     rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05)])


def state_from_joint_state(names, positions, velocities):
    """JointState order is not guaranteed. Require both measured velocities."""
    indices = [list(names).index(name) for name in JOINT_NAMES]
    state = np.array([positions[indices[0]], positions[indices[1]],
                      velocities[indices[0]], velocities[indices[1]]], dtype=float)
    if not np.isfinite(state).all():
        raise ValueError('Joint state is non-finite')
    return state
