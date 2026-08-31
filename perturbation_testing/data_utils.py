from model_runs import *
import text_utils


def create_x(trajectories):
    x = []
    for trajectory in trajectories:
        if len(trajectory) >= 2:
            x.append([trajectory[0], trajectory[1]])
    return x



def create_y(df):
    return df['sense'].values

