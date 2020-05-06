import pandas as pd
import numpy as np
import math

def MLE_p(neurons, spikes, x, time_bucket_length, poisson):
    # poisson pre-calculated
    max_val = float('-inf')
    max_x = -1
    for x_k in x:
        sum_val = 0
        for i in neurons:
            if (i, x_k) in poisson:
                pm = poisson[i, x_k]
                if pm:
                    sum_val += (spikes[i] * math.log(pm*time_bucket_length)) - (pm*time_bucket_length)
        if sum_val > max_val:
            max_x = x_k
            max_val = sum_val
    return max_x

def location_transition_probability(x_0, x_1, v, K, V, d):
    sigma = K * ((v/V) ** d)
    prob = (1/(sigma * ((2 * math.pi) ** 0.5)))
    prob *= math.e ** (-0.5 * (((x_1 - x_0)/sigma) ** 2))
    if prob == 0:
        # floating point arithmetic limitations cause errors later on when taking logs
        prob = 1e-28
    return prob
    
def get_average_speeds(df, locations):
    average_speed = {}
    no_speed_data = []
    for location in locations:
        at_loc = list(np.where(df.location == location)[0])
        if(len(at_loc) < 2):
            no_speed_data.append(location)
            continue
        speed = 0
        for loc in at_loc:
            if loc == 0:
                speed += abs(df.location.iloc[1] - df.location.iloc[0])
            elif loc == len(df) - 1:
                speed += abs(df.location.iloc[-1] - df.location.iloc[-2])
            else:
                speed += abs(df.location.iloc[loc + 1] - df.location.iloc[loc - 1])
        speed /= len(at_loc)
        average_speed[location] = speed
        if speed == 0:
            no_speed_data.append(location)
    overall_avg_speed = np.mean(list(average_speed.values()))
    for location in no_speed_data:
        average_speed[location] = overall_avg_speed
    return average_speed

def MLE_continuity_constraint(neurons, spikes, average_speed, x, x_last, K, V, d, time_bucket_length, poisson):
    # poisson pre-calculated
    max_val = float('-inf')
    max_x = -1
    for x_k in x:
        sum_val = 0
        for i in neurons:
            if (i, x_k) in poisson:
                pm = poisson[i, x_k]
                if pm:
                    sum_val += (spikes[i] * math.log(pm*time_bucket_length)) - (pm*time_bucket_length)
        transition_prob = location_transition_probability(x_last, x_k, average_speed[x_k], K, V, d)
        sum_val += math.log(transition_prob)
        if sum_val > max_val:
            max_x = x_k
            max_val = sum_val
    return max_x