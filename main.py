import torch
import random
import numpy as np
from gameai import FlappyBird
from agent import Agent
from plot import plot

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent(4, 2)
    game = FlappyBird()
    game_count = 0
    epsilion = 0.1
    while True:
        for i in range(9):
            game.play_step(0)
        # Get Old state
        state_old = game.get_state()

        # get move
        final_move = agent.act(state_old, eps=1)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = game.get_state()

        # train short memory
        agent.step(state_old, final_move, reward, state_new, done)

        if done:
            # Train long memory,plot result
            game.reset()
            game_count += 1
            print(epsilion)
            epsilion = epsilion * 0.99
            if score > reward:  # new High score
                record = score
            print('Game:', game_count, 'Score:', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / game_count
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

train()