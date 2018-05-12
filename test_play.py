from environment import Environment
from minimax_player import MinimaxPlayer
from qlearning_player import OmokEnvironment, X, W1, b1, input_layer, W2, b2, hidden_layer, W3, b3, output_layer, Y, cost, optimizer
import tensorflow as tf
import numpy as np
import random
import math
import os
import sys
import time


'''
def minimax_minimax():
	player1 = MinimaxPlayer(2)
	player2 = MinimaxPlayer(2)
	env = Environment(player1, player2)
	env.play_game_with_minimax()
'''

def qlearning_minimax():
	player1 = OmokEnvironment(10)
	player2 = MinimaxPlayer(2)
	env = Environment(player1, player2)
	env.qlearning_vs_minimax()

'''
def qlearning_qlearning():
	player1 = OmokEnvironment(10)
	player2 = OmokEnvironment(10)
	env = Environment(player1, player2)
	env.qlearning_vs_qlearning()
'''

def main(_):
	qlearning_minimax()



if __name__ == '__main__':
	'''
	# 10 * 1 행렬 = 길이 10 행렬
	state1 = np.zeros(10, dtype=np.uint8)
	print(state1)
	# 1 * 10 행렬
	state2 = np.reshape(state1, (1, 10))
	print(state2)
	'''
	tf.app.run()
