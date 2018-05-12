from board import Board
from qlearning_player import OmokEnvironment, ReplayMemory
from minimax_player import MinimaxPlayer
#from qlearning_player import OmokEnvironment, X, W1, b1, input_layer, W2, b2, hidden_layer, W3, b3, output_layer, Y, cost, optimizer, epsilon
import tensorflow as tf
import numpy as np
import random
import math
import os
import sys
import time

#------------------------------------------------------------
# 변수 설정
#------------------------------------------------------------
STONE_NONE = 0
STONE_PLAYER1 = 1
STONE_PLAYER2 = 2
STONE_MAX = 5

gridSize = 10
nbActions = gridSize * gridSize
nbStates = gridSize * gridSize
hiddenSize = 100
maxMemory = 500
batchSize = 50
epoch = 100
#epsilon = 1
epsilon = 0.8
epsilonDiscount = 0.999
epsilonMinimumValue = 0.1
discount = 0.9
learningRate = 0.2
winReward = 1
#------------------------------------------------------------

#------------------------------------------------------------
# 가설 설정
#------------------------------------------------------------
X = tf.placeholder(tf.float32, [None, nbStates])
W1 = tf.Variable(tf.truncated_normal(
	[nbStates, hiddenSize], stddev=1.0 / math.sqrt(float(nbStates))))
b1 = tf.Variable(tf.truncated_normal([hiddenSize], stddev=0.01))
input_layer = tf.nn.relu(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.truncated_normal(
	[hiddenSize, hiddenSize], stddev=1.0 / math.sqrt(float(hiddenSize))))
b2 = tf.Variable(tf.truncated_normal([hiddenSize], stddev=0.01))
hidden_layer = tf.nn.relu(tf.matmul(input_layer, W2) + b2)

W3 = tf.Variable(tf.truncated_normal(
	[hiddenSize, nbActions], stddev=1.0 / math.sqrt(float(hiddenSize))))
b3 = tf.Variable(tf.truncated_normal([nbActions], stddev=0.01))
output_layer = tf.matmul(hidden_layer, W3) + b3

Y = tf.placeholder(tf.float32, [None, nbActions])
cost = tf.reduce_sum(tf.square(Y - output_layer)) / (2 * batchSize)
optimizer = tf.train.GradientDescentOptimizer(learningRate).minimize(cost)
#------------------------------------------------------------

#------------------------------------------------------------
# 랜덤값 구함
#------------------------------------------------------------
def randf(s, e):
	return (float(random.randrange(0, (e - s) * 9999)) / 10000) + s
#------------------------------------------------------------

#------------------------------------------------------------
# qlearning with minimax
#------------------------------------------------------------
def qlearning_with_minimax(player1, player2, memory, sess, saver, epsilon, iteration):
	board = Board()
	player1.reset()
	err = 0
	gameOver = False
	currentPlayer = STONE_PLAYER1

	while (board.finished != True):
		#------------------------------------------------------------
		# 홀수 턴(qlearning player) - Black
		#------------------------------------------------------------
		if board.turn % 2 == 1:
			#------------------------------------------------------------
			# 상대 턴이 마친 후의 상태를 업데이트
			#------------------------------------------------------------
			last_action = board.last_x * board.size + board.last_y
			player1.updateState(STONE_PLAYER1, last_action)

			#--------------------------------
			# 행동 수행
			#--------------------------------
			action = -9999
			
			currentState = player1.getState()
			
			if (randf(0, 1) <= epsilon):
				action = player1.getActionRandom()
			else:
				action = player1.getAction(sess, currentState)

			if (epsilon > epsilonMinimumValue):
				epsilon = epsilon * epsilonDiscount
			
			nextState, reward, gameOver = player1.act(currentPlayer, action)

			'''
			print("Player1 ==> Turn : {0}, Color : {1}, Eval : {2}".format(
				board.turn, "BLACK", "???"))
			'''

			#------------------------------------------------------------
			# 게임의 Board를 업데이트
			#------------------------------------------------------------
			y = int(action / board.size)
			x = action % board.size
			board.put_value(x, y)

			#--------------------------------
			# 학습 수행
			#--------------------------------
			memory.remember(currentState, action, reward, nextState, gameOver)

			inputs, targets = memory.getBatch(output_layer, batchSize, nbActions, nbStates, sess, X)

			_, loss = sess.run([optimizer, cost], feed_dict={X: inputs, Y: targets})
			err = err + loss
		#------------------------------------------------------------
		# 짝수 턴(minimax player) - White
		#------------------------------------------------------------
		else:
			player2.observe(board)
			
			x, y = player2.predict()

			'''
			print("Player2 ==> Turn : {0}, Color : {1}, Eval : {2}".format(
				board.turn, "WHITE", player2.eval))
			'''

			#------------------------------------------------------------
			# 게임의 Board를 업데이트
			#------------------------------------------------------------
			board.put_value(x, y)
		
		#board.draw()

		if (board.finished == True) :
			if (board.turn % 2 == 1) :
				winner = 1
			else :
				winner = 2
			
			save_path = saver.save(sess, os.getcwd() + "/savedata/OmokModel.ckpt")
			
			return winner, save_path
		
		#time.sleep(1)
#------------------------------------------------------------
# 메인 함수
#------------------------------------------------------------
def main(_):
	# 리플레이 메모리 인스턴스 생성
	memory = ReplayMemory(gridSize, maxMemory, discount)

	# 텐서플로우 초기화
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	# 세이브 설정
	saver = tf.train.Saver()

	# 모델 로드
	if (os.path.isfile(os.getcwd() + "/savedata/OmokModel.ckpt.index") == True):
		saver.restore(sess, os.getcwd() + "/savedata/OmokModel.ckpt")
		print('saved model is loaded!')
	else:
		print("Training new model")

	iteration = 0
	player1_wincount = 0
	player2_wincount = 0

	# 게임 플레이
	while(True):
		player1 = OmokEnvironment(10)
		player2 = MinimaxPlayer(1)
		result, save_path = qlearning_with_minimax(player1, player2, memory, sess, saver, epsilon, iteration)

		# 승자 확인
		if (result == 1):
			player1_wincount += 1
		elif (result == 2):
			player2_wincount += 1
		else:
			print("WIN PLAYER ERROR!!!")
		
		iteration += 1

		if (iteration % 10 == 0):
			#------------------------------------------------------------
			# 최근 10판 승률
			#------------------------------------------------------------
			print("Iteration : " + str(iteration) + ", Win Rate : " + str(player1_wincount / (player1_wincount + player2_wincount) * 100))
			print("Saved On " + save_path)
			player1_wincount = 0
			player2_wincount = 0
	
	sess.close()
	



#------------------------------------------------------------
# 메인 함수 실행
#------------------------------------------------------------
if __name__ == '__main__':
	tf.app.run()
#------------------------------------------------------------