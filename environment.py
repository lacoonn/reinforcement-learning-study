from board import Board
from qlearning_player import OmokEnvironment, X, W1, b1, input_layer, W2, b2, hidden_layer, W3, b3, output_layer, Y, cost, optimizer
from minimax_player import MinimaxPlayer
import tensorflow as tf
import numpy as np
import random
import math
import os
import sys
import time

class Environment(object):
	def __init__(self, player1, player2):
		self.player1 = player1
		self.player2 = player2
		self.board = Board()

	def getState(self):
		return np.reshape(self.board.data, (1, self.board.size * self.board.size))

	def act(self, player, action):
		'''
		act(currentPlayer, action)
			currentPlayer : 1(black) or 2(white)
			action : DQN에서 나온 결과(0 ~ grid * grid - 1 사이의 정수)

		return nextState, reward, gameOver
			nextState : 돌을 놓은 후의 currentState
			reward : 승리했다면 1, 아니면 0
			gameOver : 게임오버이면 True, 아니면 False
		'''
		x = int(action / self.board.size)
		y = action % self.board.size
		self.board.put_value(x, y)


	#------------------------------------------------------------
	# minimax vs minimax
	#------------------------------------------------------------
	def play_game_with_minimax(self):
		while self.board.is_finished() == False:
			'''
			이 Loop 1번은 1턴을 의미합니다
			'''
			#------------------------------------------------------------
			# 홀수 턴(minimax player) - Black
			#------------------------------------------------------------
			if self.board.turn % 2 == 1:
				self.player1.observe(self.board)
				x, y = self.player1.predict()
				print("Turn : {0}, Color : {1}, Eval : {2}".format(self.board.turn, self.player1.board.MY_COLOR, self.player1.eval))
				self.board.put_value(x, y)
				#print("Turn ", self.board.turn, ", ", self.player1.board.MY_COLOR, " : ", self.player1.eval)
			#------------------------------------------------------------
			# 짝수 턴(minimax player) - White
			#------------------------------------------------------------
			else:
				self.player2.observe(self.board)
				x, y = self.player2.predict()
				print("Turn : {0}, Color : {1}, Eval : {2}".format(self.board.turn, self.player2.board.MY_COLOR, self.player2.eval))                
				self.board.put_value(x, y)
				#print("Turn ", self.board.turn, ", ", self.player2.board.MY_COLOR, " : ", self.player2.eval)
			
			#------------------------------------------------------------
			# 게임 종료 시
			#------------------------------------------------------------
			if self.board.is_finished == True:
				if self.board.turn % 2 == 1:
					self.player1.observe_finish(self.board, True)
					self.player2.observe_finish(self.board, False)
				else:
					self.player1.observe_finish(self.board, False)
					self.player2.observe_finish(self.board, True)
				self.board.draw()

			self.board.draw()
	#------------------------------------------------------------


	#------------------------------------------------------------
	# qlearning vs minimax
	#------------------------------------------------------------
	def play_game_with_qlearning(self):
		STONE_NONE = 0
		STONE_PLAYER1 = 1
		STONE_PLAYER2 = 2

		'''
		# 자동완성을 위한 임시 변수
		self.player1 = OmokEnvironment(10)
		self.player2 = MinimaxPlayer(3)
		'''

		# 텐서플로우 초기화
		sess = tf.Session()
		sess.run(tf.global_variables_initializer())

		# 세이브 설정
		saver = tf.train.Saver()

		# 모델 로드
		if (os.path.isfile(os.getcwd() + "/OmokModel.ckpt.index") == True):
			saver.restore(sess, os.getcwd() + "/OmokModel.ckpt")
			print('saved model is loaded!')


		self.player1.reset()
		gameOver = False
		currentPlayer = STONE_PLAYER1

		#while (self.player1.gameOver != True or player2.is_finished != True):
		while (self.board.finished != True):
			#------------------------------------------------------------
			# 홀수 턴(qlearning player) - Black
			#------------------------------------------------------------
			if self.board.turn % 2 == 1:
				#------------------------------------------------------------
				# 상대 턴이 마친 후의 상태를 업데이트
				#------------------------------------------------------------
				last_action = self.board.last_x * self.board.size + self.board.last_y
				self.player1.updateState(STONE_PLAYER1, last_action)

				action = -9999
				
				currentState = self.player1.getState()
				
				action = self.player1.getAction(sess, currentState)
				
				nextState, reward, gameOver = self.player1.act(currentPlayer, action)

				print("Player1 ==> Turn : {0}, Color : {1}, Eval : {2}".format(
					self.board.turn, "BLACK", self.player2.eval))

				#------------------------------------------------------------
				# 게임의 Board를 업데이트
				#------------------------------------------------------------
				x = int(action / self.board.size)
				y = action % self.board.size
				self.board.put_value(x, y)
			#------------------------------------------------------------
			# 짝수 턴(minimax player) - White
			#------------------------------------------------------------
			else:
				self.player2.observe(self.board)
				
				x, y = self.player2.predict()

				print("Player2 ==> Turn : {0}, Color : {1}, Eval : {2}".format(
					self.board.turn, "WHITE", "???"))

				#------------------------------------------------------------
				# 게임의 Board를 업데이트
				#------------------------------------------------------------
				self.board.put_value(x, y)
			
			self.board.draw()

			if (self.board.finished == True) :
				if (self.board.turn % 2 == 1) :
					winner_color = "BLACK"
				else :
					winner_color = "WHITE"
				print()
				print("GAME OVER")
				print()
				print("WINNER ==> " + winner_color)
				print()
				break
			
			time.sleep(1)

		sess.close()
	#------------------------------------------------------------
#------------------------------------------------------------
