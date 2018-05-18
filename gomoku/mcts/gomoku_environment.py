'''
Q-Learning vs Minimax를 통해서 학습
'''

from gomoku_board import Board
from board_analyzer import BoardAnalyzer

import numpy as np
import random
import math
import os
import sys
import time

#------------------------------------------------------------
# Environment
#------------------------------------------------------------
class Env(object):
	def __init__(self):
		self.grid_size = 10
		self.state_size = self.grid_size * self.grid_size
		# board는 object 내부에서 오목의 연산 처리용으로만 사용한다
		self.board = Board(self.grid_size)
		# object의 외부 api는 state를 통해서만 이루어진다
		self.state = np.zeros(self.state_size, dtype=np.uint8)

	#--------------------------------
	# 리셋
	#--------------------------------
	def reset(self):
		self.board = Board(self.grid_size)
		self.state = np.zeros(self.state_size, dtype=np.uint8)
		return self.state

	#--------------------------------
	# 현재 state 구함
	#--------------------------------
	def get_state(self):
		return np.reshape(self.state, (1, self.state_size))

	#--------------------------------
	# board 색을 뒤집음
	#--------------------------------
	def inverse(self):
		BLACK = 1
		WHITE = 2

		self.board.inverse()

		for i in range(self.state_size):
			if(self.state[i] == BLACK):
				self.state[i] = WHITE
			elif(self.state[i] == WHITE):
				self.state[i] = BLACK

	#--------------------------------
	# 현재 board 구함
	#--------------------------------
	def get_board(self):
		return self.board

	#--------------------------------
	# 현재 턴을 구함
	#--------------------------------
	def get_turn(self):
		return self.board.turn

	#--------------------------------
	# state에 action을 적용
	#--------------------------------
	def update_state(self, player, action):
		if self.state[action] == 0:
			# state에 적용
			self.state[action] = player
			# board에 적용
			x = int(action / self.grid_size)
			y = int(action % self.grid_size)
			self.board.put_value(player, x, y)
		# 반환
		return self.state

	#--------------------------------
	# board에 action을 적용
	#--------------------------------
	def update_board(self, player, x, y):
		if self.board.get_value(x, y) == 0:
			# board에 적용
			self.board.put_value(player, x, y)
			# state에 적용
			action = x * self.grid_size + y
			self.state[action] = player
		# 반환
		return self.board

	#------------------------------------------------------------
	# 랜덤값 구함
	#------------------------------------------------------------
	def randf(self, s, e):
		return (float(random.randrange(0, (e - s) * 9999)) / 10000) + s
	#------------------------------------------------------------

	#------------------------------------------------------------
	# board 출력
	#------------------------------------------------------------
	def draw_board(self):
		self.board.draw()
	#------------------------------------------------------------

	#--------------------------------
	# 게임오버 검사
	#--------------------------------
	def is_gameover(self, player):
		if self.board.turn >= self.state_size:
			#return True
			return self.board.finished
		else:
			return self.board.finished

	#--------------------------------
	# action을 실행하고 결과를 반환
	#--------------------------------
	def step(self, player, action):
		'''
		args:
			player
			action
		return:
			next_state : action을 실행한 이후의 state
			reward : action에 대한 보상
			done : 게임 종료 여부
		'''
		x = int(action / self.grid_size)
		y = int(action % self.grid_size)
		# 빈 곳에 돌을 놓으면 정상 실행
		if self.board.get_value(x, y) == 0:
			next_state = self.update_state(player, action)
			done = self.is_gameover(player)
			if done == True:
				reward = 1
			else:
				reward = 0
			'''
			analyzer = BoardAnalyzer()
			reward = analyzer.get_score(self.board, player)
			'''
		# 이미 돌이 있는 곳에 돌을 놓으면 -1 점수를 받고 종료
		else:
			self.board.turn += 1
			next_state = self.state
			done = True
			reward = -1

		return next_state, reward, done