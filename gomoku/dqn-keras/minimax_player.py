'''
minimax + alpha-beta prunning 알고리즘을 사용한 오목 AI
미리 정의된 value function 사용
depth 만큼 미리 예측해서 최종 reward를 산정
reward는 가장 멀리 예측한 board의 상태를 분석하여 얻음
'''

import copy
from board_analyzer import BoardAnalyzer

class MinimaxPlayer(object):
	def __init__(self, depth):
		'''
		얼마나 깊이 볼 것인가!
		'''
		self.DEPTH = depth
		# 마지막으로 선택한 x, y 좌표
		self.x = -1
		self.y = -1
		# 마지막으로 얻은 eval 점수
		self.eval = 0
		

	def observe(self, board):
		# Board 클래스를 알고리즘을 사용하는데 필요한 BoardAnalyzer 클래스로 변환
		self.board = BoardAnalyzer()
		self.board.analyze(board)

	def obsesrve_finish(self, board, is_winner):
		pass

	def predict(self):
		if (self.board.get_turn() <= 1):
			self.x = self.board.get_size() // 2
			self.y = self.board.get_size() // 2
			return self.x, self.y

		alpha = -999999
		beta = 999999

		self.x, self.y, self.eval = self.minimax(self.board, self.DEPTH, alpha, beta, True)

		return self.x, self.y

	def minimax(self, node, depth, alpha, beta, maximizing_player):
		if depth == 0:
			node.evaluate()
			return -1, -1, node.eval

		# Maximizing Player
		if (maximizing_player):
			best_value = -99999999
			best_x = 0
			best_y = 0

			for i in range(node.get_size()):
				for j in range(node.get_size()):
					# 해당 칸이 빈 칸인 경우에만 트리를 생성
					if node.get_value(i, j) == 0 and node.isNear(i, j):
						next_node = copy.deepcopy(node)

						if next_node.put_value(i, j):
							v = next_node.eval
						else:
							_, _, v = self.minimax(next_node, depth - 1, alpha, beta, False)
						
						best_value = self.max_node(best_value, v)
						if best_value == v:
							best_x = i
							best_y = j
						
						# Alpha-Beta Prunning
						alpha = self.max_node(alpha, best_value)
						if beta < alpha:
							return best_x, best_y, best_value
			return best_x, best_y, best_value
		# Minimizing Player
		else:
			best_value = 99999999
			best_x = 0
			best_y = 0

			for i in range(node.get_size()):
				for j in range(node.get_size()):
					# 해당 칸이 빈 칸인 경우에만 트리를 생성
					if node.get_value(i, j) == 0 and node.isNear(i, j):
						next_node = copy.deepcopy(node)

						if next_node.put_value(i, j):
							v = next_node.eval
						else:
							_, _, v = self.minimax(next_node, depth - 1, alpha, beta, False)
						
						best_value = self.min_node(best_value, v)
						if best_value == v:
							best_x = i
							best_y = j

						# Alpha-Beta Prunning
						beta = self.max_node(beta, best_value)
						if beta < alpha:
							return best_x, best_y, best_value
			return best_x, best_y, best_value

	
	def max_node(self, a, b):
		if a > b:
			return a
		else:
			return b

	def min_node(self, a, b):
		if a < b:
			return a
		else:
			return b