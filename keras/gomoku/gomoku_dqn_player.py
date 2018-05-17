import copy
import pylab
import random
import time
import os
import numpy as np
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from gomoku_environment import Env
from minimax_player import MinimaxPlayer

EPISODES = 100000


# 그리드월드 예제에서의 딥살사 에이전트
# [START] DqnAgent Class
class DqnAgent:
	def __init__(self, model_path, graph_path, epsilon):
		# 에이전트가 가능한 모든 행동 정의
		self.action_space = list(range(100))
		# 상태의 크기와 행동의 크기 정의
		self.action_size = len(self.action_space)
		self.state_size = 100
		self.discount_factor = 0.99
		self.learning_rate = 0.001

		self.epsilon = 1.  # exploration
		self.epsilon_decay = .999999
		self.epsilon_min = 0.1
		self.model = self.build_model()

		self.model_path = model_path
		self.graph_path = graph_path

		print()
		if (os.path.isfile(os.getcwd() + self.model_path) == True):
			self.epsilon = epsilon
			self.model.load_weights(self.model_path)
			print("Saved model is loaded !")
		else:
			print("Train new model !")
		print()
		
		time.sleep(1)

	# 상태가 입력 큐함수가 출력인 인공신경망 생성
	def build_model(self):
		model = Sequential()
		model.add(Dense(200, input_dim=self.state_size, activation='relu'))
		model.add(Dense(200, activation='relu'))
		model.add(Dense(self.action_size, activation='linear'))
		model.summary()
		model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
		return model
	
	# 모델 저장
	def save_model(self):
		self.model.save_weights(self.model_path)

	# 그래프 저장
	def save_graph(self, figure_num, episodes, scores, option):
		pylab.figure(figure_num)
		pylab.plot(episodes, scores, option)
		pylab.savefig(self.graph_path)


	# 입실론 탐욕 방법으로 행동 선택
	def get_action(self, state):
		if np.random.rand() <= self.epsilon:
			# 무작위 행동 반환
			return random.randrange(self.action_size)
		else:
			# 모델로부터 행동 산출
			state = np.float32(state)
			q_values = self.model.predict(state)
			return np.argmax(q_values[0])

	# 모델 학습
	def train_model(self, state, action, reward, next_state, next_action, done):
		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay

		state = np.float32(state)
		next_state = np.float32(next_state)
		target = self.model.predict(state)[0]
		# 살사의 큐함수 업데이트 식
		if done:
			target[action] = reward
		else:
			target[action] = (reward + self.discount_factor * self.model.predict(next_state)[0][next_action])

		# 출력 값 reshape
		target = np.reshape(target, [1, 100])

		# 인공신경망 업데이트
		self.model.fit(state, target, epochs=1, verbose=0)
	# [END] train_model()
# [END] DqnAgent Class

# [START] dqn
def dqn():
	BLACK = 1
	WHITE = 2
	# 환경과 에이전트 생성
	env = Env()
	player = DqnAgent('./save_model/dqn.h5', './save_graph/dqn.png', 1)
	# 현재 플레이어
	current_player = BLACK

	global_step = 0
	scores, episodes = [], []

	for e in range(EPISODES):
		done = False
		score = 0
		state = env.reset()
		state = np.reshape(state, [1, player.state_size])

		while not done:
			# env 초기화
			global_step += 1

			# Black
			if current_player == BLACK:
				# 현재 상태 획득
				state = env.get_state()
				state = np.reshape(state, [1, player.state_size])
				# 현재 상태에 대한 행동 선택
				action = player.get_action(state)
				# 선택한 행동으로 환경에서 한 타임스텝 진행 후 샘플 수집
				next_state, reward, done = env.step(BLACK, action)
				'''
				print("Action : ", action)
				print("Reward : ", reward)
				print("Next State : ", next_state)
				print()
				'''
				next_state = np.reshape(next_state, [1, player.state_size])
				next_action = player.get_action(next_state)
				# 샘플로 모델 학습
				player.train_model(state, action, reward, next_state, next_action, done)
				state = next_state
				score += reward

				state = copy.deepcopy(next_state)

				# board 출력
				#env.draw_board()
				#print()
				#time.sleep(1)
			# White
			else:
				# 현재 상태 획득
				state = env.get_state()
				state = np.reshape(state, [1, player.state_size])
				# 현재 상태에 대한 행동 선택
				action = player.get_action(state)
				# 선택한 행동으로 환경에서 한 타임스텝 진행 후 샘플 수집
				next_state, reward, done = env.step(BLACK, action)
				'''
				print("Action : ", action)
				print("Reward : ", reward)
				print("Next State : ", next_state)
				print()
				'''
				next_state = np.reshape(next_state, [1, player.state_size])
				next_action = player.get_action(next_state)
				# 샘플로 모델 학습
				player.train_model(state, action, reward, next_state, next_action, done)
				state = next_state
				score += reward

				state = copy.deepcopy(next_state)

				# board 출력
				#env.draw_board()
				#print()
				#time.sleep(1)

			if done:
				# 에피소드마다 학습 결과 출력
				scores.append(score)
				episodes.append(e)
				pylab.plot(episodes, scores, 'b')
				pylab.savefig(player.graph_path)
				print("episode : {0}, global_step : {1}, end_turn : {2}, score : {3:0.1f}, epsilon : {4:0.1f}".format(e, global_step, env.get_turn(), score, player.epsilon))
				print()
				#time.sleep(1)
			else:
				if current_player == BLACK:
					current_player = WHITE
				else:
					current_player = BLACK
				env.inverse()

		# 100 에피소드마다 모델 저장
		if e % 25 == 0:
			player.model.save_weights(player.model_path)
			print("Model is saved !")
			print()
			#time.sleep(1)
# [END] dqn

# [START] dqn vs dqn
def dqn_vs_dqn():
	PRINT_FLAG = False
	BLACK = 1
	WHITE = 2
	# 환경과 에이전트 생성
	env = Env()
	player1 = DqnAgent('./save_model/dqn.h5', './save_graph/dqn.png', 1)
	player2 = DqnAgent('./save_model/dqn2.h5', './save_graph/dqn2.png', 1)

	current_player = BLACK
	global_step = 0
	scores1, scores2, episodes = [], [], []

	for e in range(EPISODES):
		done = False
		score1 = 0
		score2 = 0
		state = env.reset()
		state = np.reshape(state, [1, player1.state_size])

		while not done:
			# env 초기화
			global_step += 1

			# 홀수 턴(qlearning player) - Black
			if env.get_turn() % 2 == 1:
				# 현재 상태 획득
				state = env.get_state()
				state = np.reshape(state, [1, player2.state_size])
				# 현재 상태에 대한 행동 선택
				action = player1.get_action(state)
				# 선택한 행동으로 환경에서 한 타임스텝 진행 후 샘플 수집
				next_state, reward, done = env.step(BLACK, action)
				'''
				print("Action : ", action)
				print("Reward : ", reward)
				print("Next State : ", next_state)
				print()
				'''
				next_state = np.reshape(next_state, [1, player1.state_size])
				next_action = player1.get_action(next_state)
				# 샘플로 모델 학습
				player1.train_model(state, action, reward, next_state, next_action, done)
				state = next_state
				score1 += reward

				state = copy.deepcopy(next_state)

				# board 출력
				if PRINT_FLAG:
					print(env.get_turn(), " : player1")
					env.draw_board()
					time.sleep(0.5)

			# 짝수 턴(minimax player) - White
			else:
				# 현재 상태 획득
				state = env.get_state()
				state = np.reshape(state, [1, player2.state_size])
				# 현재 상태에 대한 행동 선택
				action = player2.get_action(state)
				# 선택한 행동으로 환경에서 한 타임스텝 진행 후 샘플 수집
				next_state, reward, done = env.step(BLACK, action)
				'''
				print("Action : ", action)
				print("Reward : ", reward)
				print("Next State : ", next_state)
				print()
				'''
				next_state = np.reshape(next_state, [1, player2.state_size])
				next_action = player2.get_action(next_state)
				# 샘플로 모델 학습
				player2.train_model(state, action, reward, next_state, next_action, done)
				state = next_state
				score2 += reward

				state = copy.deepcopy(next_state)

				# board 출력
				if PRINT_FLAG:
					print(env.get_turn(), " : player2")
					env.draw_board()
					time.sleep(0.5)

			if done:
				# 에피소드마다 학습 결과 출력
				scores1.append(score1)
				scores2.append(score2)
				episodes.append(e)
				player1.save_graph(1, episodes, scores1, 'r')
				player2.save_graph(2, episodes, scores2, 'b')
				#print("episode : ", e, ", score : ", score, ", global_step : ", global_step, ", end_turn : ", env.get_turn(), ", epsilon : ", player1.epsilon)
				print("BLACK ==> episode : {0}, global_step : {1}, end_turn : {2}, score : {3:0.1f}, epsilon : {4:0.1f}".format(e, global_step, env.get_turn(), score1, player1.epsilon))
				print("WHITE ==> episode : {0}, global_step : {1}, end_turn : {2}, score : {3:0.1f}, epsilon : {4:0.1f}".format(e, global_step, env.get_turn(), score2, player2.epsilon))
				print()
				#time.sleep(1)
			else:
				if current_player == BLACK:
					current_player = WHITE
				else:
					current_player = BLACK
				env.inverse()

		# 25 에피소드마다 모델 저장
		if e % 25 == 0:
			player1.save_model()
			player2.save_model()
			print("Model is saved !")
			print()
			time.sleep(1)
# [END] dqn vs dqn

# [START] dqn vs miniamx
def dqn_vs_minimax():
	BLACK = 1
	WHITE = 2
	# 환경과 에이전트 생성
	env = Env()
	player1 = DqnAgent('./save_model/dqn.h5', './save_graph/dqn.png', 1)
	DEPTH = 1
	player2 = MinimaxPlayer(DEPTH)

	global_step = 0
	scores, episodes = [], []

	for e in range(EPISODES):
		done = False
		score = 0
		state = env.reset()
		state = np.reshape(state, [1, player1.state_size])

		while not done:
			# env 초기화
			global_step += 1

			# 홀수 턴(qlearning player) - Black
			if env.get_turn() % 2 == 1:
				# 현재 상태에 대한 행동 선택
				action = player1.get_action(state)
				# 선택한 행동으로 환경에서 한 타임스텝 진행 후 샘플 수집
				next_state, reward, done = env.step(BLACK, action)
				'''
				print("Action : ", action)
				print("Reward : ", reward)
				print("Next State : ", next_state)
				print()
				'''
				next_state = np.reshape(next_state, [1, player1.state_size])
				next_action = player1.get_action(next_state)
				# 샘플로 모델 학습
				player1.train_model(state, action, reward, next_state, next_action, done)
				state = next_state
				score += reward

				state = copy.deepcopy(next_state)

				# board 출력
				#env.draw_board()
				#print()
				#time.sleep(0.5)

			# 짝수 턴(minimax player) - White
			else:
				player2.observe(env.get_board())
			
				x, y = player2.predict()

				'''
				print("Player2 ==> Turn : {0}, Color : {1}, Eval : {2}".format(
					board.turn, "WHITE", player2.eval))
				'''

				# 게임의 Board를 업데이트
				env.update_board(WHITE, x, y)

				# 게임 종료 확인
				done = env.is_gameover(WHITE)

				# board 출력
				#env.draw_board()
				#print()
				#time.sleep(0.5)

			if done:
				# 에피소드마다 학습 결과 출력
				scores.append(score)
				episodes.append(e)
				pylab.plot(episodes, scores, 'b')
				pylab.savefig(player1.graph_path)
				#print("episode : ", e, ", score : ", score, ", global_step : ", global_step, ", end_turn : ", env.get_turn(), ", epsilon : ", player1.epsilon)
				print("episode : {0}, global_step : {1}, end_turn : {2}, score : {3:0.1f}, epsilon : {4:0.1f}".format(e, global_step, env.get_turn(), score, player1.epsilon))
				print()
				#time.sleep(1)

		# 100 에피소드마다 모델 저장
		if e % 25 == 0:
			player1.model.save_weights(player1.model_path)
			print("Model is saved !")
			print()
			#time.sleep(1)
# [END] dqn vs minimax

# [START] main
if __name__ == "__main__":
	#dqn()
	dqn_vs_dqn()
	#dqn_vs_minimax()
# [END] main