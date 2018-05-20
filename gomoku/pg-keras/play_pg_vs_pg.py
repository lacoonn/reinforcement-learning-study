'''
PgAgent vs PgAgent 학습
'''

import copy
import pylab
import time
import numpy as np
from environment import Env
from pg_player import PgAgent

EPISODES = 1000000

# [START] pg vs pg
if __name__ == "__main__":
	# 설정 파라미터
	MODEL_LOAD = True
	PRINT_FLAG = False
	BLACK = 1
	WHITE = 2
	# 환경과 에이전트 생성
	env = Env()
	player1 = PgAgent()
	player2 = PgAgent()

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
				print("BLACK ==> episode : {0}, global_step : {1}, end_turn : {2}, score : {3:0.1f}, epsilon : {4:0.1f}".format(e, global_step, env.get_turn(), score1, player1.epsilon))
				print("WHITE ==> episode : {0}, global_step : {1}, end_turn : {2}, score : {3:0.1f}, epsilon : {4:0.1f}".format(e, global_step, env.get_turn(), score2, player2.epsilon))
				print()
				# 그래프 출력
				'''
				scores1.append(score1)
				scores2.append(score2)
				episodes.append(e)
				player1.save_graph(1, episodes, scores1, 'r')
				player2.save_graph(2, episodes, scores2, 'b')
				'''
			else:
				if current_player == BLACK:
					current_player = WHITE
				else:
					current_player = BLACK
				env.inverse()

		# 기준 에피소드마다 모델 저장
		if e % 100 == 0:
			player1.save_model()
			player2.save_model()
			print("Model is saved !")
			print()
			time.sleep(0.5)
# [END] dqn vs dqn