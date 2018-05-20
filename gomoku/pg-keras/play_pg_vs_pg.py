'''
PgAgent vs PgAgent 학습
'''

import copy
import pylab
import time
import numpy as np
from environment import Env
from pg_agent import PgAgent

EPISODES = 1000000

# [START] pg vs pg
if __name__ == "__main__":
	# 설정 파라미터
	MODEL_LOAD = False
	PRINT_FLAG = False
	PLAYER1 = 1
	PLAYER2 = 2
	# 환경과 에이전트 생성
	env = Env()
	player1 = PgAgent()
	player2 = PgAgent()
	# 모델 로드
	if MODEL_LOAD:
		player1.load_model('./save_model/pg1.h5')
		print()
		print("Player1 Model Loaded ...")
		print()
		time.sleep(1)
		player2.load_model('./save_model/pg2.h5')
		print()
		print("Player2 Model Loaded ...")
		print()
		time.sleep(1)
	# 현재 플레이어
	current_player = PLAYER1

	global_step = 0
	scores1, scores2, episodes = [], [], []

	for e in range(EPISODES):
		done = False
		score1 = 0
		score2 = 0
		# env 초기화
		state = env.reset()
		state = np.reshape(state, [1, player1.state_size])

		while not done:
			global_step += 1

			# PLAYER1
			if current_player == PLAYER1:
				# 현재 상태 획득
				state = env.get_state()
				state = np.reshape(state, [1, player1.state_size])
				# 현재 상태에 대한 행동 선택
				action = player1.get_action(state)
				# 선택한 행동으로 환경에서 한 타임스텝 진행 후 샘플 수집
				next_state, reward, done = env.step(PLAYER1, action)
				next_state = np.reshape(next_state, [1, player1.state_size])
				if PRINT_FLAG:
					print("Action : {0} ==> {1}, {2}".format(action, int(action/10), action%10))
					print("Reward : ", reward)
					print("Next State : ", next_state)
					print()
				player1.append_sample(state, action, reward)
				score1 += reward
				state = copy.deepcopy(next_state)

				if PRINT_FLAG:
					# board 출력
					print("Episode : {0}, Turn : {1}, PLAYER1".format(e, env.get_turn()))
					env.draw_board()
					print()
					time.sleep(1)
			# PLAYER2
			else:
				# 현재 상태 획득
				state = env.get_state()
				state = np.reshape(state, [1, player2.state_size])
				# 현재 상태에 대한 행동 선택
				action = player2.get_action(state)
				# 선택한 행동으로 환경에서 한 타임스텝 진행 후 샘플 수집
				next_state, reward, done = env.step(PLAYER1, action)
				next_state = np.reshape(next_state, [1, player2.state_size])
				if PRINT_FLAG:
					print("Action : {0} ==> {1}, {2}".format(action, int(action/10), action%10))
					print("Reward : ", reward)
					print("Next State : ", next_state)
					print()
				player2.append_sample(state, action, reward)
				score2 += reward
				state = copy.deepcopy(next_state)

				if PRINT_FLAG:
					# board 출력
					print("Episode : {0}, Turn : {1}, PLAYER2".format(e, env.get_turn()))
					env.draw_board()
					print()
					time.sleep(1)

			if done:
				'''
				# 패자 reward 추가
				lose_reward = -10.0 / env.get_turn()
				if current_player == PLAYER1:
					player2.lose_game(lose_reward)
					score2 += lose_reward
				else:
					player1.lose_game(lose_reward)
					score1 += lose_reward
				'''
				# 에피소드마다 정책신경망 업데이트
				player1.train_model()
				player2.train_model()
				scores1.append(score1)
				scores2.append(score2)
				episodes.append(e)
				score1 = round(score1, 2)
				score2 = round(score2, 2)
				# 에피소드마다 결과 출력
				print("PLAYER1 ==> episode : {0}, global_step : {1}, end_turn : {2}, score : {3:0.1f}".format(e, global_step, env.get_turn(), score1))
				print("PLAYER2 ==> episode : {0}, global_step : {1}, end_turn : {2}, score : {3:0.1f}".format(e, global_step, env.get_turn(), score2))
				print()
				if PRINT_FLAG:
					time.sleep(1)
			else:
				if current_player == PLAYER1:
					current_player = PLAYER2
				else:
					current_player = PLAYER1
				env.inverse()

		# 기준 에피소드마다 모델 저장
		if e > 0 and e % 100 == 0:
			player1.save_model('./save_model/pg1.h5')
			player2.save_model('./save_model/pg2.h5')
			print("Model is saved !")
			print()
			player1.save_graph(episodes, scores1, 0, 'r', './save_graph/pg1.png')
			player2.save_graph(episodes, scores2, 1, 'b', './save_graph/pg2.png')
			time.sleep(1)
# [END] dqn vs dqn