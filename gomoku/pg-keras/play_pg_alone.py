'''
PgAgent가 혼자서 흑백을 번갈아 두면서 학습
'''

import copy
import pylab
import time
import numpy as np
from environment import Env
from pg_agent import PgAgent

EPISODES = 1000000

# [START] pg alone
if __name__ == "__main__":
	# 설정 파라미터
	MODEL_LOAD = True
	PRINT_FLAG = False
	BLACK = 1
	WHITE = 2
	# 환경과 에이전트 생성
	env = Env()
	player = PgAgent()
	# 모델 로드
	if MODEL_LOAD:
		player.load_model()
		time.sleep(1)
	# 현재 플레이어
	current_player = BLACK

	global_step = 0
	scores, episodes = [], []

	for e in range(EPISODES):
		done = False
		score = 0
		# env 초기화
		state = env.reset()
		state = np.reshape(state, [1, player.state_size])

		while not done:
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
				next_state = np.reshape(next_state, [1, player.state_size])
				if PRINT_FLAG:
					print("Action : {0} ==> {1}, {2}".format(action, int(action/10), action%10))
					print("Reward : ", reward)
					print("Next State : ", next_state)
					print()
				player.append_sample(state, action, reward)
				score += reward
				state = copy.deepcopy(next_state)

				if PRINT_FLAG:
					# board 출력
					print("Episode : {0}, Turn : {1}, PLAYER1".format(e, env.get_turn()))
					env.draw_board()
					print()
					time.sleep(1)
			# White
			else:
				# 현재 상태 획득
				state = env.get_state()
				state = np.reshape(state, [1, player.state_size])
				# 현재 상태에 대한 행동 선택
				action = player.get_action(state)
				# 선택한 행동으로 환경에서 한 타임스텝 진행 후 샘플 수집
				next_state, reward, done = env.step(BLACK, action)
				next_state = np.reshape(next_state, [1, player.state_size])
				if PRINT_FLAG:
					print("Action : {0} ==> {1}, {2}".format(action, int(action/10), action%10))
					print("Reward : ", reward)
					print("Next State : ", next_state)
					print()
				player.append_sample(state, action, reward)
				score += reward
				state = copy.deepcopy(next_state)

				if PRINT_FLAG:
					# board 출력
					print("Episode : {0}, Turn : {1}, PLAYER2".format(e, env.get_turn()))
					env.draw_board()
					print()
					time.sleep(1)

			if done:
				# 에피소드마다 정책신경망 업데이트
				player.train_model()
				scores.append(score)
				episodes.append(e)
				score = round(score, 2)
				# 에피소드마다 결과 출력
				print("episode : {0}, global_step : {1}, end_turn : {2}, score : {3:0.1f}".format(e, global_step, env.get_turn(), score))
				print()
				if PRINT_FLAG:
					time.sleep(1)
			else:
				if current_player == BLACK:
					current_player = WHITE
				else:
					current_player = BLACK
				env.inverse()

		# 100 에피소드마다 모델 저장
		if e > 0 and e % 100 == 0:
			player.save_model()
			print("Model Saved ...")
			print()
			player.save_graph(episodes, scores)
			time.sleep(1)
# [END] pg alone