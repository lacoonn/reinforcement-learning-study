'''
PgAgent vs Minimax 학습 & 대전
'''

import copy
import pylab
import time
import numpy as np
from environment import Env
from pg_player import PgAgent
from minimax_player import MinimaxPlayer

EPISODES = 1000000

# [START] pg vs miniamx
if __name__ == "__main__":
	# 설정 파라미터
	MODEL_LOAD = True
	PRINT_FLAG = False
	BLACK = 1
	WHITE = 2
	DEPTH = 1	
	# 환경과 에이전트 생성
	env = Env()
	player1 = PgAgent()
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