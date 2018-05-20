'''
Policy Gradient Player
'''

import copy
import pylab
import numpy as np
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from keras import backend as K

# 오목에서의 Policy Gradient 에이전트
class PgAgent:
	def __init__(self):
		# 가능한 모든 행동 정의
		self.action_space = list(range(100))
		# 상태와 행동의 크기 정의
		self.action_size = len(self.action_space)
		self.state_size = 100
		self.discount_factor = 0.99
		self.learning_rate = 0.001

		self.model = self.build_model()
		self.optimizer = self.build_optimizer()
		self.states, self.actions, self.rewards = [], [], []

	# 상태가 입력, 각 행동의 확률이 출력인 인공신경망 생성
	def build_model(self):
		model = Sequential()
		model.add(Dense(150, input_dim=self.state_size, activation='relu'))
		model.add(Dense(150, activation='relu'))
		model.add(Dense(self.action_size, activation='softmax'))
		model.summary()
		return model

	# 모델 저장
	def save_model(self, filepath = './save_model/pg.h5'):
		self.model.save_weights(filepath)

	# 모델 로드
	def load_model(self, filepath = './save_model/pg.h5'):
		self.model.save_weights(filepath)

	# 그래프 저장
	def save_graph(self, episodes, scores, figure = 0, option = 'b', filepath = './save_graph/pg.png'):
		pylab.figure(figure)
		pylab.plot(episodes, scores, option)
		pylab.savefig(filepath)

	# 정책신경망을 업데이트 하기 위한 오류함수와 훈련함수의 생성
	def build_optimizer(self):
		action = K.placeholder(shape=[None, self.action_size])
		discounted_rewards = K.placeholder(shape=[None, ])

		# 크로스 엔트로피 오류함수 계산
		action_prob = K.sum(action * self.model.output, axis=1)
		cross_entropy = K.log(action_prob) * discounted_rewards
		loss = -K.sum(cross_entropy)

		# 정책신경망을 업데이트하는 훈련함수 생성
		optimizer = Adam(lr=self.learning_rate)
		updates = optimizer.get_updates(self.model.trainable_weights,[],
										loss)
		train = K.function([self.model.input, action, discounted_rewards], [],
						   updates=updates)

		return train

	# 정책신경망으로 행동 선택
	def get_action(self, state):
		temp = self.model.predict(state)
		if False:
			print(state.shape)
			print(state)
			print(temp.shape)
			print(temp)
			print()
		policy = temp[0]
		return np.random.choice(self.action_size, 1, p=policy)[0]

	# 반환값 계산
	def discount_rewards(self, rewards):
		discounted_rewards = np.zeros_like(rewards)
		running_add = 0
		for t in reversed(range(0, len(rewards))):
			running_add = running_add * self.discount_factor + rewards[t]
			discounted_rewards[t] = running_add
		return discounted_rewards

	# 한 에피소드 동안의 상태, 행동, 보상을 저장
	def append_sample(self, state, action, reward):
		self.states.append(state[0])
		self.rewards.append(reward)
		act = np.zeros(self.action_size)
		act[action] = 1
		self.actions.append(act)

	# 정책신경망 업데이트
	def train_model(self):
		discounted_rewards = np.float32(self.discount_rewards(self.rewards))
		discounted_rewards -= np.mean(discounted_rewards)
		discounted_rewards /= np.std(discounted_rewards)

		self.optimizer([self.states, self.actions, discounted_rewards])
		self.states, self.actions, self.rewards = [], [], []

'''
if __name__ == "__main__":
	# 환경과 에이전트의 생성
	env = Env()
	agent = PgAgent()

	global_step = 0
	scores, episodes = [], []

	for e in range(EPISODES):
		done = False
		score = 0
		# env 초기화
		state = env.reset()
		state = np.reshape(state, [1, 15])

		while not done:
			global_step += 1
			# 현재 상태에 대한 행동 선택
			action = agent.get_action(state)
			# 선택한 행동으로 환경에서 한 타임스탭 진행 후 샘플 수집
			next_state, reward, done = env.step(action)
			next_state = np.reshape(next_state, [1, 15])
			#print("Action : ", action)
			#print("Reward : ", reward)
			#print("Next State : ", next_state)
			#print()
			agent.append_sample(state, action, reward)
			score += reward
			state = copy.deepcopy(next_state)

			if done:
				# 에피소드마다 정책신경망 업데이트
				agent.train_model()
				scores.append(score)
				episodes.append(e)
				score = round(score,2)
				print("episode:", e, "  score:", score, "  time_step:", global_step)

		# 100 에피소드마다 학습 결과 출력 및 모델 저장
		if e % 100 == 0:
			pylab.plot(episodes, scores, 'b')
			pylab.savefig("./save_graph/pg.png")
			agent.model.save_weights("./save_model/pg.h5")
			print("Model Saved ...")
'''