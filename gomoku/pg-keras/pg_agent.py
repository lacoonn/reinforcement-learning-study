'''
Policy Gradient Player
'''

import copy
import pylab
import math
import os
import numpy as np
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import Activation
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
		'''
		model = Sequential()
		model.add(Dense(150, input_dim=self.state_size, activation='relu'))
		model.add(Dense(250, activation='relu'))
		model.add(Dense(self.action_size, activation='softmax'))
		model.summary()
		'''
		
		model = Sequential()

		model.add(Dense(512, input_dim=self.state_size, init='uniform'))
		model.add(BatchNormalization())
		model.add(Activation('relu'))

		model.add(Dense(1024))
		model.add(BatchNormalization())
		model.add(Activation('relu'))

		model.add(Dense(512))
		model.add(BatchNormalization())
		model.add(Activation('relu'))

		model.add(Dense(self.action_size))
		model.add(BatchNormalization())
		model.add(Activation('softmax'))
		

		model.summary()
		return model

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
		result = self.model.predict(state)
		if False:
			print("State : {0}".format(state.shape))
			print(state)
			print("Prediction : {0}".format(result.shape))
			print(result)			
			print()
		policy = result[0]
		# 확률 합 구함
		total_prob = 0.0
		for i in range(len(policy)):
			total_prob += policy[i]
		if total_prob < 0.9:
			print("total prob : ", total_prob)
			print(policy)
			exit(1)
		return np.random.choice(self.action_size, 1, p=policy)[0]
		'''
		# 이미 돌이 있는 자리의 확률을 0으로 조정 1
		untried = 0.0
		for i in range(len(state[0])):
			if state[0][i] == 0:
				untried += 1.0
		for i in range(len(state[0])):
			if state[0][i] != 0:
				tried = policy[i]
				for j in range(len(state[0])):
					if state[0][j] == 0:
						policy[j] += tried / untried
				policy[i] = 0.0
		return np.random.choice(self.action_size, 1, p=policy)[0]
		'''
		'''
		# 이미 돌이 있는 자리의 확률을 0으로 조정 2
		try:
			# NaN 확인
			for i in range(len(policy)):
				if math.isnan(policy[i]):
					raise(Exception)
			# 이미 돌이 놓인 자리는 확률 0
			for i in range(len(policy)):
				if state[0][i] != 0:
					policy[i] = 0.0
			# 확률 합 구함
			total_prob = 0.0
			for i in range(len(policy)):
				total_prob += policy[i]
			# 확률 합 1로 보정
			weight = 1.0 / total_prob
			for i in range(len(policy)):
				policy[i] *= weight
			# 장소 선택
			random_choice = np.random.choice(self.action_size, 1, p=policy)[0]
			return random_choice
		except Exception as e:
			print(e)
			print("State : ", state)
			print("Policy : ", policy)
			#print("Choice : ", random_choice)
			exit(1)
		'''
		'''
		# np.random.choice 대신 arg_max를 사용
		for i in range(len(policy)):
			if state[0][i] != 0:
				policy[i] = 0.0
		return np.argmax(policy)
		'''
		
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
	
	# 내 마지막 행동으로 인한 패배를 rewards에 추가
	def lose_game(self, reward):
		self.rewards[-1] = reward

	# 모델 저장
	def save_model(self, filepath = './save_model/pg.h5'):
		self.model.save_weights(filepath)

	# 모델 로드
	def load_model(self, filepath = './save_model/pg.h5'):
		if (os.path.isfile(os.getcwd() + filepath[1:]) == True):
			self.model.save_weights(filepath)
			print("Model is Loaded ...")
			print()
		else:
			print("Train new Model ...")
			print()

	# 그래프 저장
	def save_graph(self, episodes, scores, figure = 0, option = 'b', filepath = './save_graph/pg.png'):
		pylab.figure(figure)
		pylab.plot(episodes, scores, option)
		pylab.savefig(filepath)