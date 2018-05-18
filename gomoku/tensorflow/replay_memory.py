import numpy as np
import random

#------------------------------------------------------------
# 리플레이 메모리 클래스
#------------------------------------------------------------
class ReplayMemory:
	#--------------------------------
	# 초기화
	#--------------------------------
	def __init__(self, gridSize, maxMemory, discount):
		self.maxMemory = maxMemory
		self.gridSize = gridSize
		self.nbStates = self.gridSize * self.gridSize
		self.discount = discount

		self.inputState = np.empty((self.maxMemory, self.nbStates), dtype=np.uint8)
		self.actions = np.zeros(self.maxMemory, dtype=np.uint8)
		self.nextState = np.empty((self.maxMemory, self.nbStates), dtype=np.uint8)
		self.gameOver = np.empty(self.maxMemory, dtype=np.bool)
		self.rewards = np.empty(self.maxMemory, dtype=np.int8)
		self.count = 0
		self.current = 0

	#--------------------------------
	# 결과 기억
	#--------------------------------
	def remember(self, currentState, action, reward, nextState, gameOver):
		self.actions[self.current] = action
		self.rewards[self.current] = reward
		self.inputState[self.current, ...] = currentState
		self.nextState[self.current, ...] = nextState
		self.gameOver[self.current] = gameOver
		self.count = max(self.count, self.current + 1)
		self.current = (self.current + 1) % self.maxMemory

	#--------------------------------
	# 배치 구함
	#--------------------------------
	def getBatch(self, model, batchSize, nbActions, nbStates, sess, X):
		memoryLength = self.count
		chosenBatchSize = min(batchSize, memoryLength)

		inputs = np.zeros((chosenBatchSize, nbStates))
		targets = np.zeros((chosenBatchSize, nbActions))

		for i in range(chosenBatchSize):
			randomIndex = random.randrange(0, memoryLength)
			current_inputState = np.reshape(self.inputState[randomIndex], (1, nbStates))

			target = sess.run(model, feed_dict={X: current_inputState})

			current_nextState = np.reshape(self.nextState[randomIndex], (1, nbStates))
			current_outputs = sess.run(model, feed_dict={X: current_nextState})

			nextStateMaxQ = np.amax(current_outputs)

			'''
			if(nextStateMaxQ > winReward):
				nextStateMaxQ = winReward
			'''

			if(self.gameOver[randomIndex] == True):
				target[0, [self.actions[randomIndex]]] = self.rewards[randomIndex]
			else:
				target[0, [self.actions[randomIndex]]
           ] = self.rewards[randomIndex] + self.discount * nextStateMaxQ

			inputs[i] = current_inputState
			targets[i] = target

		return inputs, targets
#------------------------------------------------------------