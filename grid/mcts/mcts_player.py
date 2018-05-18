from environment import Env, Node

import random
import time

#------------------------------------------------------------
# MCTS UCT
#------------------------------------------------------------
def UCT(rootstate, itermax, verbose=False):
	'''
	rootstate에서 시작하는 itermax 반복에 대한 UCT 검색을 수행
	rootstate에서 최고의 이동을 반환
	게임 결과가 [0.0, 1.0] 사이에 있는 2명의 교대 플레이어를 가정(플레이어 1부터 시작)
	'''

	# rootstate에서 시작하는 root 노드 생성
	rootnode = Node(state=rootstate)

	# itermax만큼 반복
	for i in range(itermax):
		node = rootnode
		state = rootstate.Clone()

		# Select - 노드가 완전히 Expand 되고(시도하지 않은 이동이 없고) terminal이 아닐 때까지 탐색
		# Select - 노드가 시도하지 않은 이동이 없고 child 노드가 있다면 계속 탐색
		while node.untriedMoves == [] and node.childNodes != []:
			node = node.UCTSelectChild()
			state.DoMove(node.move)

		# Expand - Expand 가능한 경우 (state / node가 terminal이 아닌 경우)
		# Expand - 시도하지 않은 이동이 남은 경우, 이동 후 child 노드 추가
		if node.untriedMoves != []:
			# 아직 이동하지 않은 경우를 선택
			m = random.choice(node.untriedMoves)
			# 선택된 경우를 이동
			state.DoMove(m)
			# child를 추가하고 그곳으로 내려감
			node = node.AddChild(m, state)

		# state.GetRandomMove () 함수를 사용하여 훨씬 더 빨리 처리 할 수 있음
		# Rollout - state가 terminal이 아닐 경우
		# Rollout - 게임이 끝날 때까지(=가능한 이동이 없을 때까지) 탐색
		while state.GetMoves() != []:
			state.DoMove(random.choice(state.GetMoves()))

		# Backpropagate - 확장 된 노드에서 역 전파하여 루트 노드로 다시 작업
		while node != None:
			# state가 terminal일 경우 - node.playerJostMoved의 POV 결과를 통해 노드를 업데이트
			node.Update(state.GetResult(node.playerJustMoved))
			node = node.parentNode

	# 트리에 대한 일부 정보 출력 - 생략 가능
	if (verbose):
		print(rootnode.TreeToString(0))
	else:
		print(rootnode.ChildrenToString())

	# 가장 많이 방문한 move를 반환
	return sorted(rootnode.childNodes, key=lambda c: c.visits)[-1].move


#------------------------------------------------------------
# Play Game
#------------------------------------------------------------
def UCTPlayGame():
	"""
		Play a sample game between two UCT players where each player gets a different number of UCT iterations (= simulations = tree nodes).
		state는 양 플레이어가 공통으로 공유하는 env 정보다
		state의 playerJustMoved에 따라서 서로 다른 플레이어가 행동한다
	"""
	# state = OthelloState(4) # uncomment to play Othello on a square board of the given size
	state = Env()  # uncomment to play OXO
	# state = NimState(15) # uncomment to play Nim with the given number of starting chips
	while (state.board.finished == False):
		print(str(state))
		if state.playerJustMoved == 1:
			# play with values for itermax and verbose = True
			m = UCT(rootstate=state, itermax=100, verbose=True)
		else:
			# play with values for itermax and verbose = True
			m = UCT(rootstate=state, itermax=100, verbose=True)
		print("Best Move: " + str(m) + "\n")
		state.DoMove(m)
		#time.sleep(5)
	if state.GetResult(state.playerJustMoved) == 1.0:
		print("Player " + str(state.playerJustMoved) + " wins!")
	elif state.GetResult(state.playerJustMoved) == 0.0:
		print("Player " + str(3 - state.playerJustMoved) + " wins!")
	else:
		print("Nobody wins!")


#------------------------------------------------------------
# Main
#------------------------------------------------------------
if __name__ == "__main__":
	""" Play a single game to the end using UCT for both players.
	"""
	UCTPlayGame()
