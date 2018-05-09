import copy
import numpy as np


class BoardAnalyzer(object):
    def __init__(self):
        self.finished = False
        self.OUTBD = -1
        self.EMPTY = 0
        self.BLACK = 1
        self.WHITE = 2
        self.eval = 0

    def analyze(self, board):
        self.size = board.get_size()
        self.data = copy.deepcopy(board.data)
        self.turn = board.get_turn()
        if self.turn % 2 == 1:  # 흑돌(1턴=홀수)
            self.MY_COLOR = self.BLACK
            self.YOUR_COLOR = self.WHITE
        else:  # 백돌(2턴=짝수)
            self.MY_COLOR = self.WHITE
            self.YOUR_COLOR = self.BLACK

        self.moves_left = self.size ** 2
        self.winner_value = 0.0 # not defined yet

    def get_size(self):
        return self.size

    def get_turn(self):
        return self.turn

    def draw(self):
        pic = np.chararray([self.size, self.size])
        for x in range(self.size):
            for y in range(self.size):
                if self.data[x, y] == 0:
                    pic[x, y] = '_'
                elif self.data[x, y] == 1:
                    pic[x, y] = 'o'
                elif self.data[x, y] == 2:
                    pic[x, y] = 'x'
                else:
                    pic[x, y] = 'N'
        print(pic)
        # print(pic[0:5, 0:5])

    def inverse(self):
        for x in range(self.size):
            for y in range(self.size):
                if self.data[x, y] == 1:
                    self.data[x, y] = 2
                elif self.data[x, y] == 2:
                    self.data[x, y] = 1

    def put_value(self, x, y):
        if self.turn % 2 == 1:
            self.data[x, y] = 1
        else:
            self.data[x, y] = 2

        '''
        # 돌을 놓음과 동시에 점수를 계산
        self.evaluate()
        '''

        # 게임 종료 여부 체크
        self._check_if_finished_after_move(x, y, self.data[x, y])
        # 게임이 끝나면 점수 계산
        if self.finished == True:
            self.evaluate()

        # 게임이 끝나지 않았을 경우 1턴 추가(1턴부터 시작이기 때문에 끝나면 올릴 필요 없음)
        if self.finished == False:
            self.turn += 1

        # 반환값으로 종료 여부를 기대하는 조건문을 위해 게임 종료 여부 반환
        return self.finished

    def get_value(self, x, y):
        if x >= 0 and x < self.size and y >= 0 and y < self.size:
            return self.data[x, y]
        else:
            return self.OUTBD

    def is_finished(self):
        return self.finished

    def check_finish_condition(self):
        '''
        미구현
        '''
        return self.finished

    def _check_if_finished_after_move(self, x, y, value):
        if self.moves_left == 0:
            self.finished = True

        self._check_if_finished_vertically(x, y, value)
        self._check_if_finished_horizontally(x, y, value)
        self._check_if_finished_diagonals(x, y, value)

    def _check_if_finished_vertically(self, x, y, value):
        self._check_if_finished(x, y, value, 0, 1)

    def _check_if_finished_horizontally(self, x, y, value):
        self._check_if_finished(x, y, value, 1, 0)

    def _check_if_finished_diagonals(self, x, y, value):
        self._check_if_finished(x, y, value, 1, 1)
        self._check_if_finished(x, y, value, 1, -1)

    def _check_if_finished(self, x, y, value, xOffset, yOffset):
        sameStonesLeft = 0
        for i in range(1, 6):
            xx = x - i * xOffset
            yy = y - i * yOffset

            if xx < 0 or xx >= self.size:
                break

            if yy < 0 or yy >= self.size:
                break

            if self.data[xx, yy] == 0.0:
                break

            if self.data[xx, yy] == value:
                sameStonesLeft += 1
            else:
                break

        sameStonesRight = 0
        for i in range(1, 5):
            xx = x + i * xOffset
            yy = y + i * yOffset

            if xx < 0 or xx >= self.size:
                break

            if yy < 0 or yy >= self.size:
                break

            if self.data[xx, yy] == 0.0:
                break

            if self.data[xx, yy] == value:
                sameStonesRight += 1
            else:
                break

        sameStones = sameStonesLeft + sameStonesRight + 1
        if sameStones >= 5:
            self.finished = True
            self.winner_value = value

    def isNear(self, x, y):
        dist = 1
        for t_x in range(x - dist, x + dist + 1):
            for t_y in range(y - dist, y + dist + 1):
                if (self.get_value(t_x, t_y) == self.WHITE) or (self.get_value(t_x, t_y) == self.BLACK):
                    return True

        return False

    def evaluate(self):
        '''
        현재 바둑판을 분석해 점수를 계산합니다
        '''
        # 0~5 연속된 돌의 개수, 0~2 막힌 방향 개수
        my_block_ary = np.zeros([6, 3], np.int8)
        your_block_ary = np.zeros([6, 3], np.int8)

        temp_eval = 0
        x = 0
        y = 0

        # 배열 초기화(파이썬에서는 필요 없음)
        for i in range(0, 6):
            for j in range(0, 3):
                my_block_ary[i][j] = 0
                your_block_ary[i][j] = 0

        # 바둑판에 배치된 모든 말에 대해 검사한다.
        for x in range(-1, self.size + 1):
            for y in range(-1, self.size + 1):
                current = self.get_value(x, y)  # 현재 위치의 상태

                OUTBD = self.OUTBD
                EMPTY = self.EMPTY
                BLACK = self.BLACK
                WHITE = self.WHITE
                MC = self.MY_COLOR
                EC = self.YOUR_COLOR
                
                myCount = 0
                blockCount = 0
                enemy = 0

                right = self.get_value(x + 1, y)
                rightDown = self.get_value(x + 1, y + 1)
                down = self.get_value(x, y + 1)
                downLeft = self.get_value(x - 1, y + 1)

                # right가 바둑알이면서 현재 위치의 상태와 다를 때
                if ((right == WHITE or right == BLACK) and current != right):
                    if right == WHITE:
                        enemy = BLACK
                    else:
                        enemy = WHITE
                    myCount = 1
                    blockCount = 0  # 좌우에 막힌 개수(최대 2)
                    if(current == OUTBD or current == enemy):
                        # 시작점이 바둑판 외부이거나 적이라면 blockCount가 1로 시작(이미 한 면이 막혔으므로)
                        blockCount = 1
                    for i in range(2, 6):
                        tempVal = self.get_value(x + i, y)
                        if(tempVal == OUTBD or tempVal == enemy):
                            blockCount += 1  # 찾은 장소가 바둑판 밖이거나 적이면 blockCount를 +1 하고 break
                            break

                        if(tempVal == EMPTY):
                            break
                        myCount += 1

                    if (right == MC):
                        my_block_ary[myCount][blockCount] += 1
                    else:
                        your_block_ary[myCount][blockCount] += 1
                    # tempEval += getBlockScore(myCount, blockCount, right)

                # right-down가 바둑알이면서 현재 위치의 상태와 다를 때
                if ((rightDown == WHITE or rightDown == BLACK) and current != rightDown):
                    if rightDown == WHITE:
                        enemy = BLACK
                    else:
                        enemy = WHITE
                    myCount = 1
                    blockCount = 0  # 좌우에 막힌 개수(최대 2)
                    if(current == OUTBD or current == enemy):
                    # 시작점이 바둑판 외부이거나 적이라면 blockCount가 1로 시작(이미 한 면이 막혔으므로)
                        blockCount = 1
                    for i in range(2, 6):
                        tempVal = self.get_value(x + i, y + i)
                        if(tempVal == OUTBD or tempVal == enemy):
                            blockCount += 1  # 찾은 장소가 바둑판 밖이거나 적이면 blockCount를 +1 하고 break
                            break

                        if(tempVal == EMPTY):
                            break
                        myCount += 1

                    if (rightDown == MC):
                        my_block_ary[myCount][blockCount] += 1
                    else:
                        your_block_ary[myCount][blockCount] += 1
                    # tempEval += getBlockScore(myCount, blockCount, rightDown)

                # down가 바둑알이면서 현재 위치와는 다를 때
                if ((down == WHITE or down == BLACK) and current != down):
                    if down == WHITE:
                        enemy = BLACK
                    else:
                        enemy = WHITE
                    myCount = 1
                    blockCount = 0  # 좌우에 막힌 개수(최대 2)
                    if(current == OUTBD or current == enemy):
                        blockCount = 1 # 시작점이 바둑판 외부이거나 적이라면 blockCount가 1로 시작(이미 한 면이 막혔으므로)
                    for i in range(2, 6):
                        tempVal = self.get_value(x, y + i)
                        if(tempVal == OUTBD or tempVal == enemy):
                            blockCount += 1  # 찾은 장소가 바둑판 밖이거나 적이면 blockCount를 +1 하고 break
                            break

                        if(tempVal == EMPTY):
                            break
                        myCount += 1

                    if (down == MC):
                        my_block_ary[myCount][blockCount] += 1
                    else:
                        your_block_ary[myCount][blockCount] += 1
                    #tempEval += getBlockScore(myCount, blockCount, down)

                # down-left가 바둑알이면서 현재 위치와는 다를 때
                if ((downLeft == WHITE or downLeft == BLACK) and current != downLeft):
                    if downLeft == WHITE:
                        enemy = BLACK
                    else:
                        enemy = WHITE
                    myCount = 1
                    blockCount = 0  # 좌우에 막힌 개수(최대 2)
                    if(current == OUTBD or current == enemy):
                    # 시작점이 바둑판 외부이거나 적이라면 blockCount가 1로 시작(이미 한 면이 막혔으므로)
                        blockCount = 1
                    for i in range(2, 6):
                        tempVal = self.get_value(x - i, y + i)
                        if(tempVal == OUTBD or tempVal == enemy):
                            blockCount += 1  # 찾은 장소가 바둑판 밖이거나 적이면 blockCount를 +1 하고 break
                            break

                        if(tempVal == EMPTY):
                            break
                        myCount += 1

                    if (downLeft == MC):
                        my_block_ary[myCount][blockCount] += 1
                    else:
                        your_block_ary[myCount][blockCount] += 1
                    # tempEval += getBlockScore(myCount, blockCount, downLeft)

        temp_eval += self.get_block_score(my_block_ary, MC)
        temp_eval += self.get_block_score(your_block_ary, EC)

        # 종료 판단
        for i in range(0, 3):
            if my_block_ary[5][i] > 0 or your_block_ary[5][i] > 0:
                self.finished = True

        self.eval = temp_eval * 2


    def get_block_score(self, block_ary, color):
        '''
        Args:
            block_ary : Array [5][3]
            color : 본인 돌의 색깔(흑, 백)
        '''
        color_weight = 1
        value = 0

        if color != self.MY_COLOR:
            color_weight = -1.1

        # --------------------------------------------------------------------
        # 연속돌 1개 + 1개 막힘
        value += block_ary[1][1] * 1

        # 연속돌 1개 + 0개 막힘
        value += block_ary[1][0] * 3
        # --------------------------------------------------------------------
        # 연속돌 2개 + 1개 막힘
        value += block_ary[2][1] * 10
        # 연속돌 2개 + 0개 막힘
        value += block_ary[2][0] * 30
        # --------------------------------------------------------------------
        # 연속돌 3개 + 1개 막힘
        value += block_ary[3][1] * 100
        # 연속돌 3개 + 0개 막힘
        if (block_ary[3][0] >= 2):
            value += 3000  # /* 승부결정 */
        else:
            value += block_ary[3][0] * 300
        # --------------------------------------------------------------------
        # 연속돌 4개 + 1개 막힘
        if (block_ary[4][1] >= 2):
            value += 3000  # /* 승부결정 */
        else:
            value += block_ary[4][1] * 1000
        # 연속돌 4개 + 0개 막힘
        value += block_ary[4][0] * 3000  # /* 승부결정 */
        # --------------------------------------------------------------------
        # 연속돌 5개
        value += block_ary[5][2] * 50000  # /* 엔딩조건 */
        value += block_ary[5][1] * 50000  # /* 엔딩조건 */
        value += block_ary[5][0] * 50000  # /* 엔딩조건 */
        # --------------------------------------------------------------------

        return (value * color_weight)
