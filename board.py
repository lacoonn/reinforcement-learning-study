import numpy as np

class Board(object):
    def __init__(self):
        self.size = 16
        # 바둑판은 생성 시 0(EMPTY)로 초기화
        self.data = np.zeros([self.size, self.size], np.int8)
        self.finished = False
        self.turn = 1

        self.OUTBD = -1
        self.EMPTY = 0
        self.BLACK = 1
        self.WHITE = 2

        self.moves_left = self.size ** 2
        self.winner_value = 0.0 # not defined yet

    def get_size(self):
        return self.size

    def get_turn(self):
        return self.turn

    def draw(self):
        pic = np.chararray([self.size, self.size], itemsize=5, unicode=True)
        for x in range(self.size):
            for y in range(self.size):
                if self.data[x, y] == self.EMPTY:
                    pic[x, y] = '_'
                    #pic[x, y] = '□'
                elif self.data[x, y] == self.BLACK:
                    #pic[x, y] = 'x'
                    pic[x, y] = '●'
                elif self.data[x, y] == self.WHITE:
                    #pic[x, y] = 'o'
                    pic[x, y] = '○'
                else:
                    pic[x, y] = 'N'
        # print(pic[0:5, 0:5])
        print(pic)

    def inverse(self):
        for x in range(self.size):
            for y in range(self.size):
                if self.data[x, y] == self.BLACK:
                    self.data[x, y] = self.WHITE
                elif self.data[x, y] == self.WHITE:
                    self.data[x, y] = self.BLACK

    def put_value(self, x, y):
        # 홀수턴(1턴)일 경우 흑돌
        if self.turn % 2 == 1:
            self.data[x, y] = self.BLACK
        # 짝수턴(2턴)일 경우 백돌
        else:
            self.data[x, y] = self.WHITE

        # 종료 여부 검사
        # self.check_finish_condition()
        self._check_if_finished_after_move(x, y, self.data[x, y])
        
        # 종료되지 않을 경우에만 턴을 추가(1턴부터 시작이기 때문)
        if self.finished == False:
            self.turn += 1
        
        # 반환값으로 종료 여부를 기대하는 조건문을 위해 종료 여부 반환
        return self.finished

    def get_value(self, x, y):
        return self.data[x, y]

    def is_finished(self):
        return self.finished

    def check_finish_condition(self):
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

    def test(self):
        t = Board()
        t.put_value(0, 0)
        t.put_value(5, 0)
        t.put_value(0, 5)
        t.draw()