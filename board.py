import numpy as np

class Board(object):
    def __init__(self):
        self.size = 16
        self.data = np.zeros([self.size, self.size], np.int8)
        self.finished = False
        self.turn = 1

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

        # self.check_finish_condition()
        self._check_if_finished_after_move(x, y, self.data[x, y])
        
        if self.finished == False:
            self.turn += 1

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

    


t = Board()
t.put_value(0, 0)
t.put_value(5, 0)
t.put_value(0, 5)
t.draw()
print(t.data)