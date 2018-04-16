from board import Board

class Envirornment(object):
    def __init__(self, player1, player2):
        self.player1 = player1
        self.player2 = player2

    def play_game(self):
        board = Board()

        while(board.is_finished == False):
            if board.turn % 2 == 1:
                self.player1.observe(board)
                x, y = self.player1.predict()
                board.put_value(x, y)

            else:
                self.player2.observe(board)
                x, y = self.player2.predict()
                board.put_value(x, y)

            if board.is_finished == True:
                if board.turn % 2 == 1:
                    self.player1.observe_finish(board, True)
                    self.player2.observe_finish(board, False)
                else:
                    self.player1.observe_finish(board, False)
                    self.player2.observe_finish(board, True)
                board.draw()