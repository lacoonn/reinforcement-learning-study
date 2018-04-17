from board import Board
import time

class Envirornment(object):
    def __init__(self, player1, player2):
        self.player1 = player1
        self.player2 = player2

    def play_game(self):
        board = Board()

        print(board.is_finished())
        while board.is_finished() == False:
            '''
            이 Loop 1번은 1턴을 의미합니다
            '''
            if board.turn % 2 == 1:
                self.player1.observe(board)
                x, y = self.player1.predict()
                print("Turn : {0}, Color : {1}, Eval : {2}".format(board.turn, self.player1.board.MY_COLOR, self.player1.eval))
                board.put_value(x, y)
                #print("Turn ", board.turn, ", ", self.player1.board.MY_COLOR, " : ", self.player1.eval)
                

            else:
                self.player2.observe(board)
                x, y = self.player2.predict()
                print("Turn : {0}, Color : {1}, Eval : {2}".format(board.turn, self.player2.board.MY_COLOR, self.player2.eval))                
                board.put_value(x, y)
                #print("Turn ", board.turn, ", ", self.player2.board.MY_COLOR, " : ", self.player2.eval)

            if board.is_finished == True:
                if board.turn % 2 == 1:
                    self.player1.observe_finish(board, True)
                    self.player2.observe_finish(board, False)
                else:
                    self.player1.observe_finish(board, False)
                    self.player2.observe_finish(board, True)
                board.draw()

            board.draw()
            '''
            # 테스트를 위해서 시간을 제어합니다
            time.sleep(1)
            '''