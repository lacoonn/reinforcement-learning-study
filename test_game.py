from environment import Envirornment
from algo import Algo

def test_game():
    player1 = Algo(2)
    player2 = Algo(2)
    env = Envirornment(player1, player2)

    print("start")
    env.play_game()

if __name__ == '__main__':
    test_game()