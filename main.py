import numpy as np
from random import choice
import random
from enum import Enum
import itertools
import matplotlib.pyplot as plt

class Item(Enum):
    X = 'X'
    O = 'O'

class Player():
    def __init__(self, name, playerType, item, wins, loses, draws):
        self.name = name
        self.type = playerType
        self.item = item
        self.wins = wins
        self.loses = loses
        self.draws = draws

    def setWins(self, wins):
        self.wins = wins

    def setLoses(self, loses):
        self.loses = loses

    def setDraws(self, draws):
        self.draws = draws

    def getName(self):
        return self.name

    def getWins(self):
        return self.wins

    def getLoses(self):
        return self.loses

    def getDraws(self):
        return self.draws

    def getType(self):
        return self.type

    def getItem(self):
        return self.item

    def getName(self):
        return self.name

class AI_RL(Player):

    def __init__(self, name,item, allCombinations, epsilon):
        super().__init__(name, 'IA', item, 0,0,0)
        self.epsilon = epsilon   # proba d'exploration
        self.learning = 0.05  # changement des probas des combinaisons
        self.current_moves = {}
        self.allCombinations = allCombinations
        self.state_values = self.loadFile()
        self.movesPlay = np.array([])

    def update(self, win):
        if win:
            for move in self.movesPlay:
                if(self.state_values[int(move)] == 0):
                    self.state_values[int(move)] = self.learning
                else:
                    self.state_values[int(move)] *= (1 + self.learning)
        else:
            for move in self.movesPlay:
                if(self.state_values[int(move)] == 0):
                    self.state_values[int(move)] -= self.learning
                else:
                    self.state_values[int(move)] *= (1 - self.learning)

        self.movesPlay = np.array([])
        self.updateFile()
        print("Fichier mis à jour")



    def bestMove(self, combinationsArray, emptyCells):
        valuesList = []
        for i in range(len(combinationsArray)):
            valuesList.append(self.state_values[i])
        argMax = np.argmax(valuesList)
        self.movesPlay = np.append(self.movesPlay, combinationsArray[argMax])
        return emptyCells[argMax]


    def combinations(self, board, emptyCellsArray):
        arrayOfNextStateId = []

        actualBoard = np.copy(board)
        actualBoardClass = Board(np.copy(board))

        for i in range(len(emptyCellsArray)):
            actualBoardClass.addItem(emptyCellsArray[i], self.item)
            arrayOfNextStateId.append(  list(self.allCombinations.keys())[  list(self.allCombinations.values()).index(actualBoardClass.getBoard().tolist())  ]  )
            actualBoardClass.resetWithTemplate(actualBoard)

        return arrayOfNextStateId

    def emptyCells(self, board):
        emptyCells = np.array([])
        for x in range(len(board)):
            for y in range (len(board[0])):
                if board[x][y] == ' ':
                    emptyCells = np.append(emptyCells,str(x)+str(y))
        return emptyCells

    def input(self, board):
        emptyCellsArray = self.emptyCells(board) # empty cells

        combinationsArray = self.combinations(board, emptyCellsArray)
        if random.random() < self.epsilon: #exploration
            randomNumber = random.randint(0, len(emptyCellsArray)-1);
            move = emptyCellsArray[randomNumber]

            self.movesPlay = np.append(self.movesPlay, combinationsArray[randomNumber])
            print(str(self.movesPlay))

            return move

        else:
            bestMove = self.bestMove(combinationsArray, emptyCellsArray) # max (proba) ou random (random < epsilon (exploration))


            print(str(self.movesPlay))
            return bestMove

    def loadFile(self):
        return np.loadtxt('trained_state_values_' + self.item + '.txt', dtype=np.float64)

    def updateFile(self):
        np.savetxt('trained_state_values_' + self.item + '.txt', self.state_values, fmt = '%.6f')


#Classe pour les joueurs humain
class Human(Player):
    def __init__(self, name, item) :
        super().__init__(name, 'Humain', item, 0,0,0)

    def input(self, board) :
        print(self.name + "Joue" + '/n')
        return input("Rentrez la position de votre coup")
    

#Classe pour la grille de jeu
class Board:

    def __init__(self, boardTemplate):
        self.boardTemplate = boardTemplate

    def resetWithTemplate(self, board):
        self.boardTemplate = np.array(board)

    def addItem(self, position, item):
        x = int(position[0])
        y = int(position[1])
        self.boardTemplate[x, y] = item

    def getBoard(self):
        return self.boardTemplate

    def resetBoard(self):
        self.boardTemplate = np.zeros((3, 3), dtype=str)
        self.boardTemplate[:, :] = ' '

    def checkPosition(self, position):
        if(not (position.isnumeric()) or len(position) != 2):#On vérifie si c'est bien un numérique, et si la longeur est bien de 2 (pour l'abscisse et l'ordonné)
            return False
        x = int(position[0])
        y = int(position[1])
        if (self.positionIsValid(x, y) == True):
            return self.positionIsTaken(x, y)
        return False

    def positionIsValid(self, x, y):
        return ((0 <= x <= 2) and (0 <= y <= 2)) #Début de la matrice à 00(haut gauche) et la fin a 22(bas droite)

    def positionIsTaken(self, x, y):
        return self.boardTemplate[x, y] == ' '

    def verifyEndGame(self, item):
        if (self.whoWin(item) == True):
            print(item + ' a gagné')
            return 'Win'
        elif (self.isDraw()):
            print('Égalité')
            return 'Draw'
        else:
            return False

    def isDraw(self):
        status = True
        for row in self.boardTemplate:
            for case in row:
                if (case == ' '):
                    status = False
        return status

    def whoWin(self, item):
        print(self.boardTemplate)
        # horizontal
        for row in self.boardTemplate:
            if (row[0] == item and row[1] == item and row[2] == item):
                return True

        # vertical
        for i in range(3):
            if (self.boardTemplate[0][i] == item
                    and self.boardTemplate[1][i] == item
                    and self.boardTemplate[2][i] == item):
                return True

        # diagonal
        if (self.boardTemplate[0][0] == item
                and self.boardTemplate[1][1] == item
                and self.boardTemplate[2][2] == item):
            return True
        if (self.boardTemplate[2][0] == item
                and self.boardTemplate[1][1] == item
                and self.boardTemplate[0][2] == item):
            return True

#Classe fonctionnement du jeu
class Game:
    player1 = None
    player2 = None

    def __init__(self, board) :
        self.board = board

    def setPlayer1(self, player1):
        self.player1 = player1

    def setPlayer2(self, player2):
        self.player2 = player2

    def getPlayer1(self):
        return self.player1

    def getPlayer2(self):
        return self.player2

    def round(self, player):
        position = player.input(board.getBoard())

        while (board.checkPosition(position) == False):
            print('Position non-valable, réessayez')
            position = player.input(board.getBoard())

        board.addItem(position, player.getItem())
        print(board.getBoard())

    def verifyEndGame(self, count, playerWhoPlayed, otherPlayer):
        if(count >= 5):
            endGame = board.verifyEndGame(playerWhoPlayed.getItem())
            if(endGame == 'Win'):
                if(playerWhoPlayed.getType() == "IA"):
                    playerWhoPlayed.update(True)
                if(otherPlayer.getType() == "IA"):
                    otherPlayer.update(False)
                playerWhoPlayed.setWins(playerWhoPlayed.getWins() + 1)
                otherPlayer.setLoses(otherPlayer.getLoses() + 1)
                return True
            elif(endGame == 'Draw'):
                playerWhoPlayed.setDraws(playerWhoPlayed.getDraws() + 1)
                otherPlayer.setDraws(otherPlayer.getDraws() + 1)
                return True
        return False

    def game(self):
        # logique du jeu
        count = 0
        position = ' '
        print(self.board.getBoard())
        
        # player 1 input - First Move
        self.round(self.player1)
        count +=1
        
        while True :

            # player 2 input
            self.round(self.player2)
            count +=1

            if(self.verifyEndGame(count, self.player2, self.player1) == True): break

            # player 1 input
            self.round(self.player1)
            count +=1

            if(self.verifyEndGame(count, self.player1, self.player2) == True): break

        return 0

    def menu(self):

        player = ['X','O',' ']
        all_possible_states = [[list(i[0:3]),list(i[3:6]),list(i[6:10])] for i in itertools.product(player, repeat = 9)]
        n_states = len(all_possible_states)
        allCombinations = {}
        
        
        
        for state in range (n_states):
            allCombinations[state] = all_possible_states[state]

        print("Choisir le mode de jeu")
        gameMode = input("1 - IA vs IA; 2 Humain vs IA; - 3 Humain vs Humain; 4 - IA vs IA pas entrainner")
        while(not (gameMode.isnumeric()) or len(gameMode) != 1):
            gameMode = input("1 - IA vs IA; 2 Humain vs IA - 3 Humain vs Humain")

        if(gameMode == '1'):
            self.setPlayer1(AI_RL('IA_1',Item.X.value, allCombinations, 0.105))
            self.setPlayer2(AI_RL('IA_1',Item.O.value, allCombinations, 0.105))

        if(gameMode == '2'):
            self.setPlayer1(Human('Joueur',Item.X.value))
            self.setPlayer2(AI_RL('IA_1',Item.O.value, allCombinations, 0.105))

        if(gameMode == '3'):
            self.setPlayer1(Human('Joueur_1',Item.X.value))
            self.setPlayer1(Human('Joueur_2',Item.O.value))

        if(gameMode == '4'):
            self.setPlayer1(AI_RL('IA_1',Item.X.value, allCombinations, 0.105))
            self.setPlayer2(AI_RL('IA_1',Item.O.value, allCombinations, 1.0))

        numberGame = input("Choisir le nombre de partie")
        while(not (numberGame.isnumeric())):
            numberGame = input("Choisir le nombre de partie")

        return numberGame

    def main(self):
        numberGame = int(self.menu())

        while numberGame>0: # mettre un argument dans l'appel du fichier (Exemple : -n 1000 => 1000 games
            player1 = self.player1
            player2 = self.player2
            if random.random() > 0.5:
                self.setPlayer1(player1)
                self.setPlayer2(player2)
            else:
                self.setPlayer1(player2)
                self.setPlayer2(player1)

            self.game()
            board.resetBoard()
            numberGame -= 1

        listOfWins = [self.getPlayer1().getWins(), self.getPlayer1().getLoses(), self.getPlayer1().getDraws()]
        plt.pie(listOfWins, labels = ['Wins', 'Loses', 'Draws'],       # valeurs et labels
           autopct = lambda z: str(round(z, 2)) + '%', # affichage des pourcentages dans les secteurs
           pctdistance = 0.7,                    # distance au centre pour l'affichage des pourcentages
           labeldistance = 1.2)

        plt.title('Diagramme en secteurs contenant le résultats des parties de ' + self.getPlayer1().getName())
        plt.show()

        print("c'est fini")

boardTemplate = np.array([
    [' ',' ',' '],
    [' ',' ',' '],
    [' ',' ',' '],
]
);

board = Board(boardTemplate)
game = Game(board)
game.main()