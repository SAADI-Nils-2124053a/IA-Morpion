import numpy as np
import random
from enum import Enum
import itertools
import matplotlib.pyplot as plt

class Item(Enum):
    X = 'X'
    O = 'O'

"""
Section : Fonctions pour faire les statistiques
"""
# Graphic showing the number of wins, loses and draws 
def winsLosesDrawsPieGraphic(L, playerName):
    plt.pie(L, labels = ['Wins', 'Loses', 'Draws'],       # valeurs et labels
       autopct = lambda z: str(round(z, 2)) + '%', # affichage des pourcentages dans les secteurs
       pctdistance = 0.7,                    # distance au centre pour l'affichage des pourcentages
       labeldistance = 1.2)

    plt.title('Diagramme en secteurs contenant le résultats des parties de ' + playerName)
    plt.show()

# Graphic showing the probability of the first and second move
def firstAndSecondMoveGraphic(firstMove, secondMove):
    
    maxFirstMove = max(max(inner_list) for inner_list in firstMove)
    minFirstMove = min(min(inner_list) for inner_list in firstMove)

    f, ax1 = plt.subplots(1)
    ax1.imshow(firstMove, cmap='Reds')
    ax1.axis(False)
    plt.title('Structure matricelle des 1er coups joués')
    plt.show()
    
    fig, ax = plt.subplots(1, figsize=(6, 6))
    ax.imshow(np.arange(maxFirstMove)[None, :], cmap='Reds', extent=[minFirstMove, maxFirstMove, 0, maxFirstMove/10]);

    plt.title('Echelle pour la Structure matricelle des 1er coups joués')
    plt.show()
    
    print("Premier coup : " , firstMove)
    
    
    maxSecondMove = max(max(inner_list) for inner_list in secondMove)
    minSecondMove = min(min(inner_list) for inner_list in secondMove)
    
    f, ax2 = plt.subplots(1)
    ax2.imshow(secondMove, cmap='Reds')
    ax2.axis(False)
    plt.title('Structure matricelle des 2eme coups joués')
    plt.show()
    
    fig, ax = plt.subplots(1, figsize=(6, 6))
    ax.imshow(np.arange(maxFirstMove)[None, :], cmap='Reds', extent=[minSecondMove, maxSecondMove, 0,maxFirstMove/10]);
    plt.title('Echelle pour la Structure matricelle des 2eme coups joués')
    plt.show()
    
    print("Second coup " , secondMove)

# Graphic showing the number of wins of each type
def victoryTypePieGraphic(victoryType):
    plt.pie(victoryType, labels = ['Horizontalement', 'Verticalement', 'Diagonalement'],       # valeurs et labels
       autopct = lambda z: str(round(z, 2)) + '%', # affichage des pourcentages dans les secteurs
       pctdistance = 0.7,                    # distance au centre pour l'affichage des pourcentages
       labeldistance = 1.2)
    plt.title('Graphique de la répartition des types de victoire')
    plt.show()

# Graphic representing the new values due to all games
def learningValueGraphic(learningList, name):
    
    x = [i for i in range (len(learningList))]
    
    plt.scatter(x,learningList)
    
    a, b = np.polyfit(x,learningList,1)
    
    x_trace = np.linspace(0,len(learningList),len(learningList)*10)
    plt.plot(x_trace, a*x_trace+b , 'red')
    plt.title("L'évolution des valeurs d'apprentissage de "+ name)
    plt.show()
 
    
"""
Fin de section
"""

# Class containing the basic attributes and methods of a player
class Player():
    def __init__(self, name, playerType, item, wins, loses, draws, learningList):
        self.name = name
        self.type = playerType
        self.item = item
        self.wins = wins
        self.loses = loses
        self.draws = draws
        self.learningList = learningList

    def setWins(self, wins):
        self.wins = wins

    def setLoses(self, loses):
        self.loses = loses

    def setDraws(self, draws):
        self.draws = draws
    
    def addLearningList(self, learning):
        self.learningList.append(learning)

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
    
    def getLearningList(self):
        return self.learningList
    
# Class representing AI with reinforcement learning
class AI_RL(Player):

    def __init__(self, name,item, allCombinations, epsilon):
        super().__init__(name, 'IA', item, 0,0,0,[])
        self.epsilon = epsilon   # probabilité d'exploration
        self.learning = 0.05  # changement des probas des combinaisons
        self.current_moves = {}
        self.allCombinations = allCombinations
        self.state_values = self.loadFile()
        self.movesPlay = np.array([])

    # Update the file of the IA by modifying points
    def update(self, win):
        averageLearningGame = np.array([])
        if win:
            for move in self.movesPlay:
                if(self.state_values[int(move)] == 0):
                    self.state_values[int(move)] = self.learning
                else:
                    self.state_values[int(move)] += self.learning * (1 - (1/(1+np.exp(-self.state_values[int(move)]))))
                averageLearningGame = np.append(averageLearningGame, self.state_values[int(move)])
               
                
        else:
            for move in self.movesPlay:
                if(self.state_values[int(move)] == 0):
                    self.state_values[int(move)] -= self.learning        
                else:
                    self.state_values[int(move)] -= self.learning * ((1/(1+np.exp(-self.state_values[int(move)]))))
                  
                averageLearningGame = np.append(averageLearningGame, self.state_values[int(move)])
                
      
        
        self.addLearningList(np.average(averageLearningGame))
        self.movesPlay = np.array([])
        self.updateFile()
        print("Fichier mis à jour")

    # Return the best move to make for the AI with the values in the AI's file
    def bestMove(self, combinationsArray, emptyCells):
        valuesList = []
        for i in combinationsArray:
            valuesList.append(self.state_values[i]) # Get the value of the next state and add it to the list
        argMax = np.argmax(valuesList)  # Get the indice of the maximum value of the list
        self.movesPlay = np.append(self.movesPlay, combinationsArray[argMax])

        return emptyCells[argMax]

    # Return the id of all the possible next states of the board
    def combinations(self, board, emptyCellsArray):
        arrayOfNextStateId = []

        actualBoard = np.copy(board)    # Use to reset the board
        actualBoardClass = Board(np.copy(board))    # Use to get the next states id

        for i in range(len(emptyCellsArray)):
            actualBoardClass.addItem(emptyCellsArray[i], self.item)  # add the player's symbol to the board
            arrayOfNextStateId.append(list(self.allCombinations.keys())[list(self.allCombinations.values()).index(actualBoardClass.getBoard().tolist())]) # Get the indice of the value (board possibility in the allCombinations' dictionnary) corresponding to the new board (we added the player's symbol to the empty cell), and get the key with the indice
            actualBoardClass.resetWithTemplate(actualBoard) # reset the board to the previous state

        return arrayOfNextStateId

    # Return a list containing all the position of the empty cells of the given board
    def emptyCells(self, board):
        emptyCells = np.array([])
        for x in range(len(board)):
            for y in range (len(board[0])):
                if board[x][y] == ' ':
                    emptyCells = np.append(emptyCells,str(x)+str(y))
        return emptyCells

    # Return the move (random or best move) of the AI
    def input(self, board):
        emptyCellsArray = self.emptyCells(board) # get empty cells of the current board

        combinationsArray = self.combinations(board, emptyCellsArray) # Get the id of the possible next state
        if random.random() < self.epsilon: # exploration - random move
            randomNumber = random.randint(0, len(emptyCellsArray)-1);
            move = emptyCellsArray[randomNumber]

            self.movesPlay = np.append(self.movesPlay, combinationsArray[randomNumber])
            return move
        else:
            bestMove = self.bestMove(combinationsArray, emptyCellsArray)

            return bestMove

    # Load file to get the values
    def loadFile(self):
        return np.loadtxt('trained_state_values_' + self.item + '.txt', dtype=np.float64)

    # Update the file with the new values
    def updateFile(self):
        np.savetxt('trained_state_values_' + self.item + '.txt', self.state_values, fmt = '%.6f')

# Class representing a human player
class Human(Player):
    def __init__(self, name, item) :
        super().__init__(name, 'Humain', item, 0,0,0,[])

    # Return the human input for a move
    def input(self, board) :
        print(self.name + "Joue")
        return input("Rentrez la position de votre coup")
    
# Class representing the game board
class Board:
    
    victoryType = [0,0,0] #[horizontalement, verticalement, diagonalement]

    def __init__(self, boardTemplate):
        self.boardTemplate = boardTemplate

    # Reset the board
    def resetWithTemplate(self, board):
        self.boardTemplate = np.array(board)

    # Add an item to the board
    def addItem(self, position, item):
        x = int(position[0])
        y = int(position[1])
        self.boardTemplate[x, y] = item

    # Get the board
    def getBoard(self):
        return self.boardTemplate

    # Get the victory types' counters
    def getVictoryType(self):
        return self.victoryType

    # Set the victory type's counter
    def setVictoryType(self, type):
        if (type == "horizontal"):
            self.victoryType[0] += 1
        elif (type == "vertical"):
            self.victoryType[1] += 1
        elif (type == "diagonal"):
            self.victoryType[2] += 1

    # Reset the board
    def resetBoard(self):
        self.boardTemplate = np.zeros((3, 3), dtype=str)
        self.boardTemplate[:, :] = ' '

    # Check the position validity
    def checkPosition(self, position):
        if(not (position.isnumeric()) or len(position) != 2):#On vérifie si c'est bien un numérique, et si la longeur est bien de 2 (pour l'abscisse et l'ordonné)
            return False
        x = int(position[0])
        y = int(position[1])
        if (self.positionIsValid(x, y) == True):
            return self.positionIsTaken(x, y)
        return False

    # Verify if the position provided is valid
    def positionIsValid(self, x, y):
        return ((0 <= x <= 2) and (0 <= y <= 2))

    # Verify if the position provided is already taken
    def positionIsTaken(self, x, y):
        return self.boardTemplate[x, y] == ' '

    # Verify if the game ended or not
    def verifyEndGame(self, item):
        if (self.whoWin(item) == True):
            print(item + ' a gagné')
            return 'Win'
        elif (self.isDraw()):
            print('Égalité')
            return 'Draw'
        else:
            return False

    # Verify if there is a draw
    def isDraw(self):
        status = True
        for row in self.boardTemplate:
            for case in row:
                if (case == ' '):
                    status = False
        return status

    # Verify if the player won or not
    def whoWin(self, item):
        print(self.boardTemplate)
        # horizontal
        for row in self.boardTemplate:
            if (row[0] == item and row[1] == item and row[2] == item):
                self.setVictoryType("horizontal")
                return True

        # vertical
        for i in range(3):
            if (self.boardTemplate[0][i] == item
                    and self.boardTemplate[1][i] == item
                    and self.boardTemplate[2][i] == item):
                self.setVictoryType("vertical")
                return True

        # diagonal
        if (self.boardTemplate[0][0] == item
                and self.boardTemplate[1][1] == item
                and self.boardTemplate[2][2] == item):
            self.setVictoryType("diagonal")
            return True
        if (self.boardTemplate[2][0] == item
                and self.boardTemplate[1][1] == item
                and self.boardTemplate[0][2] == item):
            self.setVictoryType("diagonal")
            return True

# Class representing how the game works
class Game:
    player1 = None
    player2 = None
    firstMove = [[0,0,0],[0,0,0],[0,0,0]]
    secondMove = [[0,0,0],[0,0,0],[0,0,0]]    

    def __init__(self, board) :
        self.board = board

    def setPlayer1(self, player1):
        self.player1 = player1

    def setPlayer2(self, player2):
        self.player2 = player2

    def setFirstMove(self, position):
        x=int(position[0])
        y=int(position[1])
        self.firstMove[x][y] += 1
        
    def setSecondMove(self, position):
        x=int(position[0])
        y=int(position[1])
        self.secondMove[x][y] += 1
        
    def getPlayer1(self):
        return self.player1

    def getPlayer2(self):
        return self.player2
    
    def getFirstMove(self):
        return self.firstMove
    
    def getSecondMove(self):
        return self.secondMove
        
    def round(self, player):
        position = player.input(board.getBoard())

        while (board.checkPosition(position) == False):
            print('Position non-valable, réessayez')
            position = player.input(board.getBoard())

        board.addItem(position, player.getItem())
        print(board.getBoard())
        return position

    # return if the game ended or not
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

    # Represents a game
    def game(self):
        count = 0
        print(self.board.getBoard())
        # player 1 input - First Move*
        self.setFirstMove(self.round(self.player1))
       
        count +=1
        
        self.setSecondMove(self.round(self.player2))
        count +=1
        
        while True :

            # player 1 input
            self.round(self.player1)
            count +=1

            if(self.verifyEndGame(count, self.player1, self.player2) == True): break

            # player 2 input
            self.round(self.player2)
            count +=1

            if(self.verifyEndGame(count, self.player2, self.player1) == True): break

        return 0

    # Representing the menu where the user can chose the game mode and the number of games
    def menu(self):

        player = ['X','O',' ']
        all_possible_states = [[list(i[0:3]),list(i[3:6]),list(i[6:10])] for i in itertools.product(player, repeat = 9)]
        n_states = len(all_possible_states)
        allCombinations = {}
        
        for state in range (n_states):
            allCombinations[state] = all_possible_states[state]

        print("Choisir le mode de jeu")
        gameMode = input("1 - IA vs IA; 2 Humain vs IA; - 3 Humain vs Humain; 4 - IA vs IA pas entrainnée : ")
        while(not (gameMode.isnumeric()) or len(gameMode) != 1):
            gameMode = input("1 - IA vs IA; 2 Humain vs IA; - 3 Humain vs Humain; 4 - IA vs IA pas entrainée :")

        if(gameMode == '1'):
            self.setPlayer1(AI_RL('IA_1',Item.X.value, allCombinations, 0.1))
            self.setPlayer2(AI_RL('IA_2',Item.O.value, allCombinations, 0.1))

        if(gameMode == '2'):
            self.setPlayer1(Human('Joueur',Item.X.value))
            self.setPlayer2(AI_RL('IA',Item.O.value, allCombinations, 0.105))

        if(gameMode == '3'):
            self.setPlayer1(Human('Joueur_1',Item.X.value))
            self.setPlayer1(Human('Joueur_2',Item.O.value))

        if(gameMode == '4'):
            self.setPlayer1(AI_RL('IA_1',Item.X.value, allCombinations, 0.105))
            self.setPlayer2(AI_RL('IA_2',Item.O.value, allCombinations, 1.0))

        numberGame = input("Choisir le nombre de partie : ")
        while(not (numberGame.isnumeric())):
            numberGame = input("Choisir le nombre de partie : ")

        return numberGame

    # Call the main functions of the game
    def main(self):
        numberGame = int(self.menu())
        while numberGame>0:
            player1 = self.player1
            player2 = self.player2
            if random.random() > 0.5: # which player makes the first move
                self.setPlayer1(player1)
                self.setPlayer2(player2)
            else:
                self.setPlayer1(player2)
                self.setPlayer2(player1)

            self.game()
            board.resetBoard()
            numberGame -= 1

        # Call the graphic functions
        winsLosesDrawsPieGraphic([self.getPlayer1().getWins(), self.getPlayer1().getLoses(), self.getPlayer1().getDraws()], self.getPlayer1().getName())
        
        firstAndSecondMoveGraphic(self.getFirstMove(), self.getSecondMove())
       
        victoryTypePieGraphic(self.board.getVictoryType())
        
        learningValueGraphic(self.player1.getLearningList(), self.player1.getName())
        learningValueGraphic(self.player2.getLearningList(), self.player2.getName())
        
        print("Fin des parties")

boardTemplate = np.array([
    [' ',' ',' '],
    [' ',' ',' '],
    [' ',' ',' '],
]
);


board = Board(boardTemplate)
game = Game(board)
game.main()
