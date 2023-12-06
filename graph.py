# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 21:26:45 2023

@author: dadag
"""

import itertools
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def compter_symboles(grille):
    nombre_X, nombre_O = map(lambda s: sum(l.count(s) for l in grille), ['X', 'O'])
    return nombre_X + nombre_O

def isGoodGrille(grille):
    nombre_X, nombre_O = map(lambda s: sum(l.count(s) for l in grille), ['X', 'O'])
    if (nombre_X + 1 == nombre_O or nombre_X == nombre_O + 1):
        return True
    if (nombre_X == nombre_O):
        return True
    if (nombre_X == nombre_O == 0):
        return True
    winO = isWin(grille, 'X')
    winX = isWin(grille, 'O')
    if (winO != winX):
        return True
    
    return False

def whoPlay(grille):
    nombre_X, nombre_O = map(lambda s: sum(l.count(s) for l in grille), ['X', 'O'])
    if (nombre_X < nombre_O):
        return 'X'
    else:
        return 'O'

def isWin(state, item):
    for row in state:
        if (row[0] == item and row[1] == item and row[2] == item):
            return True
    # vertical
    for i in range(3):
        if (state[0][i] == item and state[1][i] == item and state[2][i] == item):
            return True
    # diagonal
    if (state[0][0] == item and state[1][1] == item and state[2][2] == item):
        return True
    if (state[2][0] == item and state[1][1] == item and state[0][2] == item):
        return True

def whoWin(state):
    # horizontal
    for row in state:
        if (row[0]!= ' ' and row[0] == row[1] == row[2]):
            return True
    # vertical
    for i in range(3):
        if (state[0][i]!= ' ' and state[0][i] == state[1][i] == state[2][i]):
            return True
    # diagonal
    if (state[0][0]!= ' ' and state[0][0] == state[1][1] ==  state[2][2]):
        return True
    if (state[2][0]!= ' ' and state[2][0] == state[1][1] == state[0][2]):
        return True

    return False


def combinations(board):
    emptyCellsArray = emptyCells(board)
    arrayOfNextStateId = []

    boardTemplate = np.copy(board)
    actualBoardClass = np.copy(board)

    for i in range(len(emptyCellsArray)):
        actualBoardClass = addItem(emptyCellsArray[i], whoPlay(board), board)
        arrayOfNextStateId.append(list(allCombinations.keys())[list(allCombinations.values()).index(actualBoardClass)])
        actualBoardClass = boardTemplate

    return arrayOfNextStateId

def addItem(position, item, board):
    x = int(position[0])
    y = int(position[1])
    board[x][y] = item
    return board

def emptyCells(board):
    emptyCells = np.array([])
    for x in range(len(board)):
        for y in range (len(board[0])):
            if board[x][y] == ' ':
                emptyCells = np.append(emptyCells,str(x)+str(y))
    return emptyCells


player = ['X', 'O', ' ']
states_dict = {}
all_possible_states = [[list(i[0:3]), list(i[3:6]), list(i[6:9])] for i in itertools.product(player, repeat=9)]


n_states = len(all_possible_states)
allCombinations = {}
for state in range (n_states):
    allCombinations[state] = all_possible_states[state]

file = np.loadtxt('trained_state_values_X.txt', dtype=np.float64)
graph = {}

for idx, state in enumerate(all_possible_states):
    
    if( not isGoodGrille(state)):
        continue
    states_dict[idx] = state
    if(compter_symboles(state) < 5 and not (whoWin(state))):  # On ne vérifie pas ces next states
        continue

    listNextStates = combinations(state)
    dictNextStates = {}
    for nextState in listNextStates:
        dictNextStates[nextState] = {'weight':file[nextState]}
    graph[idx] = dictNextStates




G = nx.DiGraph(graph)
pos=nx.spring_layout(G, k=0.15)                                      
nx.draw(G, with_labels=True,pos=pos)
labels = nx.get_edge_attributes(G,'weight')
nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
plt.show()
print("--- fin ---")
