#Authors: Maxwell McAtee and William Lau
#Unused backpropogate, initializeWeights, and original utility at bottom of class.
import random
import sys
import unittest
sys.path.append("..")  #so other modules can be found in parent dir
from Player import *
from Constants import *
from Construction import *
from Ant import UNIT_STATS
from Move import Move
from GameState import *
from AIPlayerUtils import *
import math
import numpy as np
import csv
import os

##
#AIPlayer
#Description: The responsbility of this class is to interact with the game by
#deciding a valid move based on a given game state. This class has methods that
#will be implemented by students in Dr. Nuxoll's AI course.
#
#Variables:
#   playerId - The id of the player.
##
class AIPlayer(Player):

    #__init__
    #Description: Creates a new Player
    #
    #Parameters:
    #   inputPlayerId - The id to give the new player (int)
    #   cpy           - whether the player is a copy (when playing itself)
    ##
    def __init__(self, inputPlayerId):
        super(AIPlayer,self).__init__(inputPlayerId, "Despair")

        #Commented out for use of fixed weights
        # self.weightsL1, self.weightsL2 = self.initalizeWeights()

        #Weights for layers 1 and 2
        self.weightsL1 = [[-0.7149747391711163, -0.36440668777083457, -0.21441092724320568, 
                            0.03173104126048873, -2.6718293371837363], [-1.0931214608662787, 
                            -1.1407049381704568, -0.5031158837048424, -0.24692894844283264, 
                            -1.1286539814121164], [0.871651096867637, 0.9193101442329813, 
                            0.2356307812198628, -0.143916371219241, -0.11079975509644276], 
                            [0.3894457133886055, 0.4283607927278694, 0.7329146556514127, 
                            1.2445058202317938, 1.4593272774906245], [0.9795279576558774, 
                            1.5440453398461027, 1.192984764261922, 0.7817727730459207, 
                            2.5885247677519043], [-0.6572646294152098, 1.191925256326022, 
                            -0.580946198219355, 0.44848792883481214, 1.2428874680731117], 
                            [-0.6401790886430343, -1.3029374480512725, 0.4528034887209836, 
                            0.26237051226372377, -0.9944369890474433], [-0.06550469607511328, 
                            -1.4564837334355867, -1.1807196254792032, 0.39056983088625946, 
                            -1.409039025588328]]
        self.weightsL2 = [-0.35424838355784194, -2.9075868955547013, -2.227585019213028, 
                            0.24300213253606973, 1.6923780827126367, 3.1273868644050107, 
                            0.9876967683007722, -1.413989561067693, -2.184446284568322]

        #Not used for fixed weight version.
        self.currentTarget = 0
        self.error = 1

    ##
    # calculateResult
    #
    # Description: calculate the result of a run of the neural network
    #
    # Parameters: 
    #   inputL1 - the inputs to layer one
    #
    # Return:
    #   result - the result of the neural network
    #   inputL2 - the inputs for the second layer
    ##
    def calculateResult(self, inputL1):
        weightsL1 = np.array(self.weightsL1)
        weightsL2 = np.array(self.weightsL2)
        inputL1 = np.array(inputL1)

        product = weightsL1.dot(inputL1)

        #calculate the result of the first layer
        inputL2 = []
        inputL2.append(1)
        for i in range(len(product)):
            inputL2.append(1/(1 + math.e ** (-product[i])))

        product = weightsL2.dot(inputL2)

        #result of the second layer
        result = (1/(1 + math.e ** (-product)))

        return result, inputL2

    ##
    # network
    #
    # Description: Given a gamestate it determines the utility of that state using
    #              a neural network.
    #
    # Parameters:
    #   currentState - gamestate to be evaluated
    #
    # Return:
    #   result - the result of the neural network's computation.
    def network(self, currentState):
        # Get inventories.
        myInv = getCurrPlayerInventory(currentState)

        if myInv.player == 0:
            enemyInv = currentState.inventories[1]
        else:
            enemyInv = currentState.inventories[0]

        # Get important pieces.
        myFood = myInv.foodCount
        myFoodPlacement = getCurrPlayerFood(self, currentState)
        enemyFood = enemyInv.foodCount
        myAnts = myInv.ants
        enemyAnts = enemyInv.ants
        myCombatAnts = getAntList(currentState, myInv.player, (DRONE, SOLDIER, R_SOLDIER))
        enemyCombatAnts = getAntList(currentState, enemyInv.player, (DRONE, SOLDIER, R_SOLDIER))
        myWorkerAnts = getAntList(currentState, myInv.player, (WORKER,))
        enemyWorkerAnts = getAntList(currentState, enemyInv.player, (WORKER,))
        myQueen = myInv.getQueen()
        enemyQueen = enemyInv.getQueen()
        myAnthill = myInv.getAnthill()
        myTunnel = myInv.getTunnels()[0]
        enemyAnthill = enemyInv.getAnthill()

        inputL1 = []

        # Adjust weight based on various factors.
        inputL1.append(self.getFoodWeightAdjustment(myFood, enemyFood))
        inputL1.append(self.getAntWeightAdjustment(myAnts, myCombatAnts, enemyAnts, 
                                                    enemyCombatAnts))
        inputL1.append(self.getHealthWeightAdjustment(myQueen, myAnthill, enemyQueen, 
                                                        enemyAnthill))
        inputL1.append(self.getFoodProxWeightAdjustment(myAnts, myWorkerAnts, myAnthill, 
                                                        myTunnel, myFoodPlacement))
        inputL1.append(self.getCombatDistanceWeightAdjustment(myCombatAnts, enemyWorkerAnts, 
                                                        enemyQueen))

        result, inputL2 = self.calculateResult(inputL1)

        #Unused for fixed weights, but left in for clarity.
        error = self.currentTarget - result
        self.error = error
        self.backpropogate(inputL1, inputL2, error)

        # Automatically set good or bad weights depending on certain conditions.
        if len(myWorkerAnts) > 1:
            result = -1000
        
        if enemyQueen is None:
            result = 1000

        if myQueen.coords == myAnthill.coords or approxDist(myQueen.coords, myAnthill.coords) > 1:
            result = -1000

        for enemyAnt in enemyCombatAnts:
            if(approxDist(myQueen.coords,enemyAnt.coords) == 1):
                result = -1000

        return result

    # getCombatDistanceWeightAdjustment
    # Adjusts weight for utility() based on distance of combat ants from enemy units.
    #
    # Parameters:
    #   myCombatAnts: the amount of combat ants this player has.
    #   enemyWorkerAnts: the enemy workers.
    #   enemyQueen: the enemy queen.
    #
    # Return: amount to adjust weight by.
    ##
    def getCombatDistanceWeightAdjustment(self, myCombatAnts, enemyWorkerAnts, enemyQueen):
        adjustment = 0
        enemyWorkerWeight = 0
        enemyQueenWeight = 0

        if len(myCombatAnts) > 0:
            for combatant in myCombatAnts:
                for worker in enemyWorkerAnts:
                    distToWorker = approxDist(combatant.coords, worker.coords)
                    enemyWorkerWeight += distToWorker

                if enemyQueen is not None:
                    distToQueen = approxDist(combatant.coords, enemyQueen.coords)
                else:
                    distToQueen = 0

                enemyQueenWeight += distToQueen
            adjustment += 0.1 - (0.005 * enemyWorkerWeight)
            adjustment += 0.1 - (0.005 * enemyQueenWeight)
        return adjustment

    # getFoodProxWeightAdjustment
    # Adjusts weight for utility() based on distance of workers from food/structures.
    #
    # Parameters:
    #   myAnts: the amount of ants this player has.
    #   myWorkerAnts: the amount of worker ants this player has.
    #   myAnthill: this player's anthill.
    #   myTunnel: this player's tunnel.
    #   myFoodPlacement: where this player's food is.
    #
    # Return: amount to adjust weight by.
    ##
    def getFoodProxWeightAdjustment(self, myAnts, myWorkerAnts, myAnthill, myTunnel, 
                                        myFoodPlacement):
        adjustment = 0
        sum = 0
        hillCovered = False

        for ant in myAnts:
            if ant.coords == myAnthill.coords:
                hillCovered = True

        for worker in myWorkerAnts:
            if worker.carrying:
                tunnelDist = approxDist(worker.coords, myTunnel.coords)
                anthillDist = approxDist(worker.coords, myAnthill.coords)
                if tunnelDist < anthillDist or hillCovered:
                    sum += (.005 * tunnelDist)
                else:
                    sum += (.005 * anthillDist)

                adjustment += 0.015
            elif not worker.carrying:
                foodOne = myFoodPlacement[0]
                foodTwo = myFoodPlacement[1]

                foodOneDist = approxDist(worker.coords, foodOne.coords)
                foodTwoDist = approxDist(worker.coords, foodTwo.coords)

                if foodOneDist < foodTwoDist:
                    sum += (.004 * foodOneDist)
                else:
                    sum += (.004 * foodTwoDist)

            adjustment += (0.08 - (sum / len(myWorkerAnts)))
        return adjustment

    # getFoodWeightAdjustment
    # Adjusts weight for utility() based on food factors.
    #
    # Parameters:
    #   myFood: the amount of food this player has.
    #   enemyFood: the amount of food the opponent has.
    #
    # Return: amount to adjust weight by.
    ##
    def getFoodWeightAdjustment(self, myFood, enemyFood):
        difference = myFood - enemyFood

        # Increment based on who has more food.
        if difference > 0:
            adjustment = difference * .04
        else:
            adjustment = difference * .025
        return adjustment

    # getAntWeightAdjustment
    # Adjusts weight for utility() based on ant factors.
    #
    # Parameters:
    #   myAnts: the amount of ants this player has.
    #   myCombatAnts: the amount of ants this player has for combat.
    #   enemyAnts: the amount of ants the opponent has.
    #   enemyCombatAnts: the amount of ants the opponent has for combat.
    #
    # Return: amount to adjust weight by.
    ##
    def getAntWeightAdjustment(self, myAnts, myCombatAnts, enemyAnts, enemyCombatAnts):
        adjustment = 0  # Start with no adjustment.

        # Adjust rating based on who has more ants.
        if len(myAnts) > len(enemyAnts):
            adjustment += 0.06

        # Adjust rating based on who has more combat ants.
        if len(myCombatAnts) > len(enemyCombatAnts):
            adjustment += 0.06

        return adjustment

    # getHealthWeightAdjustment
    # Adjusts weight for utility() based on health factors.
    #
    # Parameters:
    #   myQueen: the queen this player has.
    #   myAnthill: the anthill this player has.
    #   enemyQueen: the queen the opponent has.
    #   enemyAnthill: the anthill the opponent has.
    #
    # Return: amount to adjust weight by.
    ##
    def getHealthWeightAdjustment(self, myQueen, myAnthill, enemyQueen, enemyAnthill):
        adjustment = 0  # Start with no adjustment.

        if enemyQueen is not None:
            queenHealthDifference = myQueen.health - enemyQueen.health
        else:
            queenHealthDifference = myQueen.health
        anthillHealthDifference = myAnthill.captureHealth - enemyAnthill.captureHealth

        # Adjust rating based on whose queen has more health.
        if queenHealthDifference > 0:
            adjustment += 0.05
        elif queenHealthDifference < 0:
            adjustment -= 0.05

        # Adjust rating based on whose anthill has more health.
        if anthillHealthDifference > 0:
            adjustment += 0.05
        elif anthillHealthDifference < 0:
            adjustment -= 0.05

        return adjustment

    # bestMove
    # Description: Returns a best node.
    #
    # Parameters:
    #   list: the list of nodes.
    ##
    def bestMove(self, nodeList):
        evalList = []
        nodesWithMaxEvals = []

        for node in nodeList:
            nodeEval = node["evaluation"] - node["depth"]
            evalList.append(nodeEval)

        maxValue = max(evalList)

        index = 0

        for i in evalList:
            if i == maxValue:
                nodesWithMaxEvals.append(nodeList[index])
            index += 1

        return nodesWithMaxEvals[random.randint(0, len(nodesWithMaxEvals) - 1)]

    ##
    #getPlacement
    #
    #Description: called during setup phase for each Construction that
    #   must be placed by the player.  These items are: 1 Anthill on
    #   the player's side; 1 tunnel on player's side; 9 grass on the
    #   player's side; and 2 food on the enemy's side.
    #
    #Parameters:
    #   construction - the Construction to be placed.
    #   currentState - the state of the game at this point in time.
    #
    #Return: The coordinates of where the construction is to be placed
    ##
    def getPlacement(self, currentState):
        numToPlace = 0
        #implemented by students to return their next move
        if currentState.phase == SETUP_PHASE_1:    #stuff on my side
            numToPlace = 11
            moves = []
            for i in range(0, numToPlace):
                move = None
                while move == None:
                    #Choose any x location
                    x = random.randint(0, 9)
                    #Choose any y location on your side of the board
                    y = random.randint(0, 3)
                    #Set the move if this space is empty
                    if currentState.board[x][y].constr == None and (x, y) not in moves:
                        move = (x, y)
                        #Just need to make the space non-empty. So I threw whatever I felt 
                        # like in there.
                        currentState.board[x][y].constr == True
                moves.append(move)
            return moves
        elif currentState.phase == SETUP_PHASE_2:   #stuff on foe's side
            numToPlace = 2
            moves = []
            for i in range(0, numToPlace):
                move = None
                while move == None:
                    #Choose any x location
                    x = random.randint(0, 9)
                    #Choose any y location on enemy side of the board
                    y = random.randint(6, 9)
                    #Set the move if this space is empty
                    if currentState.board[x][y].constr == None and (x, y) not in moves:
                        move = (x, y)
                        #Just need to make the space non-empty. So I threw whatever I felt
                        #  like in there.
                    
                        currentState.board[x][y].constr == True
                moves.append(move)
            return moves
        else:
            return [(0, 0)]
    
    ##
    #getMove
    #Description: Gets the next move from the Player.
    #
    #Parameters:
    #   currentState - The state of the current game waiting for the player's move (GameState)
    #
    #Return: The Move to be made
    ##
    def getMove(self, currentState):
        moves = listAllLegalMoves(currentState)
        states = []
        nodes = []

        for move in moves:
            states.append(getNextState(currentState, move))

        i = 0  # Keep track of index.
        for move in moves:
            nodes.append(self.createNode(move, states[i]))
            self.network(nodes[i]["state"])
            i += 1

        bestNode = self.bestMove(nodes)
        return bestNode['moveToReach']

    ##
    # createNode
    # Description: Creates a node with 5 values.
    #
    # Parameters:
    #   move: the move to get to the state.
    #   gameState: the current state.
    #
    # Return: the completed node.
    ##
    def createNode(self, move, gameState):
        # Represents a node.
        node = {
            "moveToReach": move,
            "state": gameState,
            "depth": 1,  # Arbitrary value, for part A.
            "evaluation": self.network(gameState) + 1,
            "parent": None  # Arbitrary value, not used for now.
        }

        return node
    
    ##
    #getAttack
    #Description: Gets the attack to be made from the Player
    #
    #Parameters:
    #   currentState - A clone of the current state (GameState)
    #   attackingAnt - The ant currently making the attack (Ant)
    #   enemyLocation - The Locations of the Enemies that can be attacked (Location[])
    ##
    def getAttack(self, currentState, attackingAnt, enemyLocations):
        #Attack a random enemy.
        return enemyLocations[random.randint(0, len(enemyLocations) - 1)]

    ##
    #registerWin
    #
    # This agent doens't learn
    #
    def registerWin(self, hasWon):
        #method templaste, not implemented
        pass

    ##
    # initializeWeights
    #
    # Description: initialize the weights in the network to random values between
    #              -1 and 1.
    #              Currently Unused
    #
    # Parameters: None
    #
    # Return:
    #   initialWeights - weights of for the first layer
    #   temp - weights for the second layer
    ##
    def initalizeWeights(self):
        initialWeights = []
        temp = []
        #first layer weights
        for i in range(8):
            for j in range(5):
                temp.append((random.uniform(-1.0,1.0)))
            initialWeights.append(temp)
            temp = []

        #second layer weights
        for i in range(9):
            temp.append((random.uniform(-1.0, 1.0)))


        return initialWeights, temp
        
    # utility
    # Description: Returns a number from 0 to 1 based on the current state of a game.
    #              Currently not used.
    #
    # Parameters:
    #   currentState: the state of the game.
    ##
    def utility(self, currentState):
        weight = 0.1  # Start with "neutral" value

        # Get inventories.
        myInv = getCurrPlayerInventory(currentState)

        if myInv.player == 0:
            enemyInv = currentState.inventories[1]
        else:
            enemyInv = currentState.inventories[0]

        # Get important pieces.
        myFood = myInv.foodCount
        myFoodPlacement = getCurrPlayerFood(self, currentState)
        enemyFood = enemyInv.foodCount
        myAnts = myInv.ants
        enemyAnts = enemyInv.ants
        myCombatAnts = getAntList(currentState, myInv.player, (DRONE, SOLDIER, R_SOLDIER))
        enemyCombatAnts = getAntList(currentState, enemyInv.player, (DRONE, SOLDIER, R_SOLDIER))
        myWorkerAnts = getAntList(currentState, myInv.player, (WORKER,))
        enemyWorkerAnts = getAntList(currentState, enemyInv.player, (WORKER,))
        myQueen = myInv.getQueen()
        enemyQueen = enemyInv.getQueen()
        myAnthill = myInv.getAnthill()
        myTunnel = myInv.getTunnels()[0]
        enemyAnthill = enemyInv.getAnthill()

        # Adjust weight based on various factors.
        weight += self.getFoodWeightAdjustment(myFood, enemyFood)
        weight += self.getAntWeightAdjustment(myAnts, myCombatAnts, enemyAnts, enemyCombatAnts)
        weight += self.getHealthWeightAdjustment(myQueen, myAnthill, enemyQueen, enemyAnthill)
        weight += self.getFoodProxWeightAdjustment(myAnts, myWorkerAnts, myAnthill, myTunnel, 
                                                    myFoodPlacement)
        weight += self.getCombatDistanceWeightAdjustment(myCombatAnts, enemyWorkerAnts, enemyQueen)

        self.currentTarget = weight

        # Automatically set good or bad weights depending on certain conditions.
        if len(myWorkerAnts) > 1:
            weight = 0
        
        if enemyQueen is None:
            weight = 1

        if myQueen.coords == myAnthill.coords or approxDist(myQueen.coords, myAnthill.coords) > 1:
            weight = 0

        for enemyAnt in enemyCombatAnts:
            if(approxDist(myQueen.coords,enemyAnt.coords) == 1):
                weight = 0

        # In case weight goes out of bounds.
        if weight <= 0:
            weight = 0.01
        elif weight >= 1:
            weight = 0.99
        return weight
    
    ##
    # backpropogate
    #
    # Description: Method used to backpropogate error in the neural network.
    #              Currently not used.
    #
    # Parameters: 
    #   inputL1 - input values for the first layer
    #   inputl2 - input values for the second layer
    #   finalError - final error that resulted from target minus actual result
    #
    # Return:
    #   weightsL1 - updated weights for first layer
    #   weightsL2 - updated weights for second layer
    ##
    def backpropogate(self, inputL1, inputL2, finalError):
        result = self.currentTarget
        weightsL1 = self.weightsL1
        weightsL2 = self.weightsL2
        i = 0
        finalSlope = (result * (1 - result))
        alpha = 0.27

        #error term from output node
        errorTerm = finalError * finalSlope

        #calculate error for the hidden nodes
        hiddenErrors = []
        for weight in weightsL2:
            hiddenErrors.append(weight * errorTerm)

        #update weights in the second layer
        for weight in weightsL2:
            weightsL2[i] = weight + (alpha * finalError * finalSlope * inputL2[i])
            i += 1

        #update weights for the first layer
        i = 0
        j = 0
        slope = 0
        for perceptron in weightsL1:
            slope = (inputL2[i+1] * (1 - inputL2[i+1]))
            for weight in perceptron:
                perceptron[j] = weight + (alpha * hiddenErrors[i+1] * slope * inputL1[j])
                j += 1
            
            weightsL1[i] = perceptron
            i += 1
            j = 0

        self.weightsL1 = weightsL1
        self.weightsL2 = weightsL2


class TestCreateNode(unittest.TestCase):
    

    def test_createNode(self):
        player = AIPlayer(0)

        move = Move(MOVE_ANT, [(1,2), 0], None)
        gameState = GameState.getBasicState()

        p1Food1 = Building((1, 1), FOOD, 0)
        p1Food2 = Building((8, 0), FOOD, 0)
        gameState.board[1][1] = p1Food1
        gameState.board[0][8] = p1Food2
        gameState.inventories[2].constrs += [p1Food1, p1Food2]

        p1Food1 = Building((8, 8), FOOD, 1)
        p1Food2 = Building((0, 8), FOOD, 1)
        gameState.board[8][8] = p1Food1
        gameState.board[8][0] = p1Food2
        gameState.inventories[2].constrs += [p1Food1, p1Food2]
        
        node = {
            "moveToReach": move,
            "state": gameState,
            "depth": 1,
            "evaluation": player.utility( gameState) + 1,
            "parent": None
        }
        
        self.assertEqual((player.createNode(move,  gameState)), node)

    def test_bestMove(self):
        node1 = {
            "moveToReach": None,
            "state": None,
            "depth": 1,
            "evaluation": 3,
            "parent": None
        }

        node2 = {
            "moveToReach": None,
            "state": None,
            "depth": 1,
            "evaluation": 2,
            "parent": None
        }
        list = [node1, node2]
        self.assertEqual(AIPlayer.bestMove(AIPlayer, list), node1)

    def test_utility(self):
        player = AIPlayer(0)
        gameState = GameState.getBasicState()

        p1Food1 = Building((1, 1), FOOD, 0)
        p1Food2 = Building((8, 0), FOOD, 0)
        gameState.board[1][1] = p1Food1
        gameState.board[0][8] = p1Food2
        gameState.inventories[2].constrs += [p1Food1, p1Food2]

        p1Food1 = Building((8, 8), FOOD, 1)
        p1Food2 = Building((0, 8), FOOD, 1)
        gameState.board[8][8] = p1Food1
        gameState.board[8][0] = p1Food2
        gameState.inventories[2].constrs += [p1Food1, p1Food2]

        self.assertEqual(player.utility(gameState), 0.01)
if __name__ == '__main__':
    unittest.main()


