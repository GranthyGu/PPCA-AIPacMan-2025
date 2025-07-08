# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood_ = successorGameState.getFood()
        newFood = newFood_.asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        minimal_dis = float("inf")
        for food in newFood:
            minimal_dis = min(minimal_dis, manhattanDistance(food, newPos))
        ghost_value = 0
        for i in range(len(newGhostStates)):
            time_ = newScaredTimes[i]
            state = newGhostStates[i].getPosition()
            dis = manhattanDistance(state, newPos)
            if time_ == 0 and dis < 2:
                ghost_value -= 50
            elif time_ > 2:
                ghost_value += (100 * time_) / (dis + 1)
        return successorGameState.getScore() + ghost_value + (1.0 / (minimal_dis + 1))

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        def alphabeta(state, depth, alpha, beta, index):
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            num_agents = state.getNumAgents()
            next_agent = (index + 1) % num_agents
            next_depth = depth + 1 if next_agent == 0 else depth
            if index == 0:
                max_eval = float('-inf')
                for action in state.getLegalActions(0):
                    successor = state.generateSuccessor(0, action)
                    eval = alphabeta(successor, next_depth, alpha, beta, next_agent)
                    max_eval = max(max_eval, eval)
                    alpha = max(alpha, eval)
                return max_eval
            else:
                min_eval = float('inf')
                for action in state.getLegalActions(index):
                    successor = state.generateSuccessor(index, action)
                    eval = alphabeta(successor, next_depth, alpha, beta, next_agent)
                    min_eval = min(min_eval, eval)
                    beta = min(beta, eval)
                return min_eval
        best_score = float('-inf')
        best_action = None
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            value = alphabeta(successor, 0, float('-inf'), float('inf'), 1)
            if value > best_score:
                best_score = value
                best_action = action
        return best_action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def alphabeta(state, depth, agentIndex, alpha, beta):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            if agentIndex == 0:
                return max_value(state, depth, agentIndex, alpha, beta)
            else:
                return min_value(state, depth, agentIndex, alpha, beta)
        def max_value(state, depth, agentIndex, alpha, beta):
            v = -float('inf')
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                nextAgent = (agentIndex + 1) % state.getNumAgents()
                nextDepth = depth if nextAgent != 0 else depth + 1
                v = max(v, alphabeta(successor, nextDepth, nextAgent, alpha, beta))
                if v > beta:
                    return v
                alpha = max(alpha, v)
            return v
        def min_value(state, depth, agentIndex, alpha, beta):
            v = float('inf')
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                nextAgent = (agentIndex + 1) % state.getNumAgents()
                nextDepth = depth if nextAgent != 0 else depth + 1
                v = min(v, alphabeta(successor, nextDepth, nextAgent, alpha, beta))
                if v < alpha:
                    return v
                beta = min(beta, v)
            return v
        bestAction = None
        bestValue = -float('inf')
        alpha = -float('inf')
        beta = float('inf')
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            value = alphabeta(successor, 0, 1, alpha, beta)
            if value > bestValue:
                bestValue = value
                bestAction = action
            alpha = max(alpha, value) # Quite IMPORTANT! Forget it in the first time!
        return bestAction


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expect(state, depth, index):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            elif index == 0:
                return max_value(state, depth, 0)
            else:
                return exp_value(state, depth, index)
        def max_value(state, depth, agentIndex):
            v = -float('inf')
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                nextAgent = (agentIndex + 1) % state.getNumAgents()
                nextDepth = depth if nextAgent != 0 else depth + 1
                v = max(v, expect(successor, nextDepth, nextAgent))
            return v
        def exp_value(state, depth, agentIndex):
            num = len(state.getLegalActions(agentIndex))
            sum = 0.0
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                nextAgent = (agentIndex + 1) % state.getNumAgents()
                nextDepth = depth if nextAgent != 0 else depth + 1
                sum += expect(successor, nextDepth, nextAgent)
            return sum /num

        bestAction = None
        bestValue = -float('inf')
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            value = expect(successor, 0, 1)
            if value > bestValue:
                bestValue = value
                bestAction = action
        return bestAction

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    1. <TODO>
    2.
    3.
    4.

    """
    "*** YOUR CODE HERE ***"
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    capsules = currentGameState.getCapsules()
    score = currentGameState.getScore()
    foodList = food.asList()
    if foodList:
        minFoodDist = min([manhattanDistance(pos, food) for food in foodList])
        score += 20.0 / (minFoodDist + 1)
    score -= 3 * len(foodList)
    for i, ghostState in enumerate(ghostStates):
        ghostPos = ghostState.getPosition()
        ghostDist = manhattanDistance(pos, ghostPos)
        if scaredTimes[i] > 0:
            if ghostDist == 0:
                score += 300  # Bonus for eating ghost
            else:
                score += 100.0 / (ghostDist + 1)
        else:
            if ghostDist <= 1:
                score -= 500
            else:
                score += ghostDist
    if capsules:
        minCapsuleDist = min([manhattanDistance(pos, capsule) for capsule in capsules])
        score += 20.0 / (minCapsuleDist + 1)
    score -= 2 * len(capsules)
    return score

# Abbreviation
better = betterEvaluationFunction
