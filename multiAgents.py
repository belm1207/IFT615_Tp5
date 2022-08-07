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


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
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
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        ============================================================
        QUESTION 1

        Vous devez compléter cette méthode afin d'améliorer l'évaluation de l'action
        donnée en paramètre par rapport à l'état actuel (donné en paramètre également).

        GameState.getScore() retourne simplement le score prévu à l'état, le score affiché
        dans l'interface.
        ============================================================

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
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        if successorGameState.isWin():
            return float("inf")

        newFoodDistance = [manhattanDistance(food, newPos) for food in newFood.asList()]

        prevFoodDistance = [manhattanDistance(food, currentGameState.getPacmanPosition()) for food in
                            currentGameState.getFood().asList()]

        newGhostDistance = [manhattanDistance(ghost.getPosition(), newPos) for ghost in newGhostStates]

        prevGhostDistance = [manhattanDistance(ghost.getPosition(), newPos) for ghost in
                             currentGameState.getGhostStates()]

        score = 0
        score += successorGameState.getScore() - currentGameState.getScore()

        if action == Directions.STOP:
            score -= 10

        if newPos in currentGameState.getCapsules():
            score += 250

        if len(newFoodDistance) < len(currentGameState.getFood().asList()):
            score += 200

        if newPos in currentGameState.getFood().asList():
            score += 200

        score -= 15 * len(newFood.asList())

        if min(newFoodDistance) < min(prevFoodDistance):
            score += 100

        if sum(newScaredTimes) > 0:
            if min(prevGhostDistance) < min(newGhostDistance):
                score += 200
            else:
                score -= 100
        else:
            if min(prevGhostDistance) < min(newGhostDistance):
                score -= 100
            else:
                score += 200

        # print("actual score : " + str(successorGameState.getScore()))
        # print("ghost distance : " + str(ghostDistance))
        # print("food distance : " + str(foodDistance))
        # print("foodDistance: " + str(foodDistance))
        print("score", score)
        print("-----------------------------------------------------")
        return score


def scoreEvaluationFunction(currentGameState):
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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        ============================================================
        QUESTION 2

        Vous devez compléter cette méthode afin d'implémenter le choix de l'action selon
        l'algorithme minimax.

        Puisqu'il vous est demandé d'arrêter la recherche à une profondeur maximale donnée (self.depth),
        vous devez utiliser la fonction d'évaluation de l'agent self.evaluationFunction().
        ============================================================

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

        def maxValue(gameState, depth):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)
            maxV = -float("inf")
            for action in gameState.getLegalActions(0):
                maxV = max(maxV, minValue(gameState.generateSuccessor(0, action), depth, 1))
            return maxV

        def minValue(gameState, depth, ghostIndex):
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            minV = float("inf")
            for action in gameState.getLegalActions(ghostIndex):
                if ghostIndex == gameState.getNumAgents() - 1:
                    minV = min(minV, maxValue(gameState.generateSuccessor(ghostIndex, action), depth - 1))
                else:
                    minV = min(minV, minValue(gameState.generateSuccessor(ghostIndex, action), depth, ghostIndex + 1))
            return minV

        pacManLegalMove = gameState.getLegalActions(0)
        maxScore = -float("inf")
        bestAction = ''

        for action in pacManLegalMove:
            nextState = gameState.generateSuccessor(0, action)
            score = minValue(nextState, self.depth, 1)
            if score > maxScore:
                maxScore = score
                bestAction = action

        return bestAction


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        ============================================================
        QUESTION 3

        Vous devez compléter cette méthode afin d'implémenter le choix de l'action selon
        l'algorithme alpha-beta pruning.
        ============================================================

        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def maxValue(gameState, alpha, beta, depth):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)
            v = -float("inf")
            tempAlpha = alpha
            for action in gameState.getLegalActions(0):
                v = max(v, minValue(gameState.generateSuccessor(0, action), tempAlpha, beta, depth, 1))
                if v > beta:
                    return v
                tempAlpha = max(tempAlpha, v)
            return v

        def minValue(gameState, alpha, beta, depth, agentIndex):
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            v = float("inf")
            tempBeta = beta
            for action in gameState.getLegalActions(agentIndex):
                if agentIndex == gameState.getNumAgents() - 1:
                    v = min(v, maxValue(gameState.generateSuccessor(agentIndex, action), alpha, tempBeta, depth - 1))
                    if v < alpha:
                        return v
                    tempBeta = min(tempBeta, v)
                else:
                    v = min(v, minValue(gameState.generateSuccessor(agentIndex, action), alpha, tempBeta, depth,
                                        agentIndex + 1))
                    if v < alpha:
                        return v
                    tempBeta = min(tempBeta, v)
            return v

        pacManLegalMove = gameState.getLegalActions(0)
        maxScore = -float("inf")
        bestAction = ''
        alpha = -float("inf")
        beta = float("inf")
        for action in pacManLegalMove:
            nextState = gameState.generateSuccessor(0, action)
            score = minValue(nextState, alpha, beta, self.depth, 1)
            if score > maxScore:
                maxScore = score
                bestAction = action
            if score > beta:
                return bestAction
            alpha = max(alpha, score)
        return bestAction


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        ============================================================
        QUESTION 4

        Vous devez compléter cette méthode afin d'implémenter le choix de l'action selon
        l'algorithme expectimax.
        ============================================================

        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        def maxValue(gameState, depth):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)
            maxV = -float("inf")
            for action in gameState.getLegalActions(0):
                maxV = max(maxV, expectValue(gameState.generateSuccessor(0, action), depth, 1))
            return maxV

        def expectValue(gameState, depth, agentIndex):
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            v = 0
            for action in gameState.getLegalActions(agentIndex):
                if agentIndex == gameState.getNumAgents() - 1:
                    v += maxValue(gameState.generateSuccessor(agentIndex, action), depth - 1)
                else:
                    v += expectValue(gameState.generateSuccessor(agentIndex, action), depth, agentIndex + 1)
            return v / len(gameState.getLegalActions(agentIndex))

        pacManLegalMove = gameState.getLegalActions(0)
        maxScore = -float("inf")
        bestAction = ''
        for action in pacManLegalMove:
            nextState = gameState.generateSuccessor(0, action)
            score = expectValue(nextState, self.depth, 1)
            if score > maxScore:
                maxScore = score
                bestAction = action
        return bestAction


def betterEvaluationFunction(currentGameState):
    """
    ============================================================
    QUESTION 5

    Vous devez compléter cette méthode afin d'évaluer l'état donné en
    paramètre.
    ============================================================

    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).
    """

    "*** VOTRE EXPLICATION ICI ***"

    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
