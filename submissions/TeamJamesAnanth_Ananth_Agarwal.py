# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html
from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from inference import inferenceByVariableElimination
from config import LEGAL_POWERS
from util import nearestPoint
from capsule import ScareCapsule, GrenadeCapsule, ArmourCapsule, SonarCapsule, JuggernautCapsule
import contestPowersBayesNet
import inference
#################
# Team creation #
#################
class Team:
    def createAgents(self, firstIndex, secondIndex, isRed, gameState,
                     first='OffensiveReflexAgent', second='DefensiveReflexAgent'):
        self.isRed = isRed
        return [eval(first)(firstIndex, gameState), eval(second)(secondIndex, gameState)]
    def chooseCapsules(self, gameState, capsuleLimit, opponentEvidences):
        net = contestPowersBayesNet.powersBayesNet()
        leftCapsules = []
        rightCapsules = []
        width, height = gameState.data.layout.width, gameState.data.layout.height
        for i in range(len(opponentEvidences)):
            jointLaserInvisibility = inference.inferenceByVariableElimination(net, ['laser', 'invisibility'],
                                                                              opponentEvidences[i], None)
            strong_laser_prob = 0
            weak_laser_prob = 0
            invisible = 0
            for assignmentDict in jointLaserInvisibility.getAllPossibleAssignmentDicts():
                if assignmentDict['laser'] == 2:
                    strong_laser_prob += jointLaserInvisibility.getProbability(assignmentDict)
                if assignmentDict['laser'] == 1:
                    weak_laser_prob += jointLaserInvisibility.getProbability(assignmentDict)
                if assignmentDict['invisibility'] == 2 or assignmentDict['invisibility'] == 1:
                    invisible += jointLaserInvisibility.getProbability(assignmentDict)
            if i == 0:
                if len(leftCapsules) < (capsuleLimit / 2):
                    if strong_laser_prob > weak_laser_prob:
                        x = random.randint(1, (width / 2) - 1)
                        y = random.randint(1, height - 2)
                        if gameState.isValidPosition((x, y), self.isRed):
                            leftCapsules.append(ArmourCapsule(x, y))
                    if len(leftCapsules) < (capsuleLimit / 2):
                        if invisible > 0.5:
                            x = random.randint(1, (width / 2) - 1)
                            y = random.randint(1, height - 2)
                            if gameState.isValidPosition((x, y), self.isRed):
                                leftCapsules.append(SonarCapsule(x, y))
                while len(leftCapsules) < (capsuleLimit / 2):
                    x = random.randint(1, (width / 2) - 1)
                    y = random.randint(1, height - 2)
                    if gameState.isValidPosition((x, y), self.isRed):
                        leftCapsules.append(GrenadeCapsule(x, y))
            elif i == 1:
                if len(rightCapsules) < (capsuleLimit / 2):
                    if strong_laser_prob > weak_laser_prob:
                        x = random.randint((width / 2), width - 2)
                        y = random.randint(1, height - 2)
                        if gameState.isValidPosition((x, y), self.isRed):
                            rightCapsules.append(ArmourCapsule(x, y))
                    if len(rightCapsules) < (capsuleLimit / 2):
                        if invisible > 0.5:
                            x = random.randint((width / 2), width - 2)
                            y = random.randint(1, height - 2)
                            if gameState.isValidPosition((x, y), self.isRed):
                                rightCapsules.append(SonarCapsule(x, y))
                while len(rightCapsules) < (capsuleLimit / 2):
                    x = random.randint((width / 2), width - 2)
                    y = random.randint(1, height - 2)
                    if gameState.isValidPosition((x, y), self.isRed):
                        rightCapsules.append(GrenadeCapsule(x, y))
        return leftCapsules + rightCapsules
##########
# Agents #
##########
class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that chooses score-maximizing actions
    """
    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)
    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = gameState.getLegalActions(self.index)
        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(gameState, a) for a in actions]
        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]
        foodLeft = len(self.getFood(gameState).asList())
        if foodLeft <= 2:
            bestDist = 9999
            for action in actions:
                successor = self.getSuccessor(gameState, action)
                pos2 = successor.getAgentPosition(self.index)
                dist = self.getMazeDistance(self.start, pos2, successor)
                if dist < bestDist:
                    bestAction = action
                    bestDist = dist
            return bestAction
        return random.choice(bestActions)
    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(action)
        else:
            return successor
    def evaluate(self, gameState, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights
    def getFeatures(self, gameState, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)
        return features
    def getWeights(self, gameState, action):
        """
        Normally, weights do not depend on the gamestate.  They can be either
        a counter or a dictionary.
        """
        return {'successorScore': 1.0}
class OffensiveReflexAgent(ReflexCaptureAgent):
    def choosePowers(self, gameState, powerLimit):
        validPowers = LEGAL_POWERS[:]
        powers = util.Counter()
        powers['capacity'] = 1
        powers['respawn'] = 2
        powers['speed'] = 1
        return powers
    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        foodList = self.getFood(successor).asList()
        features['successorScore'] = -len(foodList)  # self.getScore(successor)
        myState = successor.getAgentState(self.index)
        scaredSum = 0
        for index in self.getOpponents(successor):
            enemy = successor.getAgentState(index)
            scaredSum += enemy.scaredTimer
        features['opponentsScared'] = scaredSum
        if self.red:
            # If we're winning, time to head back and go on defens
            if len(gameState.getRedFood().asList()) > (len(gameState.getBlueFood().asList()) + 1):
                myPos = successor.getAgentState(self.index).getPosition()
                minDistance = min([self.getMazeDistance(myPos, food, successor) for food in gameState.getRedFood().asList()])
                features['goHome'] = minDistance
        else:
            if len(gameState.getBlueFood().asList()) > (len(gameState.getRedFood().asList()) + 1):
                myPos = successor.getAgentState(self.index).getPosition()
                minDistance = min([self.getMazeDistance(myPos, food, successor) for food in gameState.getBlueFood().asList()])
                features['goHome'] = minDistance
        if (not myState.isPacman): features['goHome'] = 0
        # Compute distance to the nearest food
        if len(foodList) > 0:  # This should always be True, but better safe than sorry
            myPos = successor.getAgentState(self.index).getPosition()
            minDistance = min([self.getMazeDistance(myPos, food, successor) for food in foodList])
            features['distanceToFood'] = minDistance
        return features
    def getWeights(self, gameState, action):
        return {'successorScore': 100, 'distanceToFood': -1, 'opponentsScared': 10, 'goHome': 10000}
class DefensiveReflexAgent(ReflexCaptureAgent):
    def choosePowers(self, gameState, powerLimit):
        validPowers = LEGAL_POWERS[:]
        powers = util.Counter()
        powers['laser'] = 1
        powers['speed'] = 1
        powers['respawn'] = 2
        return powers
    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()
        # Computes whether we're on defense (1) or offense (0)
        features['onDefense'] = 1
        if myState.isPacman: features['onDefense'] = 0
        # Computes distance to invaders we can see
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        features['numInvaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition(), successor) for a in invaders]
            features['invaderDistance'] = min(dists)
        if action == Directions.STOP:
            features['stop'] = 1
        elif action == Directions.LASER:
            features['laser'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1
        return features
    def getWeights(self, gameState, action):
        return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2,
                'laser': -5}
