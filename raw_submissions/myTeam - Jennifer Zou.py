from captureAgents import CaptureAgent
import distanceCalculator
import util
from util import nearestPoint
from config import LEGAL_POWERS
import random
from game import Directions
from capsule import ScareCapsule, GrenadeCapsule, ArmourCapsule, SonarCapsule, JuggernautCapsule
import contestPowersBayesNet

# Created by Jennifer Zou, Spring 2018 CS 188

#################
# Team creation #
#################
class Team:
  def createAgents(self, firstIndex, secondIndex, isRed, gameState):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.
    """
    self.isRed = isRed
    return [eval('OffensiveReflexAgent')(firstIndex, gameState), eval('DefensiveReflexAgent')(secondIndex, gameState)]

  def chooseCapsules(self, gameState, capsuleLimit, opponentEvidences):
    width, height = gameState.data.layout.width, gameState.data.layout.height

    maxSumVal = max(opponentEvidences[0].values())
    maxSumKey = [k for k, v in opponentEvidences[0].items() if v == maxSumVal]

    capsule1 = None
    capsule2 = None

    if maxSumKey == 'sum0': #respawn, capacity
      capsule1 = ScareCapsule
      capsule2 = GrenadeCapsule
    elif maxSumKey == 'sum1': #laser, invis
      capsule1 = SonarCapsule
      capsule2 = ScareCapsule

    elif maxSumKey == 'sum2': #speed, laser, capacity
      capsule1 = SonarCapsule
      capsule2 = ScareCapsule
    else: #invis, speed
      capsule1 = SonarCapsule
      capsule2 = ScareCapsule

    leftCapsules = []

    while len(leftCapsules) < (capsuleLimit / 2):
      x = random.randint(1, (width / 2) - 1)
      y = random.randint(1, height - 2)
      if gameState.isValidPosition((x, y), self.isRed):
        if len(leftCapsules) == 0:
          leftCapsules.append(capsule1(x, y))
        else:
          leftCapsules.append(capsule2(x, y))

    rightCapsules = []
    while len(rightCapsules) < (capsuleLimit / 2):
      x = random.randint((width / 2), width - 2)
      y = random.randint(1, height - 2)
      if gameState.isValidPosition((x, y), self.isRed):
        if len(rightCapsules) == 0:
          rightCapsules.append(capsule1(x, y))
        else:
          rightCapsules.append(capsule2(x, y))

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
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

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
  """
  A reflex agent that seeks food.
  Designed to carry unlimited food, be faster than average,
  and be invisible to hide from opponent.
  """

  def choosePowers(self, gameState, powerLimit):
    powers = util.Counter()
    powers['capacity'] = 2
    powers['speed'] = 1
    powers['invisibility'] = 1

    return powers

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    foodList = self.getFood(successor).asList()
    features['successorScore'] = -len(foodList)  # self.getScore(successor)

    scaredSum = 0
    for index in self.getOpponents(successor):
      enemy = successor.getAgentState(index)
      scaredSum += enemy.scaredTimer
    features['opponentsScared'] = scaredSum

    # Compute distance to the nearest food
    if len(foodList) > 0:  # This should always be True, but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, food, successor) for food in foodList])
      features['distanceToFood'] = minDistance
    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 100, 'distanceToFood': -1, 'opponentsScared': 10}


class DefensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that keeps its side Pacman-free.
  Designed to be invisible, fast, and powerful with laser ability.
  """

  def choosePowers(self, gameState, powerLimit):
    powers = util.Counter()
    powers['invisibility'] = 1
    powers['speed'] = 2
    powers['laser'] = 1

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
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2, 'laser': -5}