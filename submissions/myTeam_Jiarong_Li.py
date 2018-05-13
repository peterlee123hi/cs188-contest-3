from captureAgents import CaptureAgent
import distanceCalculator
import util
from game import Directions
from capsule import ScareCapsule, GrenadeCapsule, ArmourCapsule, SonarCapsule, JuggernautCapsule
import contestPowersBayesNet
from config import LEGAL_POWERS
import random, time, util, sys
from util import nearestPoint
# Name: Jiarong Li
# edX Username: lijiarong
# Email: jiarongli@berkeley.edu
#################
# Team creation #
#################
class Team:
  def createAgents(self, firstIndex, secondIndex, isRed, gameState, first = 'OffensiveReflexAgent', second = 'DefensiveReflexAgent'):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.
    """
    self.isRed = isRed
    return [eval(first)(firstIndex, gameState), eval(second)(secondIndex, gameState)]
  def chooseCapsules(self, gameState, capsuleLimit, opponentEvidences):
    width, height = gameState.data.layout.width, gameState.data.layout.height
    leftCapsules = []
    while len(leftCapsules) < (capsuleLimit / 2):
      x = random.randint(1, (width / 2) - 1)
      y = random.randint(1, height - 2)
      if gameState.isValidPosition((x, y), self.isRed):
        leftCapsules.append(ScareCapsule(x, y))
    rightCapsules = []
    while len(rightCapsules) < (capsuleLimit / 2):
      x = random.randint((width / 2), width - 2)
      y = random.randint(1, height - 2)
      if gameState.isValidPosition((x, y), self.isRed):
        rightCapsules.append(ScareCapsule(x, y))
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
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
  def choosePowers(self, gameState, powerLimit):
    validPowers = LEGAL_POWERS[:]
    powers = util.Counter()
    powers['capacity'] = 2
    validPowers.remove('capacity')
    for i in range(powerLimit - 2):
      choice = random.choice(validPowers)
      powers[choice] += 1
      if powers[choice] == 2:
        validPowers.remove(choice)
    return powers
  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    foodList = self.getFood(successor).asList()    
    features['successorScore'] = -len(foodList)#self.getScore(successor)
    scaredSum = 0
    for index in self.getOpponents(successor):
      enemy = successor.getAgentState(index)
      scaredSum += enemy.scaredTimer
    features['opponentsScared'] = scaredSum
    # Compute distance to the nearest food
    if len(foodList) > 0: # This should always be True, but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, food, successor) for food in foodList])
      features['distanceToFood'] = minDistance
    return features
  def getWeights(self, gameState, action):
    return {'successorScore': 100, 'distanceToFood': -1, 'opponentsScared': 10}
class DefensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """
  def choosePowers(self, gameState, powerLimit):
    validPowers = LEGAL_POWERS[:]
    powers = util.Counter()
    powers['laser'] = 2
    validPowers.remove('laser')
    for i in range(powerLimit - 2):
      choice = random.choice(validPowers)
      powers[choice] += 1
      if powers[choice] == 2:
        validPowers.remove(choice)
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
    if action == Directions.STOP: features['stop'] = 1
    elif action == Directions.LASER: features['laser'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1
    return features
  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2, 'laser': -5}