# myTeam.py
# ---------------
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
from config import LEGAL_POWERS
from util import nearestPoint
from capsule import ScareCapsule, GrenadeCapsule, ArmourCapsule, SonarCapsule, JuggernautCapsule
import contestPowersBayesNet

#################
# Team creation #
#################
class Team:
  def createAgents(self, firstIndex, secondIndex, isRed, gameState,
                   first = 'SpeedyOffensiveAgent', second = 'SpeedyOffensiveAgent'):
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
    y = 1
    mid_left = width / 2 - 1
    leftCapsules = []
    while len(leftCapsules) < (capsuleLimit / 2):
      if gameState.isValidPosition((mid_left, y), self.isRed):
        leftCapsules.append(ScareCapsule(mid_left, y))
      else:
        y += 1

    y = height - 1
    mid_right = width / 2
    rightCapsules = []
    while len(rightCapsules) < (capsuleLimit / 2):
      if gameState.isValidPosition((mid_right, y), self.isRed):
        rightCapsules.append(ScareCapsule(mid_right, y))
      else:
        y += 1

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
    self.teammate_index = [i for i in self.getTeam(gameState) if i != self.index][0]
    CaptureAgent.registerInitialState(self, gameState)

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)
    values = [self.evaluate(gameState, a) for a in actions]

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    foodLeft = len(self.getFood(gameState).asList())
    # if less than 2 food remaining, move closer to starting position
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
    weights = self.getWeights(gameState, action, features)
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action, features):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}

class SpeedyOffensiveAgent(ReflexCaptureAgent):
  def choosePowers(self, gameState, powerLimit):
    powers = util.Counter()
    power_priority = ['speed', 'invisibility', 'respawn', 'capacity', 'laser']
    power_idx = 0
    for i in range(powerLimit):
      powerChoice = power_priority[power_idx]
      powers[powerChoice] += 1
      if powers[powerChoice] == 2:
        power_idx += 1
      if power_idx >= len(power_priority):
        break
    print 'Powers chosen:'
    print powers
    return powers

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    teammateState = successor.getAgentState(self.teammate_index)
    enemies = []
    for enemyIndex in self.getOpponents(successor):
      enemyState = successor.getAgentState(enemyIndex)
      if enemyState.getPosition() is not None:
        enemies.append(enemyState)
    invaders = [a for a in enemies if a.isPacman]

    myPos = myState.getPosition()
    teammatePos = teammateState.getPosition()

    if myState.isPacman:
      features['isPacman'] = 1
    else:
      features['isPacman'] = 0

    features['successorScore'] = self.getScore(successor)

    scaredSum = 0
    for enemy in enemies:
      scaredSum += enemy.scaredTimer
    features['opponentsScared'] = scaredSum

    # Compute distance to the nearest food
    foodList = self.getFood(successor).asList()
    if len(foodList) > 0:
      minDistance = min([self.getMazeDistance(myPos, food, successor) for food in foodList])
      features['distanceToFood'] = minDistance

    features['distanceToTeammate'] = self.getMazeDistance(myPos, teammatePos, successor)

    bestDist = -1
    closestEnemy = None
    for e in enemies:
      eDist = self.getMazeDistance(myPos, e.getPosition(), successor)
      if bestDist == -1 or eDist < bestDist:
        bestDist = eDist
        closestEnemy = e

    if closestEnemy is not None:
        features['distanceToClosestOpponent'] = bestDist
        if closestEnemy in invaders:
          features['isClosestOpponentAnInvader'] = 1
        else:
          features['isClosestOpponentAnInvader'] = 0

    features['numCarrying'] = myState.numCarrying
    features['distanceToStart'] = self.getMazeDistance(myPos, self.start, successor)

    return features

  def getWeights(self, gameState, action, features):
    run_away_from_enemy = {}
    chase_opponent = {}
    go_for_food = {}
    return_to_home = {}

    if features['isPacman'] == 1 and \
       features['isClosestOpponentAnInvader'] == 1 and \
       features['distanceToClosestOpponent'] <= 8:
      return chase_opponent

    if features['distanceToClosestOpponent'] <= 4:
      return run_away_from_enemy

    if features['numCarrying'] == 5:
      return return_to_home

    return go_for_food
