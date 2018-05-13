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
      y += 1

    y = height - 1
    mid_right = width / 2
    rightCapsules = []
    while len(rightCapsules) < (capsuleLimit / 2):
      if gameState.isValidPosition((mid_right, y), self.isRed):
        rightCapsules.append(ScareCapsule(mid_right, y))
      y -= 1

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
    actions = [a for a in actions if a != 'Stop']
    values = []
    for a in actions:
      value = self.evaluate(gameState, a)
      values.append(value)

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
    return powers

  def evaluate(self, gameState, action):
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action, features)
    value = features * weights
    if self.action_selected == 'go':
      value -= 1000000
    return value

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    teammate_index = [i for i in self.getTeam(gameState) if i != self.index][0]

    myState = successor.getAgentState(self.index)
    teammateState = successor.getAgentState(teammate_index)
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

    if teammatePos is not None:
      features['distanceToTeammate'] = self.getMazeDistance(myPos, teammatePos, successor)
    else:
      features['distanceToTeammate'] = 0

    bestDist = -1
    closestEnemy = None
    for e in enemies:
      eDist = self.getMazeDistance(myPos, e.getPosition(), successor)
      if bestDist == -1 or eDist < bestDist:
        bestDist = eDist
        closestEnemy = e

    features['distanceToClosestOpponent'] = 9999
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
    featuresList = ['isPacman', 'successorScore', 'opponentsScared', 'distanceToFood',
                    'distanceToTeammate', 'distanceToClosestOpponent', 'isClosestOpponentAnInvader',
                    'numCarrying', 'distanceToStart']
    run_away_from_enemy = dict([(feature, 0) for feature in features])
    run_away_from_enemy['distanceToClosestOpponent'] = 50
    run_away_from_enemy['successorScore'] = 1000
    if features['numCarrying'] > 0:
      run_away_from_enemy['distanceToStart'] = -50
    else:
      run_away_from_enemy['distanceToFood'] = -50
      run_away_from_enemy['numCarrying'] = 2000
    run_away_from_enemy['distanceToTeammate'] = 10

    chase_opponent = dict([(feature, 0) for feature in features])
    chase_opponent['distanceToClosestOpponent'] = -50
    chase_opponent['successorScore'] = 1000
    chase_opponent['distanceToTeammate'] = 10

    go_for_food = dict([(feature, 0) for feature in features])
    go_for_food['distanceToFood'] = -100
    go_for_food['successorScore'] = 1000000000
    go_for_food['opponentsScared'] = 100
    go_for_food['distanceToTeammate'] = 10
    go_for_food['distanceToClosestOpponent'] = 20
    go_for_food['numCarrying'] = 100000

    return_to_home = dict([(feature, 0) for feature in features])
    return_to_home['distanceToStart'] = -1000
    return_to_home['successorScore'] = 1000000000
    return_to_home['opponentsScared'] = 1000
    return_to_home['distanceToTeammate'] = 100
    return_to_home['distanceToClosestOpponent'] = 50
    return_to_home['numCarrying'] = 1000000

    if features['isPacman'] == 0 and \
       features['isClosestOpponentAnInvader'] == 1 and \
       features['distanceToClosestOpponent'] <= 8:
      self.action_selected = 'chase'
      return chase_opponent

    if features['distanceToClosestOpponent'] <= 3 and \
       features['opponentsScared'] == 0:
      self.action_selected = 'run'
      return run_away_from_enemy

    if features['numCarrying'] == 5:
      self.action_selected = 'return'
      return return_to_home

    self.action_selected = 'go'
    return go_for_food
