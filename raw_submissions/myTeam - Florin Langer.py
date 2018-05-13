"""
Students' Names: Florin Langer, Jeffrey Wirjo
Contest Number: 3
Description of Bot: Our bot is bad. This was the first contest we entered, so it was a bit more difficult to get off the ground.
It chooses random (attempted to use inference) power capsules in positions as close to the x-wise center as possible since that's where they're most useful on average.
It also returns to its side to cash in if it has dots and an opponent is close.
"""

from captureAgents import CaptureAgent
import distanceCalculator
from game import Directions
from capsule import ScareCapsule, GrenadeCapsule, ArmourCapsule, SonarCapsule, JuggernautCapsule
import contestPowersBayesNet
from config import LEGAL_POWERS
from capsule import LEGAL_CAPSULES # 5 choices
import random, time, util, sys
from util import nearestPoint
import contestPowersBayesNet

#################
# Team creation #
#################
class Team:
  def createAgents(self, firstIndex, secondIndex, isRed, gameState,
    first = 'OffensiveReflexAgent', second = 'DefensiveReflexAgent'):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.
    """
    self.isRed = isRed
    return [eval(first)(firstIndex, gameState), eval(second)(secondIndex, gameState)]

  def chooseCapsules(self, gameState, capsuleLimit, opponentEvidences):
    # Do custom processing on the opponent's power evidence in order to
    # effectively choose your capsules.
    # net = contestPowersBayesNet.powersBayesNet()
    # for instance to calculate the joint distribution of the
    # enemy agent 0 choosing invisibility and speed:
    # from inference import inferenceByVariableElimination
    # jointSpeedInvisibility = inferenceByVariableElimination(net,
    #                                             ['invisibility', 'speed'],
    #                                             opponentEvidences[0], None)

    # or compute the marginal factors of the enemy powers, given the
    # evidenceAssignmentDict of the sum variables in the enemy powersBayesNet
    # enemyAgentMarginals = []
    # for evidence in opponentEvidences:
    #     marginals = contestPowersBayesNet.computeMarginals(evidence)
    #     enemyAgentMarginals.append(marginals)

    # now perform other inference for other factors or use these marginals to
    # choose your capsules

    # these are not good choices, this is just to show you how to choose capsules

    # Choose best powers
    # net = contestPowersBayesNet.powersBayesNet()
    # from inference import inferenceByVariableElimination
    # power_values = []
    # for i in range(len(LEGAL_CAPSULES)):
    #     power_values.append(inferenceByVariableElimination(net,
    #                                                 [LEGAL_CAPSULES[i]],
    #                                                 opponentEvidences[0], None))
    # best_i = 0
    # best_val = 0
    # for i in range(len(power_values)):
    #     best = 0
    #     best_assignment = None
    #     for assignment in power_values[i].getAllPossibleAssignmentDicts():
    #         if power_values[i].getProbability(assignment) > best:
    #             best_assignment = assignment
    #             best = power_values[i].getProbability(assignment)
    #     if best > best_val:
    #         best_val = best
    #         best_i = i

    width, height = gameState.data.layout.width, gameState.data.layout.height
    leftCapsules = []
    # Choose two on team side
    i = 0
    for x in range((width/2)-1, 0, -1):
        for y in range(1, height-1):
            if gameState.isValidPosition((x, y), self.isRed):
                # LEGAL_CAPSULES[best_i]
                leftCapsules.append(LEGAL_CAPSULES[random.randint(0, len(LEGAL_CAPSULES) - 1)](x, y))
                i+=1
                if not (len(leftCapsules) < (capsuleLimit / 2)):
                    break # Need two breaks for python
        if not (len(leftCapsules) < (capsuleLimit / 2)):
            break

    # print(power_results, power_values)

    rightCapsules = []
    # Choose two on other side
    for x in range(width/2, width-1):
        for y in range(height-2, 0, -1):
            if gameState.isValidPosition((x, y), self.isRed):
                # TODO: Can use bayes to choose capsule types
                rightCapsules.append(LEGAL_CAPSULES[random.randint(0, len(LEGAL_CAPSULES) - 1)](x, y))
                if not (len(rightCapsules) < (capsuleLimit / 2)):
                    break # Need two breaks for python
        if not (len(rightCapsules) < (capsuleLimit / 2)):
            break

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
    oppPoses = [gameState.getAgentPosition(i) for i in self.getOpponents(gameState)]
    if all(oppPoses):
        closestOpponent = min([self.getMazeDistance(gameState.getAgentPosition(self.index), oppPos, gameState) for oppPos in oppPoses])
    else:
        closestOpponent = 5

    if closestOpponent <= 4 and gameState.getAgentState(self.index).numCarrying > 0:
        if self.red:
            if 'West' in actions:
                return 'West'
            if 'North' in actions:
                return 'North'
            if 'South' in actions:
                return 'South'
        else:
            if 'East' in actions:
                return 'East'
            if 'North' in actions:
                return 'North'
            if 'South' in actions:
                return 'South'

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
