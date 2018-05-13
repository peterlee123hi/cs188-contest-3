# baselineTeam.py
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
import inference
import factorOperations

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
    net = contestPowersBayesNet.powersBayesNet()
    # for instance to calculate the joint distribution of the
    # enemy agent 0 choosing invisibility and speed:
    # from inference import inferenceByVariableElimination
    jointSpeedInvisibility = inference.inferenceByVariableElimination(net,
                                                ['invisibility', 'speed'],
                                                opponentEvidences[0], None)

    # or compute the marginal factors of the enemy powers, given the
    # evidenceAssignmentDict of the sum variables in the enemy powersBayesNet
    enemyAgentMarginals = []
    for evidence in opponentEvidences:
        marginals = contestPowersBayesNet.computeMarginals(evidence)
        enemyAgentMarginals.append(marginals)

    # now perform other inference for other factors or use these marginals to
    # choose your capsules

    # these are not good choices, this is just to show you how to choose capsules
    width, height = gameState.data.layout.width, gameState.data.layout.height
    defenceCapsules = []
    while len(defenceCapsules) < (capsuleLimit / 2):
      if self.isRed:
        x = width / 2 - 2
      else:
        x = width / 2 + 1
      y = random.randint(1, height - 2)
      if gameState.isValidPosition((x, y), self.isRed) and (x,y) not in [c.getPosition for c in defenceCapsules]:
        if len(defenceCapsules)==0:
            defenceCapsules.append(SonarCapsule(x, y))
        else:
            defenceCapsules.append(GrenadeCapsule(x, y))

    offenseCapsules = []
    while len(offenseCapsules) < capsuleLimit / 2:
      if self.isRed:
        x = width / 2
      else:
        x = width / 2 - 1
      y = random.randint(1, height - 2)
      if gameState.isValidPosition((x, y), self.isRed) and (x,y) not in [c.getPosition for c in offenseCapsules]:
        if len(offenseCapsules)==0:
            offenseCapsules.append(JuggernautCapsule(x, y))
        else:
            offenseCapsules.append(GrenadeCapsule(x, y))

    return defenceCapsules + offenseCapsules

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
    # make a ban set when pacman get stuck
    # if Directions.STOP in bestActions:
    #   self.banSet.append(gameState.getAgentState(self.index).getPosition())
    # in grenade and laser both available, use laser
    if Directions.BLAST in bestActions and Directions.LASER in bestActions:
        return Directions.LASER
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
    powers['laser'] = 1
    powers['speed'] = 1
    powers['invisibility'] = 2
    # validPowers.remove('capacity')
    #
    # for i in range(powerLimit - 2):
    #   choice = random.choice(validPowers)
    #   powers[choice] += 1
    #   if powers[choice] == 2:
    #     validPowers.remove(choice)

    return powers

  def getFeatures(self, gameState, action):
    thisState=gameState.getAgentState(self.index)
    foodCapacity = thisState.getFoodCapacity()
    initialpos=gameState.getInitialAgentPosition(self.index)
    powers=thisState.powers
    capX=0 # the x position of capsules that I gonna eat
    if initialpos[0]<3:
      capX = gameState.data.layout.width / 2
    else:
      capX = gameState.data.layout.width / 2 - 1


    # when take foods back, reset banSet
    # if not thisState.isPacman:
    #   self.banSet=[]

    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    foodList = self.getFood(successor).asList()


    capsules=self.getCapsules(successor)
    for c in capsules:
        if isinstance(c,GrenadeCapsule) or isinstance(c,JuggernautCapsule) and (c.getPosition()[0]==capX):
            foodList.append(c.getPosition())
    if len(self.getCapsules(gameState)) == len(self.getCapsules(successor)) and len(capsules)>0 and isinstance(c,ScareCapsule):
        foodList.append(capsules[0].getPosition())
    features['successorScore'] = -len(foodList)#self.getScore(successor)
    # scaredSum = 0
    # for index in self.getOpponents(successor):
    #   enemy = successor.getAgentState(index)
    #   scaredSum += enemy.scaredTimer
    # features['opponentsScared'] = scaredSum


    # Compute distance to the nearest food
    if len(foodList) > 0: # This should always be True, but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, food, successor) for food in foodList])
      features['distanceToFood'] = minDistance

    # if the position is banned, return a minimum score to avoid it
    # if successor.getAgentState(self.index).getPosition() in self.banSet:
    #     features['successorScore']=-9999
    #     return features

    # get back when consumed enough food
    midpos=0
    if initialpos[0]<3:
        midpos=gameState.data.layout.width/2-1
    else:
        midpos=gameState.data.layout.width/2+1
    midPoses=[(midpos,i+1) for i in range(gameState.data.layout.height-1) if not gameState.data.layout.walls.data[gameState.data.layout.width/2-1][i+1]]
    homeDises=[self.getMazeDistance(myPos, p, successor) for p in midPoses]
    if thisState.numCarrying>foodCapacity or (gameState.data.timeleft<=300 and thisState.numCarrying>0):
      disToHome=min(homeDises)
    else:
      disToHome=0
    features['distanceToHome']=disToHome

    # compute defender statistics
    minGhostDis = 100
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    defenders = [a for a in enemies if not a.isPacman and a.getPosition() != None]
    def_sonars = [d.powers.sonar for d in defenders]
    defendersPos = []
    for i in defenders:
      defendersPos.append(i.getPosition())
    distanceToGhost = []
    for i in defendersPos:
      distanceToGhost.append(self.getMazeDistance(myPos, i, successor))
    # if there is a defender and distance less than 5 and sonars are open,avoid them
    if len(distanceToGhost) != 0 and min(distanceToGhost)<=5 and sum(def_sonars)>0:
      minGhostDis = min(distanceToGhost)
      minGhostDis*=5
      minGhostDis+=30
    if len(distanceToGhost) != 0 and min(distanceToGhost)<=2:
        minGhostDis=min(distanceToGhost)*20
    defenders_scaredTimer = [i.scaredTimer for i in defenders]
    if len(defenders_scaredTimer)!=0 and min(defenders_scaredTimer) > 6:
        minGhostDis = 100
    features['distanceToGhost'] = minGhostDis


    # slain enemies is the prior job for us
    features['killCount']=0
    nextKill=successor.getAgentState(self.index).killCount
    if thisState.killCount!=nextKill:
        features['killCount'] = nextKill
    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 100, 'distanceToFood': -1, 'opponentsScared': 10, 'killCount': 10000, 'distanceToHome':-10, 'distanceToGhost':1}


# Panyanzhou's modification start here

class DefensiveReflexAgent(ReflexCaptureAgent):
  # choose DefensiveReflexAgent's power: Laser 1, Invisibility 1, Respawn 1, Speed 1.
  def choosePowers(self, gameState, powerLimit):
    powers = util.Counter()
    powers['laser'] = 1
    powers['invisibility'] = 2
    # powers['respawn'] = 1
    powers['speed'] = 1

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
    closestInvader = None
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition(), successor) for a in invaders]
      features['invaderDistance'] = min(dists)
      # At the same time, record the closest invader.
      closestInvader = invaders[dists.index(min(dists))]

    if action == Directions.STOP: features['stop'] = 1
    elif action == Directions.LASER: features['laser'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    # panyanzhou's modification
    # Run when you are scared, but keep an eye on the opponents.
    # If they have laser, run farther, if not, follow them closer
    myScaredTimer = myState.scaredTimer
    lowerDisBound = 0
    if not myScaredTimer > 0: # You are not scared, the distance can be 0
      if len(invaders) > 0:
        if closestInvader.powers.laser > 0 and closestInvader.powers.sonar>0:
          lowerDisBound = 6
        if closestInvader.powers.laser > 0 and closestInvader.powers.sonar<0:
          lowerDisBound = 3
    else: # you are not scared, distance can't be 0
      if len(invaders)>0:
        if closestInvader.powers.laser > 0 and closestInvader.powers.sonar > 0:
          lowerDisBound = 6
        if closestInvader.powers.laser > 0 and closestInvader.powers.sonar < 0:
          lowerDisBound = 3
        if closestInvader.powers.laser < 0 and closestInvader.powers.sonar > 0:
          lowerDisBound = 1
        if closestInvader.powers.laser < 0 and closestInvader.powers.sonar < 0:
          lowerDisBound = 1

    if features['invaderDistance'] <= lowerDisBound:
      features['invaderDistance'] = -features['invaderDistance']

    # Measure the distance to home, DefensiveAgent needn't come back home.
    enemyInitPos = gameState.getInitialAgentPosition(self.getOpponents(successor)[0])
    initialpos = gameState.getInitialAgentPosition(self.index)
    midPosx = (enemyInitPos[0] + initialpos[0]) / 2
    disToMid = abs(myPos[0] - midPosx)
    disToHome = abs(initialpos[0] - myPos[0])
    if disToMid < 6:
      features['disToHome'] = abs(initialpos[0] - midPosx)
    else:
      features['disToHome'] = disToHome

    # Eat my defensive capsules!
    capX=0 # the x position of capsules that I gonna eat
    if initialpos[0]<3:
      capX = gameState.data.layout.width / 2 - 2
    else:
      capX = gameState.data.layout.width / 2 + 1
    myPos = successor.getAgentState(self.index).getPosition()
    wantList=[]
    if len(invaders)>0:
      capsules=self.getCapsules(successor)
      for c in capsules:
        if isinstance(c,GrenadeCapsule) or isinstance(c,SonarCapsule) and (c.getPosition()[0]==capX):
          wantList.append(c.getPosition())
    if len(wantList) > 0:
      minDistance = min([self.getMazeDistance(myPos, food, successor) for food in wantList])
      features['distanceToCap'] = minDistance



    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10,
            'stop': -100, 'reverse': -2, 'laser': -5, 'disToHome': 1, 'distanceToCap':-100}

