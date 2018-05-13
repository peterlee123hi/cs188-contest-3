# myTeam1.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html
from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions, Actions, Grid
import game
from config import LEGAL_POWERS
from util import nearestPoint, manhattanDistance
from capsule import ScareCapsule, GrenadeCapsule, ArmourCapsule, SonarCapsule, JuggernautCapsule
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
    self.foodNum = 0
    # self.pathToExit = []
    self.myTeam = ''
    self.exitCol = []
    self.walls = gameState.getWalls()
    self.prevActions = [None, None, None, None]
    # get what team the bot is on
    if self.getTeam(gameState)[0] % 2 == 0:
      # exit direction left
      self.myTeam = 'red'
    else:
      # exit direction right
      self.myTeam = 'blue'
    # find available exit column spaces
    if self.myTeam == 'blue':
      exitCol = (gameState.data.layout.width) // 2
    else:
      exitCol = (gameState.data.layout.width - 1) // 2
    for i in range(1, gameState.data.layout.height - 1):
      # self.debugDraw([((gameState.data.layout.width - 1) // 2, i)], [0, 1, 0])
      if not self.walls[exitCol][i]:
        self.exitCol.append((exitCol, i))
    # for entry in self.exitCol:
    #   self.debugDraw([entry], [0, 1, 0])
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
    bestAction = random.choice(bestActions)
    # if bestAction == "Laser":
    newGameState = self.getSuccessor(gameState, bestAction)
    if not newGameState.getAgentState(self.index).isPacman:
      self.foodNum = 0
    self.foodNum += len(self.getFood(gameState).asList()) - len(self.getFood(newGameState).asList())
    self.prevActions.append(bestAction)
    if len(self.prevActions) > 20:
      self.prevActions = self.prevActions[14:21]
    return bestAction
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
    powers = util.Counter()
    powers['laser'] = 1
    powers['invisibility'] = 1
    powers['speed'] = 2
    return powers
  def getFeatures(self, gameState, action):
    # Start like getFeatures of OffensiveReflexAgent
    features = util.Counter()
    successor = self.getSuccessor(gameState,action)
    #Get other variables for later use
    food = self.getFood(gameState)
    # capsules = gameState.getCapsules()
    foodList = food.asList()
    walls = gameState.getWalls()
    x, y = gameState.getAgentState(self.index).getPosition()
    vx, vy = Actions.directionToVector(action)
    newx = int(x + vx)
    newy = int(y + vy)
    # Get set of invaders and defenders
    enemies = [gameState.getAgentState(a) for a in self.getOpponents(gameState)]
    # ghosts
    invaders = [a for a in enemies if not a.isPacman and a.getPosition() != None]
    # attacking pacmen
    defenders =[a for a in enemies if a.isPacman and a.getPosition() != None]
    new_enemies = [gameState.getAgentState(a) for a in self.getOpponents(successor)]
    # activating laser
    new_defenders = [a for a in new_enemies if not a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(new_defenders)
    # Check if pacman has stopped
    if action == Directions.STOP:
      features["stuck"] = 1.0
    # Remove somepoints from laser
    if action == Directions.LASER:
      features["laser"] = 5.0
    if self.prevActions[-4] != None and (self.prevActions[-3] == Directions.REVERSE[self.prevActions[-4]]) and (self.prevActions[-4] == self.prevActions[-2]) and (self.prevActions[-3] == self.prevActions[-1]) and action == self.prevActions[-4]:
      features['repeatMovement'] = 1
    # Get ghosts close by
    for ghost in invaders:
      ghostpos = ghost.getPosition()
      ghostNeighbors = Actions.getLegalNeighbors(ghostpos, walls)
      if (newx, newy) == ghostpos:
        # Encounter a Normal Ghost
        if ghost.scaredTimer == 0:
          features["scaredGhosts"] = 0
          features["normalGhosts"] = 1
        else:
          # Encounter a Scared Ghost (still prioritize food)
          features["eatFood"] += 2
          features["eatGhost"] += 1   
      elif ((newx, newy) in ghostNeighbors) and (ghost.scaredTimer > 0):
        features["scaredGhosts"] += 1
      elif ((newx, newy) in ghostNeighbors) and (ghost.scaredTimer == 0):
        features["normalGhosts"] += 1
    # How to act if scared or not scared
    if gameState.getAgentState(self.index).scaredTimer == 0:    
      for ghost in defenders:
        ghostpos = ghost.getPosition()
        ghostNeighbors = Actions.getLegalNeighbors(ghostpos, walls)
        if (newx, newy) == ghostpos:
          features["eatInvader"] = 1
    else:
      for ghost in enemies:
        if ghost.getPosition()!= None:
          ghostpos = ghost.getPosition()
          ghostNeighbors = Actions.getLegalNeighbors(ghostpos, walls)
          if (newx, newy) in ghostNeighbors or (newx, newy) == ghostpos:
            features["eatInvader"] = -10
    # Get capsules when nearby
    # for cx, cy in capsules:
    #   if newx == cx and newy == cy and successor.getAgentState(self.index).isPacman:
    #     features["eatCapsule"] = 1
    # When to eat
    if not features["normalGhosts"]:
      if food[newx][newy]:
        features["eatFood"] = 1.0
      if len(foodList) > 0:
        tempFood =[]
        for food in foodList:
          food_x, food_y = food
          adjustedindex = self.index-self.index % 2
          check1 = food_y > (adjustedindex / 2) * walls.height / 3
          check2 = food_y < ((adjustedindex / 2) + 1) * walls.height / 3
          if (check1 and check2):
            tempFood.append(food)
        if len(tempFood) == 0:
          tempFood = foodList
      if len(foodList) > 0:
        mazedist = [self.getMazeDistance((newx, newy), food, gameState) for food in tempFood]
      else:
        mazedist = [None]
      if min(mazedist) is not None:
        walldimensions = walls.width * walls.height
        features["nearbyFood"] = float(min(mazedist)) / walldimensions  
    # If we've eaten enough food, try and go to an exit route
    if self.foodNum >= 4: #and (newx, newy) in self.pathToExit:
      # closestExit = self.pathToExit[-1]
      closestExit = self.exitCol[0]
      dist = self.getMazeDistance((newx, newy), closestExit, gameState)
      for entry in self.exitCol:
        if self.getMazeDistance((newx, newy), entry, gameState) < dist:
          closestExit = entry
          dist = self.getMazeDistance((newx, newy), entry, gameState)
      # features["pathOnExitRoute"] = 1
      normalized = manhattanDistance((0,0), closestExit)
      features["closeToExitPos"] = manhattanDistance(closestExit, (newx, newy)) / float(normalized)
    return features
  def getWeights(self, gameState, action):
    return {'eatInvader': 5,'teammateDist': 1.5, 'nearbyFood': -5, 'eatCapsule': 10,
    'normalGhosts': -300, 'eatGhost': 1.0, 'scaredGhosts': 0.1, 'stuck': -10, 'eatFood': 1, 'pathOnExitRoute': 10, 'closeToExitPos': -20, 'repeatMovement': -1, 'laser': -20, 'numInvaders': -3000}
class DefensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """
  def choosePowers(self, gameState, powerLimit):
    powers = util.Counter()
    powers['laser'] = 1
    powers['invisibility'] = 1
    powers['speed'] = 2
    return powers
  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()
    # computes whether we're on offense (-1) or defense (1)
    features['onDefense'] = 1
    if myState.isPacman: 
      features['onDefense'] = -1
    # standard from baseline - more numInvaders, larger minimum distance, worse successor (-30000, -1500).
    # Note if the opponent is greater than or equal to 5 blocks away (manhattan distance) then we only get a noisy reading.
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      distsManhattan = [manhattanDistance(myPos, a.getPosition()) for a in invaders]
      dists = [self.getMazeDistance(myPos, a.getPosition(), gameState) for a in invaders]
      # get a more exact reading if food disappears between turns
      if min(distsManhattan) >= 5: 
        prevGamestate = self.getPreviousObservation()
        currGamestate = self.getCurrentObservation()
        prevFood = self.getFood(prevGamestate).asList()
        currFood = self.getFood(currGamestate).asList()
        missingFood = list(set(currFood) - set(prevFood))
        dists.extend([self.getMazeDistance(myPos, a, gameState) for a in missingFood])
        features['invaderDistance'] = min(dists)
      else:
        features['invaderDistance'] = min(dists)
    # standard from baseline - is the action was to stop (-400) or to go back / reverse (-250)
    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1
    # Remove somepoints from laser
    if action == Directions.LASER:
      features["laser"] = 5.0
    # transform defense agent into offensive when scared OR when there are no invaders
    # ADJUST WHEN DONE WITH OFFENSIVE AGENT - ADD ALL CHARACTERISTICS HERE ALSO CONSIDER THE OFFENSE OR DEFENSE PLAY (ABOVE). 
    if(successor.getAgentState(self.index).scaredTimer > 0):
      features['numInvaders'] = 0 # change all of the defense ones to 0
      if(features['invaderDistance'] <= 2): features['invaderDistance'] = 2
    # use the minimum noisy distance between our agent and their agent
    teamNums = self.getTeam(gameState)
    features['stayApart'] = self.getMazeDistance(gameState.getAgentPosition(teamNums[0]), gameState.getAgentPosition(teamNums[1]), gameState)
    features['offenseFood'] = 0
    # IF THERE ARE NO INVADERS THEN GO FOR FOOD / REFLEX AGENT. I LIKE THIS. COPY THE OFFENSE CODE HERE AS WELL.
    if(len(invaders) == 0 and successor.getScore() != 0):
      features['onDefense'] = -1
      if len(self.getFood(successor).asList()) != 0:
        features['offenseFood'] = min([self.getMazeDistance(myPos,food, gameState) for food in self.getFood(successor).asList()])
      features['foodCount'] = len(self.getFood(successor).asList())
      features['stayAprts'] += 2
      features['stayApart'] *= features['stayApart']
        # If we've eaten enough food, try and go to an exit route
    if self.foodNum >= 4 and myState.isPacman: #and (newx, newy) in self.pathToExit:
      # closestExit = self.pathToExit[-1]
      x, y = gameState.getAgentState(self.index).getPosition()
      vx, vy = Actions.directionToVector(action)
      newx = int(x + vx)
      newy = int(y + vy)
      closestExit = self.exitCol[0]
      dist = self.getMazeDistance((newx, newy), closestExit, gameState)
      for entry in self.exitCol:
        if self.getMazeDistance((newx, newy), entry, gameState) < dist:
          closestExit = entry
          dist = self.getMazeDistance((newx, newy), entry, gameState)
      # features["pathOnExitRoute"] = 1
      normalized = manhattanDistance((0,0), closestExit)
      features["closeToExitPos"] = manhattanDistance(closestExit, (newx, newy)) / float(normalized)
    if not myState.isPacman: # NEW - IN GENERAL WE SHOULD BE CLOSE TO THE EXIT COLUMNS, BUT WE DIVIDE BY A BIT MORE BECAUSE IT'S NOT SUPER PERTINENT
      # closestExit = self.pathToExit[-1]
      x, y = gameState.getAgentState(self.index).getPosition()
      vx, vy = Actions.directionToVector(action)
      newx = int(x + vx)
      newy = int(y + vy)
      closestExit = self.exitCol[0]
      dist = self.getMazeDistance((newx, newy), closestExit, gameState)
      for entry in self.exitCol:
        if self.getMazeDistance((newx, newy), entry, gameState) < dist:
          closestExit = entry
          dist = self.getMazeDistance((newx, newy), entry, gameState)
      # features["pathOnExitRoute"] = 1
      normalizedAdjusted = manhattanDistance((0,0), closestExit) * 5
      features["closeToExitPos"] = manhattanDistance(closestExit, (newx, newy)) / float(normalizedAdjusted)
    if myState.isPacman:
      walls = gameState.getWalls()
      x, y = gameState.getAgentState(self.index).getPosition()
      vx, vy = Actions.directionToVector(action)
      newx = int(x + vx)
      newy = int(y + vy)
      # Get ghosts close by
      for ghost in invaders:
        ghostpos = ghost.getPosition()
        ghostNeighbors = Actions.getLegalNeighbors(ghostpos, walls)
        if (newx, newy) == ghostpos:
          # Encounter a Normal Ghost
          if ghost.scaredTimer == 0:
            features["scaredGhosts"] = 0
            features["normalGhosts"] = 1
          else:
            # Encounter a Scared Ghost (still prioritize food)
            features["eatFood"] += 2
            features["eatGhost"] += 1   
        elif ((newx, newy) in ghostNeighbors) and (ghost.scaredTimer > 0):
          features["scaredGhosts"] += 1
        elif ((newx, newy) in ghostNeighbors) and (ghost.scaredTimer == 0):
          features["normalGhosts"] += 1
      # capsules = gameState.getCapsules()
      # for cx, cy in capsules:
      #     if newx == cx and newy == cy and successor.getAgentState(self.index).isPacman:
      #       features["eatCapsule"] = 1
    return features
  def getWeights(self,gameState, action):
    return {'foodCount': -20,'offenseFood': -1, 'numInvaders': -30000, 'onDefense': 10, 'stayApart': 50, 'invaderDistance':-1500, 'stop':-400,'reverse':-250, "closeToExitPos": -50, 'normalGhosts': -3000,
    'eatInvader': 5,'teammateDist': 1.5, 'nearbyFood': -5, 'eatCapsule': 10, 'eatGhost': 1.0, 'scaredGhosts': 0.1, 'stuck': -10, 'eatFood': 1, 'pathOnExitRoute': 10, 'repeatMovement': -1, 'laser': -20}