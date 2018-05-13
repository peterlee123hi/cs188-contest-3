######################
# Logan Howard       #
# logan@berkeley.edu #
# glinia             #
######################
'''
########################
# STRATEGY DESCRIPTION #
########################
A combination of reflex agents and path planning. The powers and capsules are fixed for both the offensive and defensive agents.
The defensive agent stays mostly towards the center, only rarely turning into a Pacman, trying to stay as a ghost. This agent likes to shoot Pacmen with lasers, and wanders around a little bit, chasing Pacmen it sees.
The offensive agent paths to food, avoiding ghosts when they get too close. Once full, it returns to the center as fast as it can to deposit its food.
'''
from captureAgents import CaptureAgent
import distanceCalculator
import util, random
from game import Directions, Actions
from capsule import ScareCapsule, GrenadeCapsule, ArmourCapsule, SonarCapsule, JuggernautCapsule
from config import LEGAL_POWERS
import contestPowersBayesNet
# stuff from old projects
def manhattanHeuristic(position, problem, info={}):
    "The Manhattan distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])
def manhattenDist(pt1, pt2):
    return abs(pt1[0] - pt2[0]) + abs(pt1[1] - pt2[1])
def euclideanHeuristic(position, problem, info={}):
    "The Euclidean distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return ( (xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2 ) ** 0.5
class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).
    You do not need to change anything in this class, ever.
    """
    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()
    def isGoalState(self, state):
        """
          state: Search state
        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()
    def getSuccessors(self, state):
        """
          state: Search state
        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()
    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take
        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()
class PositionSearchProblem(SearchProblem):
    """
    A search problem defines the state space, start state, goal test, successor
    function and cost function.  This search problem can be used to find paths
    to a particular point on the pacman board.
    The state space consists of (x,y) positions in a pacman game.
    Note: this search problem is fully specified; you should NOT change it.
    """
    def __init__(self, gameState, goal, pacPos):
        """
        Stores the start and goal.
        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.walls = gameState.getWalls()
        self.startState = pacPos
        self.goal = goal
        self.costFn = lambda x: 1
    def getStartState(self):
        return self.startState
    def isGoalState(self, state):
        return state == self.goal
    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.
         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """
        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append( ( nextState, action, cost) )
        return successors
    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions. If those actions
        include an illegal move, return 999999.
        """
        if actions == None: return 999999
        x,y= self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x,y))
        return cost
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
    return [OffensiveAgent(firstIndex, gameState), DefensiveAgent(secondIndex, gameState)]
  def chooseCapsules(self, gameState, capsule_limit, opponent_evidences):
    width, height = gameState.data.layout.width, gameState.data.layout.height
    left_caps, right_caps = [], []
    while len(left_caps) < (capsule_limit / 2):
      x = random.randint(1, (width / 2) - 1)
      y = random.randint(1, height - 2)
      if gameState.isValidPosition((x, y), self.isRed):
        if self.isRed:
          if len(left_caps) % 2 == 0:
            left_caps.append(ArmourCapsule(x, y))
          else:
            left_caps.append(JuggernautCapsule(x, y))
        else:
          if len(left_caps) % 2 == 0:
            left_caps.append(SonarCapsule(x, y))
          else:
            left_caps.append(ScareCapsule(x, y))
    while len(right_caps) < (capsule_limit / 2):
      x = random.randint(1, (width / 2) - 1)
      y = random.randint(1, height - 2)
      if gameState.isValidPosition((x, y), self.isRed):
        if self.isRed:
          if len(right_caps) % 2 == 0:
            right_caps.append(SonarCapsule(x, y))
          else:
            right_caps.append(ScareCapsule(x, y))
        else:
          if len(left_caps) % 2 == 0:
            right_caps.append(ArmourCapsule(x, y))
          else:
            right_caps.append(JuggernautCapsule(x, y))
    return left_caps + right_caps
class BaseAgent(CaptureAgent):
  def registerInitialState(self, gameState):
    self.num_holding = 0
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)
  def chooseAction(self, gameState):
    actions = gameState.getLegalActions(self.index)
    scores = [self.evaluate(gameState, a) for a in actions]
    max_score = max(scores)
    best_idxs = [i for i in range(len(scores)) if scores[i] == max_score]
    return actions[random.choice(best_idxs)]
  def getSuccessor(self, gameState, action):
    successor = gameState.generateSuccessor(action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != util.nearestPoint(pos):
      return successor.generateSuccessor(action)
    return successor
  def evaluate(self, gameState, action):
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights
  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features
  def getWeights(self, gameState, action):
    return {'successorScore': 1.0}
  def createSolution(self, node, parents):
    sol = []
    while parents[node] is not None:
        actionTaken = node[1]
        sol.append(actionTaken)
        node = parents[node]
    sol.reverse()
    return sol
  def aStarSearch(self, problem):
    """Search the node that has the lowest combined cost and heuristic first.
    States are inserted to the fringe as (state, cost so far) tuples."""
    closed = set()
    fringe = util.PriorityQueue()
    start = (problem.getStartState(), 0, 0)
    parents = {start: None}
    fringe.push((start, 0), 0)
    # fringe has (node, cost_so_far) on it
    while not fringe.isEmpty():
        node = fringe.pop()
        state = node[0][0]
        if problem.isGoalState(state):
            return self.createSolution(node[0], parents)
        if state in closed:
            continue
        closed.add(state)
        for successor in problem.getSuccessors(state):
            state = successor[0]
            if state in closed:
                continue
            parents[successor] = node[0]
            h_cost = 0#heuristic(state, problem)
            action_cost = successor[2]
            cost_so_far = node[1]
            fringe.push((successor, cost_so_far + action_cost),
                        cost_so_far + action_cost + h_cost)
    return []
def avgPos(posList):
    if len(posList) == 0:
        return (0, 0)
    x, y = 0, 0
    for pos in posList:
        x += pos[0]
        y += pos[1]
    x /= len(posList)
    y /= len(posList)
    return (x, y)
class OffensiveAgent(BaseAgent):
  def chooseAction(self, gameState):
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    width, height = gameState.data.layout.width, gameState.data.layout.height
    pos = gameState.getAgentState(self.index).getPosition()
    num_invaders = len(invaders)
    dist_to_center = abs(pos[0] - width / 2)
    if (dist_to_center > 3 and not gameState.getAgentState(self.index).isPacman) or (gameState.getAgentState(self.index).isPacman and self.num_holding >= 5):
      centerish = (int(width / 2), int(pos[1]))
      while not gameState.isValidPosition(centerish, True): # isRed?
        if centerish[1] > height / 2:
          centerish = (centerish[0], centerish[1] - 1)
        else:
          centerish = (centerish[0], centerish[1] + 1)
      if centerish != pos:
        return self.aStarSearch(PositionSearchProblem(gameState, centerish, pos))[0]
    dist_to_ghost = 1000
    for index in self.getOpponents(gameState):
      gPos = gameState.getAgentState(index).getPosition()
      if gPos is None:
        continue
      dist_to_ghost = min(dist_to_ghost, manhattenDist(pos, gPos))
    food_list = self.getFood(gameState).asList()
    if dist_to_ghost > 2 and (dist_to_center > 0 or self.num_holding == 0) and len(food_list) > 0:
      closest_food = min(food_list, key=lambda f: self.getMazeDistance(pos, f, gameState))
      if closest_food != pos:
        best = self.aStarSearch(PositionSearchProblem(gameState, closest_food, pos))[0]
        cur_food = len(self.getFood(gameState).asList())
        next_food = len(self.getFood(self.getSuccessor(gameState, best)).asList())
        if cur_food != next_food:
          self.num_holding += 1
        if not self.getSuccessor(gameState, best).getAgentState(self.index).isPacman:
          self.num_holding = 0
        return best
    actions = gameState.getLegalActions(self.index)
    scores = [self.evaluate(gameState, a) for a in actions]
    max_score = max(scores)
    best_idxs = [i for i in range(len(scores)) if scores[i] == max_score]
    best = actions[random.choice(best_idxs)]
    cur_food = len(self.getFood(gameState).asList())
    next_food = len(self.getFood(self.getSuccessor(gameState, best)).asList())
    if cur_food > next_food:
      self.num_holding += 1
    if not self.getSuccessor(gameState, best).getAgentState(self.index).isPacman:
      self.num_holding = 0
    return best
  def choosePowers(self, gameState, powerLimit):
    valid_powers = LEGAL_POWERS[:]
    powers = util.Counter()
    powers['speed'] = 1
    powers['respawn'] = 1
    powers['invisibility'] = 2
    return powers
  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    pos = successor.getAgentState(self.index).getPosition()
    food_list = self.getFood(successor).asList()
    avg_food_pos = avgPos(food_list)
    lots_of_food = abs(pos[0] - avg_food_pos[0]) + abs(pos[1] - avg_food_pos[1])
    closest_food = pos
    if len(food_list) > 0:
        closest_food = min(food_list or [0], key=lambda f: self.getMazeDistance(pos, f, successor))
    dist_to_food = self.getMazeDistance(pos, closest_food, successor)
    dist_to_ghost = 1000
    for index in self.getOpponents(successor):
      gPos = successor.getAgentState(index).getPosition()
      if gPos is None:
        continue
      dist_to_ghost = min(dist_to_ghost, manhattenDist(pos, gPos))
    if dist_to_ghost < 2:
      features['distToGhost'] = -1000
    features['successorScore'] = -len(food_list)
    features['distanceToFood'] = 3. / (lots_of_food + 1.) + 2 / (dist_to_food + 1.)
    features['capsule'] = 0 # FIXME: lol
    if pos == gameState.getAgentState(self.index).getPosition():
      features['samePlace'] = -0.5 * random.random()
    if self.num_holding >= 3 and not successor.getAgentState(self.index).isPacman:
      features['beingPacman'] = 100
    return features
  def getWeights(self, gameState, action):
    return {'successorScore': 1, 'distanceToFood': 1, 'capsule': 1, 'samePlace': 1, 'beingPacman': 1, 'distToGhost': 1}
class DefensiveAgent(BaseAgent):
  def chooseAction(self, gameState):
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    width, height = gameState.data.layout.width, gameState.data.layout.height
    pos = gameState.getAgentState(self.index).getPosition()
    num_invaders = len(invaders)
    dist_to_center = abs(pos[0] - width / 2)
    if dist_to_center > width / 4 or (num_invaders == 0 and dist_to_center > 2):
      centerish = (int(width / 2), int(height / 2))
      while not gameState.isValidPosition(centerish, True): # isRed?
        centerish = (centerish[0], centerish[1] + 1)
      return self.aStarSearch(PositionSearchProblem(gameState, centerish, pos))[0]
    actions = gameState.getLegalActions(self.index)
    scores = [self.evaluate(gameState, a) for a in actions]
    max_score = max(scores)
    best_idxs = [i for i in range(len(scores)) if scores[i] == max_score]
    return actions[random.choice(best_idxs)]
  def choosePowers(self, gameState, powerLimit):
    valid_powers = LEGAL_POWERS[:]
    powers = util.Counter()
    powers['speed'] = 2
    powers['laser'] = 2
    return powers
  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    width, height = gameState.data.layout.width, gameState.data.layout.height
    my_state = successor.getAgentState(self.index)
    pos = my_state.getPosition()
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    features['invaderDistance'] = min([self.getMazeDistance(pos, a.getPosition(), successor) for a in invaders] or [0])
    if my_state.isPacman:
      features['wrongSide'] = 1
    features['distToCenter'] = abs(pos[0] - width / 2) + abs(pos[1] - height / 2)
    features['samePlace'] = 0.5 * random.random()
    return features
  def getWeights(self, gameState, action):
    return {'invaderDistance': -20, 'numInvaders': -1000, 'wrongSide': -5, 'distToCenter': -0.01, 'samePlace': -0.05}
