from captureAgents import CaptureAgent
import distanceCalculator
import util
import random
from game import Directions
from config import LEGAL_POWERS
from capsule import ScareCapsule, GrenadeCapsule, ArmourCapsule, SonarCapsule, JuggernautCapsule
import contestPowersBayesNet
#################
# Team creation #
#################
class Team:
  def createAgents(self, firstIndex, secondIndex, isRed, gameState, first = "MayoAndMango", second = "PineappleAndPizza"):
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
    capsules = []
    while len(capsules) < capsuleLimit / 4:
      x = random.randint(1, (width / 2) - 2)
      y = random.randint(1, (height / 2) - 2)
      if gameState.isValidPosition((x, y), self.isRed):
        capsules.append(ScareCapsule(x, y))
    while len(capsules) < capsuleLimit / 4:
      x = random.randint((width / 2) - 2, width - 2)
      y = random.randint(1, (height / 2) - 2)
      if gameState.isValidPosition((x, y), self.isRed):
        capsules.append(ScareCapsule(x, y))
    while len(capsules) < capsuleLimit / 4:
      x = random.randint((width / 2) - 2, width - 2)
      y = random.randint((height / 2) - 2, height - 2)
      if gameState.isValidPosition((x, y), self.isRed):
        capsules.append(ScareCapsule(x, y))
    while len(capsules) < capsuleLimit / 4:
      x = random.randint(1, (width / 2) - 2)
      y = random.randint((height / 2) - 2, height - 2)
      if gameState.isValidPosition((x, y), self.isRed):
        capsules.append(ScareCapsule(x, y))
    return capsules
class MayoAndMango(CaptureAgent):
    def registerInitialState(self, gameState):
      CaptureAgent.registerInitialState(self, gameState)
      self.start = gameState.getAgentPosition(self.index)
      for index in self.getTeam(gameState):
          if index != self.index:
              self.teammate = index
      foodList = self.getFood(gameState).asList()
      self.returning = False
      self.totalY = 0
      for food in foodList:
          self.totalY += food[1]
      self.totalY //= len(foodList)
    def choosePowers(self, gameState, powerLimit):
      valid = LEGAL_POWERS[:]
      powers = {}
      for power in LEGAL_POWERS[:]:
          powers[power] = 0
      for i in range(powerLimit - 2):
        choice = random.choice(valid)
        powers[choice] += 1
        if powers[choice] == 2:
          valid.remove(choice)
      return powers
    def getSuccessor(self, gameState, action):
        return gameState.generateSuccessor(action)
    def returnPath(self, gameState, actions):
        closest = 999999999
        for action in actions:
            successor = self.getSuccessor(gameState, action)
            agentPosition = successor.getAgentPosition(self.index)
            distance = self.getMazeDistance(self.start, agentPosition, successor)
            if distance < closest:
                best = action
                closest = distance
        return best
    def chooseAction(self, gameState):
        if self.red:
            foodLeft = len(gameState.getBlueFood().asList())
        else:
            foodLeft = len(gameState.getRedFood().asList())
        actions = gameState.getLegalActions(self.index)
        if foodLeft <= 2:
            return self.returnPath(gameState, actions)
        values = [self.evaluate(gameState, a) for a in actions]
        maxValue = max(values)
        if maxValue == 99999999999:
            return self.returnPath(gameState, actions)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]
        return random.choice(bestActions)
    def evaluate(self, gameState, action):
        pathValue = self.getFeatures(gameState, action)[1] * -1
        if pathValue == -99999999999:
            self.returning = True
            return 99999999999
        return self.getFeatures(gameState, action)[0] * 100 + pathValue
    def getFeatures(self, gameState, action):
        successor = self.getSuccessor(gameState, action)
        foodList = self.getFood(successor).asList()
        foodFactor = -len(foodList)
        if self.returning:
            return [foodFactor, 99999999999]
        agentPosition = successor.getAgentState(self.index).getPosition()
        candidateDistances = [self.getMazeDistance(agentPosition, food, successor) for food in foodList if food[1] >= self.totalY]
        minDistance = 99999999999
        if len(candidateDistances) > 0:
            minDistance = min(candidateDistances)
        elif len(foodList) > 7:
            candidateDistances = [self.getMazeDistance(agentPosition, food, successor) for food in foodList]
            minDistance = min(candidateDistances)
        if len(foodList) <= 5 and len(candidateDistances) > 0:
            oldLocations = []
            totalDistance = 0
            string = len(foodList)
            location = successor.getAgentState(self.teammate).getPosition()
            while string > 2:
                oldLocations.append(location)
                candidateDistances = [(self.getMazeDistance(location, food, successor), food) for food in foodList if food[1] < self.totalY and food not in oldLocations]
                mininimumDistance = 99999999999
                if len(candidateDistances) > 0:
                    minTuple = min(candidateDistances, key = lambda x: x[0])
                    mininimumDistance = minTuple[0]
                    location = minTuple[1]
                    string -= 1
                    totalDistance += mininimumDistance
                else:
                    return [foodFactor, minDistance]
            if totalDistance < minDistance - 5 and totalDistance > 0 and minDistance > 5:
                return [foodFactor, 99999999999]
        return [foodFactor, minDistance]
class PineappleAndPizza(CaptureAgent):
    def registerInitialState(self, gameState):
      CaptureAgent.registerInitialState(self, gameState)
      self.start = gameState.getAgentPosition(self.index)
      for index in self.getTeam(gameState):
          if index != self.index:
              self.teammate = index
      foodList = self.getFood(gameState).asList()
      self.returning = False
      self.totalY = 0
      for food in foodList:
          self.totalY += food[1]
      self.totalY //= len(foodList)
    def getSuccessor(self, gameState, action):
        return gameState.generateSuccessor(action)
    def choosePowers(self, gameState, powerLimit):
      valid = LEGAL_POWERS[:]
      powers = {}
      for power in LEGAL_POWERS[:]:
          powers[power] = 0
      for i in range(powerLimit - 2):
        choice = random.choice(valid)
        powers[choice] += 1
        if powers[choice] == 2:
          valid.remove(choice)
      return powers
    def returnPath(self, gameState, actions):
        closest = 999999999
        for action in actions:
            successor = self.getSuccessor(gameState, action)
            agentPosition = successor.getAgentPosition(self.index)
            distance = self.getMazeDistance(self.start, agentPosition, successor)
            if distance < closest:
                best = action
                closest = distance
        return best
    def chooseAction(self, gameState):
        if self.red:
            foodLeft = len(gameState.getBlueFood().asList())
        else:
            foodLeft = len(gameState.getRedFood().asList())
        actions = gameState.getLegalActions(self.index)
        if foodLeft <= 2:
            return self.returnPath(gameState, actions)
        values = [self.evaluate(gameState, a) for a in actions]
        maxValue = max(values)
        if maxValue == 99999999999:
            self.returning = True
            return self.returnPath(gameState, actions)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]
        return random.choice(bestActions)
    def evaluate(self, gameState, action):
        pathValue = self.getFeatures(gameState, action)[1] * -1
        if pathValue == -99999999999:
            return 99999999999
        return self.getFeatures(gameState, action)[0] * 100 + pathValue
    def getFeatures(self, gameState, action):
        successor = self.getSuccessor(gameState, action)
        foodList = self.getFood(successor).asList()
        foodFactor = -len(foodList)
        if self.returning:
            return [foodFactor, 99999999999]
        agentPosition = successor.getAgentState(self.index).getPosition()
        candidateDistances = [self.getMazeDistance(agentPosition, food, successor) for food in foodList if food[1] < self.totalY]
        minDistance = 99999999999
        if len(candidateDistances) > 0:
            minDistance = min(candidateDistances)
        elif len(foodList) > 7:
            candidateDistances = [self.getMazeDistance(agentPosition, food, successor) for food in foodList]
            minDistance = min(candidateDistances)
        if len(foodList) <= 5 and len(candidateDistances) > 0:
            oldLocations = []
            totalDistance = 0
            string = len(foodList)
            location = successor.getAgentState(self.teammate).getPosition()
            while string > 2:
                oldLocations.append(location)
                candidateDistances = [(self.getMazeDistance(location, food, successor), food) for food in foodList if food[1] >= self.totalY and food not in oldLocations]
                mininimumDistance = 99999999999
                if len(candidateDistances) > 0:
                    minTuple = min(candidateDistances, key = lambda x: x[0])
                    mininimumDistance = minTuple[0]
                    location = minTuple[1]
                    string -= 1
                    totalDistance += mininimumDistance
                else:
                    return [foodFactor, minDistance]
            if totalDistance < minDistance - 5 and totalDistance > 0 and minDistance > 5:
                return [foodFactor, 99999999999]
        return [foodFactor, minDistance]
