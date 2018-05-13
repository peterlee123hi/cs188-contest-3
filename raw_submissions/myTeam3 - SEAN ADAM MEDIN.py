# myTeam.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from captureAgents import CaptureAgent
import distanceCalculator
import util
from game import Directions
from capsule import ScareCapsule, GrenadeCapsule, ArmourCapsule, SonarCapsule, JuggernautCapsule
import contestPowersBayesNet
import baselineTeam as baseline
import random
import numpy
import math
import heapq
import time
from math import sqrt
from math import exp
import numpy.random as np
import numpy.linalg as LA

#################
# Team creation #
#################

class Team:

  def createAgents(self, firstIndex, secondIndex, isRed, gameState,
                 first = 'RAN', second = 'RAN'):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """

    self.isRed = isRed
    self.agent1 = eval(first)(firstIndex,gameState)
    self.agent2 = eval(second)(secondIndex,gameState)
    return [self.agent1, self.agent2]

  def chooseCapsules(self, gameState, capsuleLimit, opponentEvidences):
    P = PowersInference()
    e_1 = [ opponentEvidences[0]['sum0'], opponentEvidences[0]['sum1'], opponentEvidences[0]['sum2'], opponentEvidences[0]['sum3'] ]
    e_2 = [ opponentEvidences[1]['sum0'], opponentEvidences[1]['sum1'], opponentEvidences[1]['sum2'], opponentEvidences[1]['sum3'] ]

    P.infer(e_1, 0)
    P.infer(e_2, 1)

    times_added = self.addDefaultCaps(gameState)
    new_caps = self.addMySide(gameState,times_added)
    self.addOtherSide(gameState,times_added,new_caps)

    #print(new_caps)
    return [ScareCapsule(cap[0],cap[1]) for cap in new_caps]

  def addDefaultCaps(self,gameState):
    my_caps = self.agent1.getCapsules(gameState)
    times_added = []
    for cap_class in my_caps:
      cap = cap_class.getPosition()
      best_place = None
      time_out = 0
      add_time = 0
      vars1 = self.bestPathToCap(gameState,self.agent1,RAN.PAC1PATH,cap)
      if vars1[0]:
        times_added = times_added + vars1[1]
        #print('perfect')
        continue
      vars2 = self.bestPathToCap(gameState,self.agent2,RAN.PAC2PATH,cap)
      if vars2[0]:
        times_added = times_added + vars2[1]
        #print('perfect')
        continue
      if vars1[2] > vars2[2]:
        RAN.PAC2PATH.insert(vars2[1],cap)
        times_added.append(vars2[3])
      else:
        RAN.PAC1PATH.insert(vars1[1],cap)
        times_added.append(vars1[3])
    return times_added     

  def bestPathToCap(self,gameState,agent,path,cap):
    times_added = []
    start1 = gameState.getInitialAgentPosition(agent.index)
    orig_len = agent.getMazeDistance(start1,path[0],gameState)
    to_cap = agent.getMazeDistance(start1,cap,gameState)
    from_cap = agent.getMazeDistance(cap,path[0],gameState)
    if orig_len == to_cap + from_cap:
      times_added.append(to_cap)
      path.insert(0,cap)
      return True,times_added
    else:
      time_out = to_cap + from_cap - orig_len
      best_place = 0
      add_time = to_cap
    net_len = orig_len
    for i in range(len(path) - 1):
      orig_len = agent.getMazeDistance(path[i],path[i+1],
        gameState)
      to_cap = agent.getMazeDistance(path[i],cap,gameState)
      from_cap = agent.getMazeDistance(cap,path[i+1],gameState)
      if orig_len == to_cap + from_cap:
        times_added.append(net_len + to_cap)
        path.insert(i+1,cap)
        return True,times_added
      elif time_out > to_cap + from_cap - orig_len:
        time_out = to_cap + from_cap - orig_len
        best_place = i+1
        add_time = net_len + to_cap
      net_len += orig_len
    orig_len = agent.get_best_dist(gameState,path[-1])
    to_cap = agent.getMazeDistance(path[-1],cap,gameState)
    from_cap = agent.get_best_dist(gameState,cap)
    if orig_len == to_cap + from_cap:
      times_added.append(to_cap + net_len)
      path.append(cap)
      return True,times_added
    else:
      time_out = to_cap + from_cap - orig_len
      best_place = len(path)
      add_time = to_cap + net_len
    return False,best_place,time_out,add_time

  def addMySide(self,gameState,times_added):
    x_pos = gameState.data.layout.width / 2
    new_caps = []
    if self.isRed:
      x_pos -= 3
    else:
      x_pos += 2

    start1 = gameState.getInitialAgentPosition(self.agent1.index)
    start2 = gameState.getInitialAgentPosition(self.agent2.index)
    while x_pos > 1 and x_pos < gameState.data.layout.width - 2:
      for y_pos in range(gameState.data.layout.height):
        if not gameState.isValidPosition((x_pos,y_pos),self.isRed):
          continue
        orig_dist1 = self.agent1.getMazeDistance(start1,RAN.PAC1PATH[0],gameState)
        to_cap1 = self.agent1.getMazeDistance(start1,(x_pos,y_pos),gameState)
        from_cap1 = self.agent1.getMazeDistance((x_pos,y_pos),RAN.PAC1PATH[0],gameState)
        if orig_dist1 == to_cap1 + from_cap1:
          RAN.PAC1PATH.insert(0,(x_pos,y_pos))
          new_caps.append((x_pos,y_pos))
          times_added.append(to_cap1)
          if len(new_caps) == 2:
            return new_caps
          continue

        orig_dist2 = self.agent2.getMazeDistance(start2,RAN.PAC2PATH[0],gameState)
        to_cap2 = self.agent2.getMazeDistance(start2,(x_pos,y_pos),gameState)
        from_cap2 = self.agent2.getMazeDistance((x_pos,y_pos),RAN.PAC2PATH[0],gameState)
        if orig_dist2 == to_cap2 + from_cap2:
          RAN.PAC2PATH.insert(0,(x_pos,y_pos))
          new_caps.append((x_pos,y_pos))
          times_added.append(to_cap2)
          if len(new_caps) == 2:
            return new_caps
      if self.isRed:
        x_pos -= 1
      else:
        x_pos += 1
    return new_caps

  def findTimes(self,gameState,times_added):
    time_remaining = [20]
    new_times = []
    i = 0
    while i < len(times_added) - 1:
      if time_remaining[i] - (times_added[i+1] - times_added[i]):
        new_times.append(times_added[i] + time_remaining[i])
        if len(new_times) == 2:
          return new_times
        times_added.insert(i+1,times_added[i] + 15)
        time_remaining.append(time_remaining[i] + 5)
      else:
        time_remaining.append(time_remaining[i] - (times_added[i+1] -
          times_added[i]) + 20)
      i += 1

    if len(new_times) < 2:
      new_times.append(times_added[-1] + time_remaining[-1])
    if len(new_times) < 2:
      new_times.append(times_added[-1] + time_remaining[-1] + 15)
    return new_times

  def findPlacement(self,gameState,start_idx,path,new_caps):
    while start_idx > 2:
      if self.agent1.getMazeDistance(path[start_idx+1],path[start_idx],
        gameState) > 1:
        pos = path[start_idx + 1]
        tests = [(pos[0] + 1, pos[1]),(pos[0] - 1, pos[1]),
          (pos[0], pos[1] + 1), (pos[0], pos[1] - 1)]
        for test in tests:
          if gameState.isValidPosition(test,self.isRed) and test not in new_caps:
            path.insert(start_idx + 1,test)
            return test
      start_idx -= 1
    return None

  def addOtherSide(self,gameState,times_added,new_caps):
    times_added = sorted(times_added)
    new_times = self.findTimes(gameState,times_added)

    start1 = gameState.getInitialAgentPosition(self.agent1.index)
    start2 = gameState.getInitialAgentPosition(self.agent2.index)
    path1 = self.agent1.getMazeDistance(start1,RAN.PAC1PATH[0],gameState)
    path2 = self.agent2.getMazeDistance(start2,RAN.PAC2PATH[0],gameState)
    
    idx1 = 1
    idx2 = 1
    while len(new_caps) < 4 and idx1 < len(RAN.PAC1PATH) and idx2 < len(RAN.PAC2PATH):
      added_dist1 = self.agent1.getMazeDistance(RAN.PAC1PATH[idx1],
        RAN.PAC1PATH[idx1 - 1], gameState)
      added_dist2 = self.agent2.getMazeDistance(RAN.PAC2PATH[idx2],
        RAN.PAC2PATH[idx2 - 1], gameState)
      if path1 + added_dist1 < path2 + added_dist2:
        if added_dist1 + path1 > new_times[0] or idx1 == len(RAN.PAC1PATH) - 1:
          result = self.findPlacement(gameState,idx1 - 1, RAN.PAC1PATH,new_caps)
          if result != None:
            new_caps.append(result)

            if len(new_caps) == 2:
              return
            del new_times[0]
        idx1 += 1
        path1 += added_dist1
      else:
        if path2 + added_dist2 > new_times[0] or idx2 == len(RAN.PAC2PATH) - 1:
          result = self.findPlacement(gameState,idx2 - 1, RAN.PAC2PATH,new_caps)
          if result != None:
            new_caps.append(result)
            if len(new_caps) == 2:
              return
            del new_times[0]
        idx2 += 1
        path2 += added_dist2

##########
# Agents #
##########

argmin = lambda x: min( range(len(x)), key = lambda i: x[i] )
nonnegative = lambda x: all( [n >= 0 for n in x] )
step_away = lambda x, l: [ y for y in l if manhattan(x, y) == 1 ]


# be afraid, be very afraid
# powers are 2 speed and 2 capacity. Figures out path and places scare capsules
# in such a way that we can get all the food and get back without the 
# opponents having a chance at killing us
class RAN(CaptureAgent):
  '''
  During Initialization
  1. Chooses a path based on a heuristic (if there are no more of these paths, 
  then randomly rearrange dots in current path to get new path)
  2. Iterates through food and looks to see if each food can be moved to
     a part of the path to make the total number of required moves shorter
  3. Repeat steps 1 and 2, keeping the best result
  4. Utilizes all the time allowed to do this

  Heuristic: Collect next dot that minimizes CLOSE_PENALTY * distance to other
  pacman's next dot + GRADIENT_PENALTY * distance to end. CLOSE_PENALTY and
  GRADIENT_PENALTY varies

  After Initialization:
  Continue doing steps 1-4, but always with a randomly rearranged path
  '''

  #current paths that each pacman should take
  PAC1PATH = []
  PAC2PATH = []
  OUT = []
  WEIGHTS = []
  CONVERGED = True
  FOOD_LIMIT = 50
  QUEUE = None
  CLOSED = None
  ITERS = 0

  #index of pac1
  PAC1 = None

  opp_info = []
    

  def choosePowers(self, gameState, powerLimit):
    return {'invisibility': 0, 'speed': 2, 'laser': 0, 'respawn': 0, 'capacity': 2}

  def registerInitialState(self, gameState):
    self.dists = {}
    start = time.time()
    self.gameState = gameState
    CaptureAgent.registerInitialState(self, gameState)

    if len(RAN.opp_info) == 0:
      power_dics = self.getOpponentTeamPowers(gameState)
      opps = self.getOpponents(gameState)
      for opp, power_dic in zip(opps,power_dics):
         initial_pos = gameState.getInitialAgentPosition(opp)
         RAN.opp_info.append(OppInfo(opp, power_dic, initial_pos))
      OppInfo.lastState = gameState

    if self.red:
      food = gameState.getBlueFood().asList()
    else:
      food = gameState.getRedFood().asList() 

    for i in range(len(food)):
      fd1 = food[i]
      self.dists[fd1] = self.get_best_dist(gameState, fd1)

    if RAN.PAC1 == None:
      RAN.PAC1 = self.index
      random.shuffle(food)
      RAN.WEIGHTS.append((.7,.3))
      self.targeted_reset(gameState,food)
      val1,val2 = self.calc_path_lens(gameState,RAN.PAC1PATH,RAN.PAC2PATH)
      best = max(val1,val2)
      RAN.QUEUE = util.PriorityQueue()
      RAN.QUEUE.push((tuple(RAN.PAC1PATH), tuple(RAN.PAC2PATH),tuple(RAN.OUT)),
        best)
      RAN.CLOSED = set()
      RAN.CLOSED.add((tuple(RAN.PAC1PATH), tuple(RAN.PAC2PATH),tuple(RAN.OUT)))
      close_steps = max(25 - 1.5 * sqrt(len(food)),3);
      close_weights = [i / close_steps for i in range(int(close_steps))]
      gradient_steps = max(20 - 1.5 * sqrt(len(food)),4)
      gradient_weights = [i / gradient_steps for i in range(int(gradient_steps))]
      for cw in close_weights:
        for gw in gradient_weights:
          RAN.WEIGHTS.append((cw,gw))
          RAN.WEIGHTS.append((cw,-gw))
      random.shuffle(RAN.WEIGHTS)
    else:
      val1,val2 = self.calc_path_lens(gameState,RAN.PAC1PATH,RAN.PAC2PATH)
      best = max(val1,val2)
    #print best

    while time.time() - start < 11:
      saved = (list(RAN.PAC1PATH),list(RAN.PAC2PATH),list(RAN.OUT))
      score = self.improve(gameState,start,11)
      path1,path2 = self.calc_path_lens(gameState,RAN.PAC1PATH,RAN.PAC2PATH) 
      if score < best:
        best = score
        #print score

      else:
        RAN.PAC1PATH = saved[0]
        RAN.PAC2PATH = saved[1]
        RAN.OUT = saved[2]

  # completely randomly produces new paths for the pacmans through the food
  def complete_reset(self,food):
    RAN.PAC1PATH = []
    RAN.PAC2PATH = []
    RAN.OUT = []
    random.shuffle(food)
    for fd in food:
      if random.choice([0,1]) == 0:
        RAN.PAC1PATH.append(fd)
      else:
        RAN.PAC2PATH.append(fd)
    RAN.OUT = [RAN.PAC1PATH.pop(), RAN.PAC2PATH.pop()]

  # randoly switches between 0 and 1/4 of the dots in each path between paths
  # and shuffles each path (and switches which dots are out of the path)
  def partial_reset(self):
    num_rm1 = random.randint(0,len(RAN.PAC1PATH) / 4)
    pac1_smple = random.sample(RAN.PAC1PATH,num_rm1)
    for p in pac1_smple:
      RAN.PAC1PATH.remove(p)

    num_rm2 = random.randint(0,len(RAN.PAC2PATH) / 4)
    pac2_smple = random.sample(RAN.PAC2PATH,num_rm2)
    for p in pac2_smple:
      RAN.PAC2PATH.remove(p)

    for p in pac1_smple:
      RAN.PAC2PATH.insert(random.randint(0,len(RAN.PAC2PATH)),p)

    for p in pac2_smple:
      RAN.PAC1PATH.insert(random.randint(0,len(RAN.PAC1PATH)),p)

    if len(RAN.PAC1PATH) > 1:
      r1 = random.sample(RAN.PAC1PATH,1)
      r1 = r1[0]
      RAN.PAC1PATH.remove(r1)
      RAN.PAC1PATH.insert(random.randint(0,len(RAN.PAC1PATH)),RAN.OUT[0])
      RAN.OUT[0] = r1
    if len(RAN.PAC2PATH) > 1:
      r2 = random.sample(RAN.PAC2PATH,1)
      r2 = r2[0]
      RAN.PAC2PATH.remove(r2)
      RAN.PAC2PATH.insert(random.randint(0,len(RAN.PAC2PATH)),RAN.OUT[1])
      RAN.OUT[1] = r2
    random.shuffle(RAN.PAC1PATH)
    random.shuffle(RAN.PAC2PATH)

  #calculates path based on heuristic
  def targeted_reset(self,gameState,food):

    new_try = RAN.WEIGHTS.pop()
    food_c = list(food)
    close_penalty = new_try[0]
    gradient = new_try[1]
    RAN.PAC1PATH = []
    RAN.PAC2PATH = []
    RAN.OUT = []
    path1 = 0
    path2 = 0
    pac1 = gameState.getAgentPosition(RAN.PAC1)
    pac2 = gameState.getAgentPosition((RAN.PAC1 + 2) % 4)
    while len(food_c) > 2:
      isPac1 = True
      if path1 > path2:
        isPac1 = False

      best = 100000
      best_fd = food_c[0]
      for fd in food_c:
        if isPac1:
          test = self.getMazeDist(pac1,fd)
          test -= self.getMazeDist(pac2,fd) * close_penalty
        else:
          test = self.getMazeDist(pac2,fd)
          test -= self.getMazeDist(pac1,fd) * close_penalty
        test += gradient * self.get_best_dist(gameState,fd)
        if test < best:
          best = test
          best_fd = fd

      if isPac1:
        RAN.PAC1PATH.append(best_fd)
        path1 += self.getMazeDist(pac1,best_fd)
        pac1 = best_fd
      else:
        RAN.PAC2PATH.append(best_fd)
        path2 += self.getMazeDist(pac2,best_fd)
        pac2 = best_fd
      food_c.remove(best_fd)

    RAN.OUT = list(food_c)

  def chooseAction(self, gameState):
    self.gameState = gameState
    start = time.time()
    
    for inf in RAN.opp_info:
      inf.updatePos(self, gameState)
      OppInfo.lastState = gameState
      OppInfo.lastAgent = self

    if self.red:
      food = gameState.getBlueFood().asList()
    else:
      food = gameState.getRedFood().asList()

    caps = [cap.getPosition() for cap in self.getCapsules(gameState)]

    #checks to see if things on the path have been eaten
    i = 0
    while i < len(RAN.PAC1PATH):
      if RAN.PAC1PATH[i] not in food and RAN.PAC1PATH[i] not in caps:
        del RAN.PAC1PATH[i]
        i -= 1
      i += 1

    i = 0
    while i < len(RAN.PAC2PATH):
      if RAN.PAC2PATH[i] not in food and RAN.PAC2PATH[i] not in caps:
        del RAN.PAC2PATH[i]
        i -= 1
      i += 1

    if len(RAN.OUT) > 1 and RAN.OUT[1] not in food:
      del RAN.OUT[1]
    if len(RAN.OUT) > 0 and RAN.OUT[0] not in food:
      del RAN.OUT[0]

    '''
    if len(food) > 2  and len(food) < RAN.FOOD_LIMIT:
      path1,path2 = self.calc_path_lens(gameState,RAN.PAC1PATH,RAN.PAC2PATH) 
      best = max(path1,path2)

      while time.time() - start < .3:
        saved = (list(RAN.PAC1PATH),list(RAN.PAC2PATH),list(RAN.OUT))
        
        score = self.improve(gameState,start,.3)
        
        if score < best:
          best = score
          #print score
        else:
          RAN.PAC1PATH = saved[0]
          RAN.PAC2PATH = saved[1]
          RAN.OUT = saved[2]
    '''
    offense = False
    if RAN.PAC1 == self.index:
      if len(RAN.PAC1PATH) == 0:
        if gameState.getAgentState(self.index).numCarrying > 0:
          return self.find_best_back(gameState)
        else:
          offense = True
      else:
        dest = RAN.PAC1PATH[0]
    elif len(RAN.PAC2PATH) == 0: 
      if gameState.getAgentState(self.index).numCarrying > 0:
        return self.find_best_back(gameState)
      else:
        offense = True
    else:
      dest = RAN.PAC2PATH[0]

    actions = gameState.getLegalActions(self.index)
    acts = {}
    for act in actions:
      acts[act] = gameState.generateSuccessor(act)
    acts= self.towardsSafety(gameState,acts)
    if offense:
      acts = self.killOpp(gameState,acts)
      acts = self.attackOpp2(gameState,acts)
      return random.choice(acts.keys())
    actions = list(acts.keys())
    cur_dist = self.getMazeDist(dest,gameState.getAgentPosition(self.index))
    vals = []
    for act in actions:

      succ = gameState.generateSuccessor(act)

      next_dist = self.getMazeDist(dest,succ.getAgentPosition(self.index))

      vals.append(next_dist)
      
    minVal = min(vals)
    best_acts = [a for a,v in zip(actions,vals) if v == minVal]

    return random.choice(best_acts)

  # moves dots around until convergence (or until time.time() - start > limit)
  def improve(self,gameState,start,limit):

    if self.red:
      food = gameState.getBlueFood().asList()
    else:
      food = gameState.getRedFood().asList()


    while len(RAN.OUT) < 2:
      if len(RAN.PAC1PATH) > 0:
        RAN.OUT.append(RAN.PAC1PATH.pop())
      elif len(RAN.PAC2PATH) > 0:
        RAN.OUT.append(RAN.PAC2PATH.pop())
      else:
        break

    pac1 = gameState.getAgentPosition(RAN.PAC1)
    pac2 = gameState.getAgentPosition((RAN.PAC1 + 2) % 4)
    path1,path2 = self.calc_path_lens(gameState,RAN.PAC1PATH,RAN.PAC2PATH)
    best = (list(RAN.PAC1PATH),list(RAN.PAC2PATH),list(RAN.OUT),path1,path2)
    scaler = -(len(food) / 8.0)
    random.shuffle(food)
    end = 0
    while time.time() - start < limit:
      RAN.ITERS += 1
      while RAN.QUEUE.isEmpty() and time.time() - start < limit:
        self.complete_reset(food)
        test_state = (tuple(RAN.PAC1PATH),tuple(RAN.PAC2PATH),tuple(RAN.OUT))
        if test_state not in RAN.CLOSED:
          path1,path2 = self.calc_path_lens(gameState,RAN.PAC1PATH,RAN.PAC2PATH)
          RAN.QUEUE.push(test_state,max(path1,path2))
          RAN.CLOSED.add(test_state)
      if RAN.QUEUE.isEmpty():
        break

      next_state = RAN.QUEUE.pop()
      if len(food) != len(RAN.PAC1PATH) + len(RAN.PAC2PATH) + len(RAN.OUT):
        continue
      path1,path2 = self.calc_path_lens(gameState,RAN.PAC1PATH,RAN.PAC2PATH)
      if max(path1,path2) * .9 > max(best[3],best[4]) and RAN.ITERS > 100:
        RAN.ITERS = 0
        RAN.QUEUE = util.PriorityQueue()
        if len(RAN.WEIGHTS) > 0:
          self.targeted_reset(gameState,food)
        else:
          self.complete_reset(food)
      else:
        RAN.PAC1PATH = list(next_state[0])
        RAN.PAC2PATH = list(next_state[1])
        RAN.OUT = list(next_state[2])

      end += 12
      if end > len(food):
        end = 12
      for fd in food[max(0,end - 12):end]:

        if fd in RAN.OUT:
          ops = []
          #looks at cases where out switches with something in pac1's path
          if len(RAN.PAC1PATH) > 1:
            change = (self.getMazeDist(pac1,fd) + 
              self.getMazeDist(fd,RAN.PAC1PATH[1]) - 
                self.getMazeDist(pac1,RAN.PAC1PATH[0]) -
                self.getMazeDist(RAN.PAC1PATH[0],RAN.PAC1PATH[1]))
            ops.append(exp((max(path1 + change,path2) - max(path1,path2)) / scaler))

            for i in range(1,len(RAN.PAC1PATH) - 1):
              change = (self.getMazeDist(RAN.PAC1PATH[i-1],fd) + 
              self.getMazeDist(fd,RAN.PAC1PATH[i+1]) - 
                self.getMazeDist(RAN.PAC1PATH[i-1],RAN.PAC1PATH[i]) -
                self.getMazeDist(RAN.PAC1PATH[i],RAN.PAC1PATH[i+1]))
              ops.append(exp((max(path1 + change,path2) - max(path1,path2)) / scaler))
            change = (self.getMazeDist(RAN.PAC1PATH[-2],fd) + 
              self.get_best_dist(gameState,fd) - 
                self.getMazeDist(RAN.PAC1PATH[-2],RAN.PAC1PATH[-1]) -
                self.get_best_dist(gameState,RAN.PAC1PATH[-1]))
            ops.append(exp((max(path1 + change,path2) - max(path1,path2)) / scaler))
          elif len(RAN.PAC1PATH) == 1:
            change = (self.getMazeDist(pac1,fd) +
              self.get_best_dist(gameState,fd) - 
              self.getMazeDist(pac1,RAN.PAC1PATH[0]) - 
              self.get_best_dist(gameState,RAN.PAC1PATH[0]))
            ops.append(exp((max(path1 + change,path2) - max(path1,path2)) / scaler))

          #looks at cases where out switches with something in pac2's path
          if len(RAN.PAC2PATH) > 1:
            change = (self.getMazeDist(pac2,fd) + 
              self.getMazeDist(fd,RAN.PAC2PATH[1]) - 
                self.getMazeDist(pac2,RAN.PAC2PATH[0]) -
                self.getMazeDist(RAN.PAC2PATH[0],RAN.PAC2PATH[1]))
            ops.append(exp((max(path1,path2+change) - max(path1,path2)) / scaler))
            for i in range(1,len(RAN.PAC2PATH) - 1):
              change = (self.getMazeDist(RAN.PAC2PATH[i-1],fd) + 
              self.getMazeDist(fd,RAN.PAC2PATH[i+1]) - 
                self.getMazeDist(RAN.PAC2PATH[i-1],RAN.PAC2PATH[i]) -
                self.getMazeDist(RAN.PAC2PATH[i],RAN.PAC2PATH[i+1]))
              ops.append(exp((max(path1,path2+change) - max(path1,path2)) / scaler))
            change = (self.getMazeDist(RAN.PAC2PATH[-2],fd) + 
              self.get_best_dist(gameState,fd) - 
                self.getMazeDist(RAN.PAC2PATH[-2],RAN.PAC2PATH[-1]) -
                self.get_best_dist(gameState,RAN.PAC2PATH[-1]))
            ops.append(exp((max(path1,path2+change) - max(path1,path2)) / scaler))
          elif len(RAN.PAC2PATH) == 1:
            change = (self.getMazeDist(pac2,fd) + 
              self.get_best_dist(gameState,fd) - 
              self.getMazeDist(pac2,RAN.PAC2PATH[0]) - 
              self.get_best_dist(gameState,RAN.PAC2PATH[0]))
            ops.append(exp((max(path1,path2+change) - max(path1,path2)) / scaler))
          
          ops.append(1)
          ops = ops / numpy.sum(ops)
          operations = np.choice(len(ops),size = min(3,len(ops)), replace = False,p = ops)
          for operation in operations:
            pac1path = list(RAN.PAC1PATH)
            pac2path = list(RAN.PAC2PATH)
            out = list(RAN.OUT)
            if operation < len(pac1path):
              out.append(pac1path[operation])
              out.remove(fd)
              pac1path[operation] = fd
            elif operation < len(pac1path) + len(pac2path):
              operation -= len(pac1path)
              out.append(pac2path[operation])
              out.remove(fd)
              pac2path[operation] = fd

            new_state = (tuple(pac1path),tuple(pac2path),tuple(out))
            if new_state not in RAN.CLOSED:
              RAN.CLOSED.add(new_state)
              val1,val2 = self.calc_path_lens(gameState,RAN.PAC1PATH,RAN.PAC2PATH)
              RAN.QUEUE.push(new_state,max(val1,val2))
          
        else:
          ops = []
          #calculates how much fd currently costs
          cur_cost1 = 0
          cur_cost2 = 0
          cur_cost1s = 0
          cur_cost2s = 0
          
          in1 = True
          if fd in RAN.PAC1PATH:
            idx = RAN.PAC1PATH.index(fd)
            if idx == 0:
              cur_cost1 += self.getMazeDist(pac1,fd)
              first = lambda x: self.getMazeDist(pac1,x)
            else:
              cur_cost1 += self.getMazeDist(RAN.PAC1PATH[idx-1],fd)
              first = lambda x: self.getMazeDist(RAN.PAC1PATH[idx-1],x)
            if idx == len(RAN.PAC1PATH) - 1:
              cur_cost1 += self.get_best_dist(gameState,fd)
              cur_cost1s = cur_cost1
              second = lambda x: self.get_best_dist(gameState,x)
              if idx == 0:
                cur_cost1 -= self.get_best_dist(gameState,pac1)
              else:
                cur_cost1 -= self.get_best_dist(gameState,RAN.PAC1PATH[idx-1])
            else:
              cur_cost1 += self.getMazeDist(RAN.PAC1PATH[idx+1],fd)
              cur_cost1s = cur_cost1
              second = lambda x: self.getMazeDist(RAN.PAC1PATH[idx+1],x)
              cur_cost1 -= first(RAN.PAC1PATH[idx + 1])
          else:
            in1 = False
            idx = RAN.PAC2PATH.index(fd)
            if idx == 0:
              cur_cost2 += self.getMazeDist(pac2,fd)
              first = lambda x: self.getMazeDist(pac2,x)
            else:
              cur_cost2 += self.getMazeDist(RAN.PAC2PATH[idx-1],fd)
              first = lambda x: self.getMazeDist(RAN.PAC2PATH[idx-1],x)
            if idx == len(RAN.PAC2PATH) - 1:
              cur_cost2 += self.get_best_dist(gameState,fd)
              second = lambda x: self.get_best_dist(gameState,x)
              cur_cost2s = cur_cost2
              if idx == 0:
                cur_cost2 -= self.get_best_dist(gameState,pac2)
              else:
                cur_cost2 -= self.get_best_dist(gameState,RAN.PAC2PATH[idx-1])
            else:
              cur_cost2 += self.getMazeDist(RAN.PAC2PATH[idx+1],fd)
              cur_cost2s = cur_cost2
              second = lambda x: self.getMazeDist(RAN.PAC2PATH[idx+1],x)
              cur_cost2 -= first(RAN.PAC2PATH[idx + 1])

          #check cost of inserting food into different places in PAC1PATH
          if len(RAN.PAC1PATH) > 0:
            if RAN.PAC1PATH[0] != fd:
              alt_cost1 = (self.getMazeDist(pac1,fd) + 
                self.getMazeDist(fd,RAN.PAC1PATH[0]) - 
                self.getMazeDist(RAN.PAC1PATH[0],pac1))
              cost1 = path1 - cur_cost1 + alt_cost1
              cost2 = path2 - cur_cost2
              ops.append(exp((max(cost1,cost2) - max(path1,path2)) / scaler))

            for i in range(0,len(RAN.PAC1PATH) - 1):
              if RAN.PAC1PATH[i] == fd or RAN.PAC1PATH[i+1] == fd:
                continue
              alt_cost1 = (self.getMazeDist(RAN.PAC1PATH[i],fd) +
                self.getMazeDist(RAN.PAC1PATH[i+1],fd) - 
                self.getMazeDist(RAN.PAC1PATH[i],RAN.PAC1PATH[i+1]))
              cost1 = path1 - cur_cost1 + alt_cost1
              cost2 = path2 - cur_cost2
              ops.append(exp((max(cost1,cost2) - max(path1,path2)) / scaler))
            if RAN.PAC1PATH[-1] != fd:
              alt_cost1 = (self.getMazeDist(RAN.PAC1PATH[-1],fd) + 
                self.get_best_dist(gameState,fd) - 
                self.get_best_dist(gameState,RAN.PAC1PATH[-1]))
              cost1 = path1 - cur_cost1 + alt_cost1
              cost2 = path2 - cur_cost2
              ops.append(exp((max(cost1,cost2) - max(path1,path2)) / scaler))
          elif len(RAN.PAC1PATH) == 0:
            alt_cost1 = (self.getMazeDist(pac1,fd) + 
              self.get_best_dist(gameState,fd))
            cost1 = alt_cost1
            cost2 = path2 - cur_cost2
            ops.append(exp((max(cost1,cost2) - max(path1,path2)) / scaler))
          
          #check cost of inserting food into different places in PAC2PATH
          if len(RAN.PAC2PATH) > 0:
            if RAN.PAC2PATH[0] != fd:
              alt_cost2 = (self.getMazeDist(pac2,fd) + 
                self.getMazeDist(fd,RAN.PAC2PATH[0]) - 
                self.getMazeDist(RAN.PAC2PATH[0],pac2))
              cost1 = path1 - cur_cost1
              cost2 = path2 - cur_cost2 + alt_cost2
              ops.append(exp((max(cost1,cost2) - max(path1,path2)) / scaler))
            for i in range(0,len(RAN.PAC2PATH) - 1):
              if RAN.PAC2PATH[i] == fd or RAN.PAC2PATH[i+1] == fd:
                continue
              alt_cost2 = (self.getMazeDist(RAN.PAC2PATH[i],fd) +
                self.getMazeDist(RAN.PAC2PATH[i+1],fd) - 
                self.getMazeDist(RAN.PAC2PATH[i],RAN.PAC2PATH[i+1]))
              cost1 = path1 - cur_cost1
              cost2 = path2 - cur_cost2 + alt_cost2
              ops.append(exp((max(cost1,cost2) - max(path1,path2)) / scaler))
            if RAN.PAC2PATH[-1] != fd:
              alt_cost2 = (self.getMazeDist(RAN.PAC2PATH[-1],fd) + 
                self.get_best_dist(gameState,fd) - 
                self.get_best_dist(gameState,RAN.PAC2PATH[-1]))
              cost1 = path1 - cur_cost1
              cost2 = path2 - cur_cost2 + alt_cost2
              ops.append(exp((max(cost1,cost2) - max(path1,path2)) / scaler))
          elif len(RAN.PAC2PATH) == 0:
            alt_cost2 = (self.getMazeDist(pac2,fd) + 
              self.get_best_dist(gameState,fd))
            cost1 = path1 - cur_cost1
            cost2 = alt_cost2
            ops.append(exp((max(cost1,cost2) - max(path1,path2)) / scaler))

          ops.append(1)

          ops = ops / numpy.sum(ops)
          operations = np.choice(len(ops),size = min(1,len(ops)),replace = False,p = ops)

          for operation in operations:
            pac1path = list(RAN.PAC1PATH)
            pac2path = list(RAN.PAC2PATH)
            if in1:
              if operation < len(pac1path) - 1:
                
                if operation < idx:
                  pac1path.remove(fd)
                  pac1path.insert(operation,fd)
                else:
                  pac1path.remove(fd)
                  pac1path.insert(operation + 1,fd)
                
              elif operation < len(pac1path) + len(pac2path):
                operation -= len(pac1path) - 1
                pac1path.remove(fd)
                pac2path.insert(operation,fd)
            else:
              if operation < len(pac1path) + 1:
                pac2path.remove(fd)
                pac1path.insert(operation,fd)
              elif operation < len(pac1path) + len(pac2path):
                operation -= len(pac2path) + 1
                if operation < idx:
                  pac2path.remove(fd)
                  pac2path.insert(operation,fd)
                else:
                  pac2path.remove(fd)
                  pac2path.insert(operation + 1,fd)
          
            new_state = (tuple(pac1path),tuple(pac2path),tuple(RAN.OUT))
            if new_state not in RAN.CLOSED:
              RAN.CLOSED.add(new_state)
              val1,val2 = self.calc_path_lens(gameState,pac1path,pac2path)
              RAN.QUEUE.push(new_state,max(val1,val2))


        path1,path2 = self.calc_path_lens(gameState,RAN.PAC1PATH,RAN.PAC2PATH)
        if max(path1,path2) < max(best[3],best[4]):
          best = (list(RAN.PAC1PATH),list(RAN.PAC2PATH),list(RAN.OUT),path1,path2)
          #print max(path1,path2)


    RAN.PAC1PATH = list(best[0])
    RAN.PAC2PATH = list(best[1])
    RAN.OUT = list(best[2])
    path1 = best[3]
    path2 = best[4]

    return max(path1,path2)

  # returns action that leads to returning to one's side the qucikest
  def find_best_back(self,gameState):

    best_score = 100000
    actions = gameState.getLegalActions(self.index)
    best_act = actions[0]

    for act in actions:

      succ = gameState.generateSuccessor(act)
      pos = succ.getAgentPosition(self.index)
      best = self.get_best_dist(succ, pos)
      if best < best_score:
        best_act = act
        best_score = best

    return best_act

  # gets the distance to return back to my side
  def get_best_dist(self, gameState, pos):

    if gameState.isRed(pos) == self.red:
      return -1

    if pos in self.dists:
      return self.dists[pos]

    x = gameState.data.layout.width / 2
    if self.red:
      x -= 1
    else:
      x += 1
    y_ops = [i for i in range(gameState.data.layout.height)]

    best = 1000000
    for y in y_ops:
      if not gameState.hasWall(x,y):

        test = self.getMazeDist((x,y), pos)

        if test < best:
          best = test
    return best

  # calculates lengths of each pacman's path
  def calc_path_lens(self,gameState,p1,p2):

    if self.red:
      food = gameState.getBlueFood().asList()
    else:
      food = gameState.getRedFood().asList()

    pa1 = 0
    pac1 = gameState.getAgentPosition(RAN.PAC1)
    pa2 = 0
    pac2 = gameState.getAgentPosition((RAN.PAC1 + 2) % 4)

    #calculate length of current paths
    cur = pac1
    for i in range(len(p1)):
      fd = p1[i]
      pa1 += self.getMazeDist(cur,fd)
      cur = fd

    pa1 += self.get_best_dist(gameState,cur)

    cur = pac2
    for i in range(len(p2)):
      fd = p2[i]
      pa2 += self.getMazeDist(cur,fd)
      cur = fd
      
    pa2 += self.get_best_dist(gameState,cur)

    return pa1,pa2

  #checks if path1 is the length of p1 and path2 is the length of p2
  # for debugging purposes, does nothing when called at the moment
  def check_path(self,gameState,p1,p2,path1,path2):
    return
    val1,val2 = self.calc_path_lens(gameState,p1,p2)
    if val1 != path1 or val2 != path2:
      print p1
      print p2
      print val1
      print path1
      print val2
      print path2
      sys.exit('error message')

  def getMazeDist(self, pos1, pos2):

    val = self.getMazeDistance(pos1,pos2,self.gameState)
    return val

  # Returns best moves for getting safe
  def towardsSafety(self, gameState, acts):

    state = gameState.getAgentState(self.index)
    foodHeld = state.numCarrying
    foodRemaining = len(self.getFood(gameState).asList())
    head_back = foodRemaining <= 2 or foodHeld >= state.getFoodCapacity()

    # looks at safety back for every path choice
    safe_dists = {}
    for act in acts:
      safe_path = self.inferiorSafeMethod(acts[act])
      if not safe_path[0]:
        unsafe_path = self.getUnsafePath(acts[act])
        if unsafe_path[0]:
          safe_dists[act] = (2, unsafe_path[1])
      else:
        safe_dists[act] = (0, safe_path[1])

    # only keeps paths with optimal safety
    if len(safe_dists.keys()) == 0:

      return acts

    vals = [safe_dists[act][0] for act in safe_dists]
    minVal = min(vals)
    best_acts = {}
    best_dist = 100000
    for act in safe_dists:
      if safe_dists[act][0] == minVal:
        if not head_back:
          best_acts[act] = acts[act]
        else:
          if safe_dists[act][1] == best_dist:
            best_acts[act] = acts[act]
          elif safe_dists[act][1] < best_dist:
            best_dist = safe_dists[act][1]
            best_acts = {}
            best_acts[act] = acts[act]

    return best_acts    

  # If red is true/false, gets shortest distance to red/blue side
  def bestDistBack(self, red, gameState, pos):

    if red == gameState.isRed(pos):
      return 0

    x = gameState.data.layout.width / 2
    if red:
      x -= 1
    else:
      x += 1
    y_ops = [i for i in range(gameState.data.layout.height)]

    best = 1000000
    for y in y_ops:
      if not gameState.hasWall(x,y):
        test = self.getMazeDistance((x, y), pos, gameState)
        if test < best:
          best = test
    return best    

  # Returns if there is and unsafe path where I probably/maybe
  # won't immediately die and the distance back
  def getUnsafePath(self, gameState):
    if gameState.getAgentState(self.index).getIsRespawning():
      return False, 0
    my_pos = gameState.getAgentPosition(self.index)
    if self.safeMove(my_pos, 1, gameState, True):
      return True, self.bestDistBack(self.red, gameState, my_pos)
    else:
      #print('unsafe')
      #for opp in RAN.opp_info:
      #    print(gameState.getAgentState(opp.idx).scaredTimer)
      return False, 0

  # More efficient less comprehensive safe method
  def inferiorSafeMethod(self, gameState):
    my_agent = gameState.getAgentState(self.index)
    my_pos = gameState.getAgentPosition(self.index)
    my_dist_back = self.bestDistBack(self.red, gameState, my_pos) + 1
    alt_dist = my_dist_back

    my_caps = self.getCapsules(gameState)
    for cap in my_caps:
      cap_loc = cap.getPosition()
      test_dist = self.getMazeDistance(cap_loc, my_pos, gameState)
      if alt_dist == None or test_dist < alt_dist:
        alt_dist = test_dist


    if not self.safeMove(my_pos,1,gameState,True):
      return False,0

    for opp in RAN.opp_info:

      opp_agent = gameState.getAgentState(opp.idx)
      speed_ratio = opp_agent.getSpeed() / my_agent.getSpeed()

      if opp_agent.getIsRespawning():
        continue
      elif opp_agent.scaredTimer > 0:
        continue
      for pos in opp.loc_distrib:
        opp_dist_back = self.bestDistBack(self.red,gameState,pos)
        dist_to_me = self.getMazeDistance(my_pos, pos, gameState)
        if math.floor(dist_to_me / speed_ratio) <= alt_dist / 2 or (
          opp_dist_back * speed_ratio < my_dist_back and
          math.floor(dist_to_me / speed_ratio) <= alt_dist):
          return False,0

    return True, my_dist_back

  def safeMove(self, new_pos, moves_to_new_pos, gameState, lasers):
    my_state = gameState.getAgentState(self.index)
    if my_state.getIsRespawning():
      return False
    invis = my_state.getInvisibility()
    if invis == 2:
      danger_range = 2
    elif invis == 1:
      danger_range = 5
    else:
      danger_range = None
    for opp in RAN.opp_info:
      opp_state = gameState.getAgentState(opp.idx)
      if opp_state.isRespawning:
        continue
      speed_ratio = opp_state.getSpeed() / my_state.getSpeed()

      opp_moves = math.ceil(moves_to_new_pos * speed_ratio)
      if opp_state.scaredTimer > 0 and not (self.red == gameState.isRed(new_pos)
        and my_state.scaredTimer > 0):
        continue
      for pos in opp.loc_distrib:
        if (util.manhattanDistance(pos,new_pos) - opp_moves + 1 <= 2 and
          opp_state.getBlastPower() > 0) and opp_state.scaredTimer == 0:
          return False
        elif self.getMazeDistance(pos,new_pos,gameState) <= opp_moves and (
          (my_state.scaredTimer > 0 and self.red == gameState.isRed(new_pos))
          or (self.red != gameState.isRed(new_pos) and opp_state.scaredTimer == 0)):
          return False
        elif lasers and opp_state.scaredTimer == 0 and len(opp.loc_distrib) == 1:
          ran = 0
          if danger_range == 2 and opp_state.getLaserPower() > 0:
            ran = 2
          elif (danger_range == 5 and opp_state.getLaserPower() > 0
            ) or (opp_state.getLaserPower() == 1 and danger_range == None):
            ran = 5
          elif opp_state.getLaserPower() == 2:
            ran = None

          add_x = 1
          while not gameState.hasWall(
            new_pos[0] + add_x, new_pos[1]) and (ran == None or
            add_x <= ran):
            if self.getMazeDistance(
            (new_pos[0] + add_x,new_pos[1]),pos,gameState) < opp_moves:
              return False
            add_x += 1

          sub_x = 1
          while not gameState.hasWall(
            new_pos[0] - sub_x, new_pos[1]) and (ran == None or
            sub_x <= ran):
            if self.getMazeDistance(
            (new_pos[0] - sub_x,new_pos[1]),pos,gameState) < opp_moves:
              return False
            sub_x += 1

          add_y = 1
          while not gameState.hasWall(
            new_pos[0], new_pos[1] + add_y) and (ran == None or
            add_y <= ran):
            if self.getMazeDistance(
            (new_pos[0],new_pos[1] + add_y),pos,gameState) < opp_moves:
              return False
            add_y += 1

          sub_y = 1
          while not gameState.hasWall(
            new_pos[0], new_pos[1] - sub_y) and (ran == None or
            sub_y <= ran):
            if self.getMazeDistance(
            (new_pos[0],new_pos[1] - sub_y),pos,gameState) < opp_moves:
              return False
            sub_y += 1


    return True

  def killOpp(self, gameState, acts):
    act_keys = acts.keys()
    vals = []
    for act in acts:
      opps_killed = 0
      for opp in self.opp_info:
        prev_opp_state = gameState.getAgentState(opp.idx)
        new_opp_state = acts[act].getAgentState(opp.idx)
        if prev_opp_state.deathCount < new_opp_state.deathCount:
          opps_killed += 1
      vals.append(opps_killed)

    kill_count = max(vals)
    best_acts = [a for a,v in zip(act_keys,vals) if v == kill_count]
    for act in act_keys:
      if act not in best_acts:
        del acts[act]

    if len(acts.keys()) > 1 and Directions.BLAST in acts:
      del acts[Directions.BLAST]

    return acts

  # Chooses moves that minimizes distance to opponent on my side
  def attackOpp2(self, gameState, acts):

    # Looks to see which opponents should be chased down
    to_chase = []
    for opp in RAN.opp_info:
      for pos in opp.loc_distrib:
        if gameState.isRed(pos) == self.red:
          to_chase.append(pos)
          break

    if len(to_chase) == 0:
      return acts

    # if necessary minimizes distance to opponent's that need to be chased
    vals = []
    act_lst = acts.keys()
    other_pos = gameState.getAgentPosition((self.index + 2) % 4)
    for act in act_lst:
      my_pos = acts[act].getAgentPosition(self.index)
      my_dists = []
      other_dists = []
      for pos in to_chase:
        my_dists.append(self.getMazeDistance(my_pos,pos,acts[act]))
        other_dists.append(self.getMazeDistance(other_pos,pos,acts[act]))

      if len(to_chase) == 1:
        vals.append(min(my_dists[0],other_dists[0]))
      else:
        vals.append(min(my_dists[0] + other_dists[1],
          my_dists[1] + other_dists[0]))

    bestVal = min(vals)
    best_acts = [a for a,v in zip(act_lst,vals) if v == bestVal]
    new_acts = {}
    for act in best_acts:
      new_acts[act] = acts[act]
    return new_acts

class PowersInference:

    def __init__(self):
        self.A = [ [0, 0, 0, 1, 1],
                   [1, 0, 1, 1, 0],
                   [0, 1, 1, 0, 1],
                   [1, 1, 0, 0, 0],
                   [1, 0, 0, 0, 0] ]

        self.A = numpy.array(self.A)
        self.assignment = {}
        self.prediction = {}

    # Used to generate possible sets
    def get_noisy_set(self, sums):
        b_set = [ numpy.array(sums + [x]) for x in range(3) ]
        x_set = [ numpy.linalg.solve(self.A, b) for b in b_set ]
        x_set = [ x for x in x_set if nonnegative(x) and sum(x) <= 4 ]

        return x_set

    # Collects all possible valid sets for given sums for agent index
    def infer(self, sums, index):
        sums  = list(sums)
        x_set = self.get_noisy_set(sums)

        self.assignment[index] = [ [ int(round(n)) for n in x ] for x in x_set ]
        return self.assignment[index]

    # Returns a single possible valid answer, given the sums for agent index
    def make_guess(self, index):
        if index not in self.prediction.keys():
            self.prediction[index] = self.assignment[index][0]
        return self.prediction[index]

    # Returns a set of possible values power can hold for agent index
    def get_knowledge(self, index, power):
        powers = {'invisibility': 0, 'speed': 1, 'laser': 2, 'respawn': 3, 'capacity': 4}
        query, p = set(), powers[power]

        for solution in self.assignment[index]:
            query.add( solution[p] )

        return query

# General known or infered information about a given opponent
class OppInfo:

  # Distributions when extra information about location was obtained about
  # where but not which
  extra_distribs = []
  opps = []
  lastState = None
  lastAgent = None


  #power dic holds 'laser', 'speed', 'capacity', 'invisibility', 'respawn''
  def __init__(self, idx, power_dic, start_loc):

    self.idx, self.loc = idx, start_loc
    self.power_dic = power_dic
    self.loc_distrib = set( [self.loc] )

    OppInfo.opps.append(self)

  # accounts for opponent having eaten food in estimating location and returns
  # number of food opponent had eaten since last checked
  def recently_eaten(self, agent, gameState):
    prev_op_agent = OppInfo.lastState.getAgentState(self.idx)
    cur_op_agent = gameState.getAgentState(self.idx)

    if prev_op_agent.numCarrying < cur_op_agent.numCarrying:
      prevFood = agent.getFoodYouAreDefending(OppInfo.lastState).asList()
      curFood = agent.getFoodYouAreDefending(gameState).asList()
      eaten = [ f for f in prevFood if f not in curFood ]
      if len(eaten) + prev_op_agent.numCarrying == cur_op_agent.numCarrying:
        self.loc_distrib = set(eaten)
        self.loc = eaten[0]
        return len(eaten)
    return 0

  def time_update(self, agent, gameState, eaten):
    time_diff = OppInfo.lastState.data.timeleft - gameState.data.timeleft

    if time_diff <= 1:
      return

    cur_op_agent = gameState.getAgentState(self.idx)

    # need to consider two steps before I take one
    agent_state = gameState.getAgentState(agent.index)
    other_opp = OppInfo.opps[0]
    if other_opp == self:
      other_opp = OppInfo.opps[1]
    other_state = gameState.getAgentState(other_opp.idx)

    if agent.index == OppInfo.lastAgent.index:
      if time_diff == 3 or cur_op_agent.getSpeed() > other_state.getSpeed():
        if eaten == 0:
          self.guessPos(gameState,agent)
        else:
          eaten -= 1
    elif agent.index > OppInfo.lastAgent.index:
      if self.idx < agent.index and self.idx > OppInfo.lastAgent.index:
        if eaten == 0:
          self.guessPos(gameState,agent)
        else:
          eaten -= 1
    else:
      if self.idx == 3 or self.idx == 0:
        if eaten == 0:
          self.guessPos(gameState,agent)
        else:
          eaten -= 1
      if time_diff >= 4:
        if eaten == 0:
          self.guessPos(gameState,agent)
        else:
          eaten -= 1
      if time_diff == 3 and cur_op_agent.getSpeed() > other_state.getSpeed():
        if eaten == 0:
          self.guessPos(gameState,agent)
        else:
          eaten -= 1

  def updatePos(self, agent, gameState):

    if gameState.getAgentState(self.idx).isRespawning:
      self.loc = gameState.getInitialAgentPosition(self.idx)
      self.loc_distrib = set([self.loc])
      return
    elif gameState.getAgentPosition(self.idx) != None:
      self.loc = gameState.getAgentPosition(self.idx)
      self.loc_distrib = set([self.loc])
      return

    # looks to see if food has been eaten and updates position accordingly
    eaten = self.recently_eaten( agent, gameState )
    if OppInfo.lastAgent != None:
      self.time_update( agent, gameState, eaten)

    # if no knowledge, take all on border we know opponent is on
    if len(self.loc_distrib) == 0:
      x_val = gameState.data.layout.width / 2
      if agent.red == gameState.getAgentState(self.idx).isPacman:
        x_val -= 1
      y_vals = range(0,gameState.data.layout.height - 1)
      for y in y_vals:
        if not gameState.hasWall(x_val,y):
          self.loc_distrib.add((x_val,y))

    my_pos = gameState.getAgentPosition(agent.index)
    oth_pos = gameState.getAgentPosition((agent.index + 2) % 4)

    if len(self.loc_distrib) > 6:
      new_loc_distrib = set()
      while len(new_loc_distrib) < 6:
        for pos in [my_pos,oth_pos]:
          best = 0
          best_loc = None
          for loc in self.loc_distrib:
            new_dist = agent.getMazeDistance(loc,pos,gameState)
            if best_loc == None or new_dist < best:
              best = new_dist
              best_loc = loc
          self.loc_distrib.remove(best_loc)
          new_loc_distrib.add(best_loc)

      self.loc_distrib = new_loc_distrib

    '''
    dist = util.Counter()
    for pos in self.loc_distrib:
      dist[pos] = 1.0 / len(self.loc_distrib)

    if self.idx == 0:
      agent.displayDistributionsOverPositions([dist])
    '''

    #print('index ' + str(self.idx))
    #print(len(self.loc_distrib))

  def guessPos(self, gameState, agent):
    new_distrib = set()
    agent_pos = gameState.getAgentPosition(agent.index)
    for pos in self.loc_distrib:
      if self.possiblePos(pos, gameState, agent, agent_pos):
        new_distrib.add(pos)
      if self.possiblePos((pos[0] + 1, pos[1]), gameState, agent, agent_pos):
        new_distrib.add((pos[0] + 1, pos[1]))
      if self.possiblePos((pos[0] - 1, pos[1]), gameState, agent, agent_pos):
        new_distrib.add((pos[0] - 1, pos[1]))
      if self.possiblePos((pos[0], pos[1] + 1), gameState, agent, agent_pos):
        new_distrib.add((pos[0], pos[1] + 1))
      if self.possiblePos((pos[0], pos[1] - 1), gameState, agent, agent_pos):
        new_distrib.add((pos[0], pos[1] - 1))
    self.loc_distrib = new_distrib

  def possiblePos(self, pos, gameState, agent, agent_pos):
    op_agent = gameState.getAgentState(self.idx)
    if gameState.hasWall(pos[0], pos[1]) or (op_agent.isPacman == (agent.red
      != gameState.isRed(pos))):
      return False
    elif ( agent.getMazeDistance(pos, agent_pos, gameState ) > 5 and
        op_agent.getInvisibility() == 1):
      return True
    elif ( agent.getMazeDistance(pos, agent_pos, gameState ) > 2 and
        op_agent.getInvisibility() == 2):
      return True
    else:
      return False

  def getPos(self):
    if len( self.loc_distrib ) == 1:
      self.loc = list( self.loc_distrib )[0]
      return self.loc
    else:
      # there are better ways of doing this, we'll see how important it
      # is that we do this in a good way
      return random.choice( list(self.loc_distrib) )

  def predict_goal(self):
    x, y = self.getPos()
    # Fix to be mazeDistance 5 or less
    cluster = [ (x + i, y + j) for i in range(-5, 6) for j in range(-5, 6) ]
    cluster = [ p for p in cluster if self.pather.Position.legal(p) ]

    border, capsules = self.lastState.width / 2, self.lastState.getCapsules()
    targets = [ p for p in cluster if self.lastState.hasFood(p) or p in capsules ]
    cluster = targets if ( border in capsules or border not in cluster ) else targets + [ border ]

    if len( cluster ) == 0:
      # If no easily predicable goal, returns the position. This leads to poor
      # interception performance, but will at least chance the opponent down.
      return [ (x, y) ]
    # Will fix
    return random.choice(cluster)
