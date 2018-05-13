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

from baselineTeam import OffensiveReflexAgent, DefensiveReflexAgent
#################
# Team creation #
#################
class Team:
  def createAgents(self, firstIndex, secondIndex, isRed, gameState,
                   first = 'JacksonSipple', second = 'JacksonSipple'):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.
    """
    self.agents = [eval(first)(firstIndex, gameState), eval(second)(secondIndex, gameState)]
    self.isRed = isRed

    return self.agents

  def chooseCapsules(self, gameState, capsuleLimit, opponentEvidences):
     P = PowersInference()
     e_1 = [ opponentEvidences[0]['sum0'], opponentEvidences[0]['sum1'], opponentEvidences[0]['sum2'], opponentEvidences[0]['sum3'] ]
     e_2 = [ opponentEvidences[1]['sum0'], opponentEvidences[1]['sum1'], opponentEvidences[1]['sum2'], opponentEvidences[1]['sum3'] ]

     P.infer(e_1, 0)
     P.infer(e_2, 1)

     #Tells us maze distances
     distancer = distanceCalculator.Distancer(gameState.data.layout)
     distancer.getMazeDistances()

     kmeans = KMeans( gameState, self.isRed )
     food = gameState.getRedFood().asList() if not self.isRed else gameState.getBlueFood().asList()
     scare_locations = kmeans.pick_capsules( food, distancer )

     capsules = []
     for location in scare_locations:
         x, y = location
         capsules.append( ScareCapsule(x, y) )

     """
     grenade_locations = self.pick_grenades( gameState, distancer )

     for (x, y) in grenade_locations:
         capsules.append( GrenadeCapsule(x, y) )

     """

     return capsules

  def pick_grenades(self, gameState, distancer):

    if self.isRed:
        idxes = gameState.getRedTeamIndices()
    else:
        idxes = gameState.getBlueTeamIndices()

    grenades = []
    for idx in idxes:
        pos = gameState.getInitialAgentPosition(idx)
        tests = [(pos[0] + 1, pos[1]), (pos[0] - 1, pos[1]), (pos[0], pos[1] + 1),
                    (pos[0], pos[1] - 1)]
        for test in tests:
            if not gameState.hasWall(test[0], test[1]) and test not in grenades:
                grenades.append(test)
                break
    return grenades

##########
# Agents #
##########
argmin = lambda x: min( range(len(x)), key = lambda i: x[i] )
nonnegative = lambda x: all( [n >= 0 for n in x] )
step_away = lambda x, l: [ y for y in l if manhattan(x, y) == 1 ]

# Prioritizes staying alive. Then prioritizes killing opponent. Then
# prioritizes attacking opponents when they are on my side (one pacman
# per opponent). Then and only then prioritizes eating food.
# Thought capacity was useless. Scare pellets seemed to work best
# (made logic for grenades but not as good)
# Named after Jackson Sipple for no particular reason except 
# to possibly confuse him later
class JacksonSipple(CaptureAgent):

    ATTACK_OPP = None
    opp_info = []


    def choosePowers(self, gameState, powerLimit):
        return {'invisibility': 1, 'speed': 2, 'laser': 1, 'respawn': 0, 'capacity': 0}

    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)

        if JacksonSipple.ATTACK_OPP == None:
            JacksonSipple.ATTACK_OPP = random.randint(0,1)

        self.gameState = gameState
        self.sight = {}
        # initialize info about opponent

        if len(JacksonSipple.opp_info) == 0:
            power_dics = self.getOpponentTeamPowers(gameState)
            opps = self.getOpponents(gameState)
            for opp, power_dic in zip(opps,power_dics):
                initial_pos = gameState.getInitialAgentPosition(opp)
                JacksonSipple.opp_info.append(OppInfo(opp, power_dic, initial_pos))
            OppInfo.lastState = gameState

        # get information about bestDistBack from every opponent square
        self.best_back = {}
        for x in range(gameState.data.layout.width):
            for y in range(gameState.data.layout.height):
                if not gameState.hasWall(x,y):
                    self.best_back[(x, y)] = self.bestDistBack(
                        not gameState.isRed((x,y)),gameState,(x,y))

    def chooseAction(self, gameState):

        for inf in JacksonSipple.opp_info:
            inf.updatePos(self, gameState)
        OppInfo.lastState = gameState
        OppInfo.lastAgent = self

        actions = gameState.getLegalActions(self.index)
        acts = {}
        for act in actions:
            acts[act] = gameState.generateSuccessor(act)

        self.gameState = gameState
        acts = self.getGrenades(gameState,acts)
        acts = self.towardsSafety(gameState, acts)
        acts = self.killOpp(gameState, acts)
        if JacksonSipple.ATTACK_OPP == 0:
            acts = self.attackOpp(gameState, acts)
        else:
            acts = self.attackOpp2(gameState,acts)
        acts = self.eatFood(gameState, acts)

        return random.choice(acts.keys())

    def getPos(self):
        return self.gameState.getAgentPosition(self.index)

    def getGrenades(self, gameState, acts):
        if gameState.getAgentState(self.index).getBlastPower() == 0:
            for act in acts:
                if acts[act].getAgentState(self.index).getBlastPower() > 0:
                    return {act: acts[act]}
        return acts

    # Returns best moves for defending capsules
    def defendCapsules(self, gameState, acts):
        return acts

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

    # Returns best moves for chasing opponent
    def attackOpp(self, gameState, acts):
        if len( acts.keys() ) == 1:
            return acts

        values = {}
        for action in acts.keys():
            values[action] = 0

        for opponent in JacksonSipple.opp_info:
            distance = min( [self.getMazeDistance( gameState.getAgentPosition(self.index), position, gameState ) for position in opponent.loc_distrib] )
            if gameState.isRed( gameState.getAgentPosition(self.index) ) == self.red: #Defending
                bad_feeling = gameState.getAgentState(self.index).scaredTimer > distance
            else:
                bad_feeling = gameState.getAgentState(opponent.idx).scaredTimer < distance

            overkill = gameState.getAgentState(opponent.idx).getIsRespawning() # They're already dead dude
            overkill = overkill or self.deepInEnemyTerritory(opponent.getPos(), gameState)
            if bad_feeling or overkill:
                continue

            distances = self.intercept_opponent(opponent, acts)
            for action in distances.keys():
                values[action] += distances[action]

        actions  = acts.keys()
        best_val = min( values.values() )

        for action in actions:
            if values[action] > best_val:
                acts.pop(action)

        if len( acts.keys() ) > 1 and "Stop" in acts.keys():
            acts.pop("Stop")

        return acts

    def legal(self, position):
        negative = position[0] < 0 or position[1] < 0
        real_big = position[0] >= self.gameState.data.layout.width
        real_big = position[1] >= self.gameState.data.layout.height or real_big
        return not negative and not real_big and not self.gameState.hasWall(position[0], position[1])

    def intercept_opponent(self, opponent, actions):
        agent_distribution = [ actions[action].getAgentPosition( self.index ) for action in actions ]
        opponent_distribution = opponent.loc_distrib

        weights = self.chase_opponent( agent_distribution, opponent_distribution )
        control = weights[:2]
        regress = weights[2:]

        distances = {}
        for action in actions.keys():
            new_position = actions[action].getAgentPosition( self.index )
            distances[action] = numpy.dot( new_position, regress )

        return distances

    def chase_opponent(self, agent_distribution, opponent_distribution):
        coordinates = [ list(location) + list(position) for location in opponent_distribution for position in agent_distribution ]
        distances = [ self.getMazeDistance(position, location, self.gameState) for location in opponent_distribution for position in agent_distribution ]

        weights = numpy.linalg.lstsq( coordinates, distances, rcond = -1 )[0]
        return weights

    def expand_set(self, position, opponent_position, opponent_radius, trial=0):
        laser = self.gameState.getAgentState(self.index).getLaserPower()
        sanctuaries = [ location for location in opponent_radius if not self.interceptable(position, opponent_position, location, laser) ]

        if len( sanctuaries ) == 0 or trial == 5:
            return opponent_radius

        for haven in sanctuaries:
            movement = [ (haven[0] + i, haven[1] + j) for i in range(-1, 2) for j in range(-1, 2) if self.legal((haven[0] + i, haven[1] + j)) ]
            for new in movement:
                opponent_radius.add(new)

        for haven in sanctuaries:
            opponent_radius.remove(haven)

        return self.expand_set(position, opponent_position, opponent_radius, trial + 1)

    def interceptable(self, position, opponent_position, location, laser):
        if self.getMazeDistance(position, location, self.gameState) <= self.getMazeDistance(opponent_position, location, self.gameState):
            return True

        if laser == 0:
            return False

        limit = 6 if laser == 1 else 21
        targets = self.line_of_sight(location, position, opponent_position, limit)
        for target in targets:
            if self.getMazeDistance(position, target, self.gameState) < self.getMazeDistance(opponent_position, target, self.gameState):
                return True

        return False

    def line_of_sight(self, target, pos, o_pos, limit):
        if target in self.sight.keys():
            return self.sight[ target ]

        x_laser, increment = [], 1 if ( pos[0] - o_pos[0] ) > 0 else -1
        i = increment
        while self.legal( (o_pos[0] + i, o_pos[1]) ) and abs(i) < limit:
            x_laser += [ (o_pos[0] + i, o_pos[1]) ]
            i += increment

        y_laser, increment = [], 1 if ( pos[1] - o_pos[1] ) > 0 else -1
        j = increment
        while self.legal( (o_pos[0], o_pos[1] + j) ) and abs(j) < limit:
            y_laser += [ (o_pos[0], o_pos[1] + j) ]
            j += increment

        self.sight[ target ] = x_laser + y_laser
        return self.sight[ target ]

    def deepInEnemyTerritory(self, position, gameState):
        distance_to_border = self.bestDistBack( self.red, gameState, position )
        return distance_to_border > 7

    # Chooses moves that minimizes distance to opponent on my side
    def attackOpp2(self, gameState, acts):

        # Looks to see which opponents should be chased down
        to_chase = []
        for opp in JacksonSipple.opp_info:
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

    # Returns best moves for eating food
    def eatFood(self, gameState, acts):
        # first checks if a food would be eaten
        new_acts = {}
        for act in acts:
            new_food_len = len(self.getFood(acts[act]).asList())
            past_food_len = len(self.getFood(gameState).asList())
            if new_food_len < past_food_len:
                new_acts[act] = acts[act]

        if len(new_acts.keys()) > 0:
            acts = new_acts

        # now looks for optimal next food

        vals = []
        act_keys = list(acts.keys())
        ally_pos = gameState.getAgentPosition((self.index + 2) % 4)
        for act in act_keys:
            food = self.getFood(acts[act]).asList()
            my_pos = acts[act].getAgentPosition(self.index)
            best_score = 100000

            for fd in food:
                score = self.getMazeDistance(my_pos, fd, acts[act]
                    ) - .8 * self.getMazeDistance(my_pos, ally_pos, acts[act])
                if score < best_score:
                    best_score = score
            vals.append(best_score)

        minVal = min(vals)
        best_acts = [a for a,v in zip(act_keys,vals) if v == minVal]
        new_acts = {}
        for act in best_acts:
            new_acts[act] = acts[act]

        return new_acts

    # If red is true/false, gets shortest distance to red/blue side
    def bestDistBack(self, red, gameState, pos):

        if red == gameState.isRed(pos):
            return 0

        if pos in self.best_back:
            return self.best_back[pos]

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

    # Returns (1) if there is a safe path back (taking into account lasers)
    # and (2) the distance
    def getSafePath(self, gameState, lasers):

        pq = []
        my_start_pos = gameState.getAgentPosition(self.index)
        heapq.heappush(pq, (1, 1, my_start_pos))
        seen = set([my_start_pos])
        pseudo_lasers = True
        while pq:
            next_pos = heapq.heappop(pq)
            if not self.safeMove(next_pos[2],next_pos[1],gameState,pseudo_lasers):
                continue
            if not lasers:
                pseudo_lasers = False
            if self.red == gameState.isRed(next_pos[2]):
                return True,next_pos[1]

            tests = [(next_pos[2][0] + 1, next_pos[2][1]),
                    (next_pos[2][0] - 1, next_pos[2][1]),
                    (next_pos[2][0], next_pos[2][1] + 1),
                    (next_pos[2][0], next_pos[2][1] - 1)]
            for test in tests:
                if not gameState.hasWall(test[0], test[1]
                    ) and test not in seen:
                    seen.add(test)
                    heapq.heappush(pq, (self.bestDistBack(self.red,gameState,test) +
                        next_pos[1] + 1, next_pos[1] + 1, test))

        return False,0

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
            #for opp in JacksonSipple.opp_info:
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

        for opp in JacksonSipple.opp_info:

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
        for opp in JacksonSipple.opp_info:
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


# Determines solution sets that satisfy the system of equations established by
# the sums. Laser is ALWAYS correct.

# 46% of the time, every other power is also uniquely known. In 46% of cases,
# every other power is narrowed to two options. This is symmetric (ie, half of
# these pair exactly with the other half). The remaining three cases cannot be
# differentiated from one another.
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
        return random.choice( cluster )

class KMeans:
    def __init__(self, gameState, red):
        self.game_map = gameState
        self.red = red

    def pick_capsules(self, food, distancer):
        distance_f = lambda x, y: distancer.getDistance( x, y, self.game_map.getWalls() )
        mean_s = lambda x, i: sum([n[i] for n in x]) / len(x)
        mean_f = lambda x: ( mean_s(x, 0), mean_s(x, 1) )

        clusters = self.k_means(distance_f, mean_f, food, self.splitFoods(food, 8))

        fronts = [ self.front(n, distance_f) for n in clusters ]
        means  = [ mean_f(n) for n in clusters ]
        fronts, means = self.adjust(fronts), self.adjust(means)

        dist_indices = list(range(len(clusters)))
        dist_indices.sort(key = lambda i: distance_f( fronts[i], means[i] ))

        size_indices = list(range(len(clusters)))
        size_indices.sort(key = lambda i: - len(clusters[i]))

        capsules, i = set([ means[ size_indices[0] ], means[ size_indices[1] ] ]), 2
        while len(capsules) == 1:
            capsules.add( means[ size_indices[i] ] )
            i += 1

        i = 0
        while len(capsules) < 4:
            capsules.add( fronts[ dist_indices[i] ] )
            i += 1

        return list(capsules)

    def front(self, cluster, distance_f):
        front, best = random.choice( cluster ), 2**64
        width = self.game_map.data.layout.width / 2
        width = width - random.randint(1, 3) if self.red else width + random.randint(1, 3)

        for y in range(0, self.game_map.data.layout.height):
            point = (width, y)
            if not self.valid(point):
                continue

            min_dist = min([ distance_f(point, x) for x in cluster ])

            if best > min_dist:
                front, best = point, min_dist

        return front

    def valid(self, position):
        return self.game_map.isValidPosition(position, self.red)
        """
        negative = position[0] < 0 or position[1] < 0
        real_big = position[0] > self.game_map.data.layout.width
        real_big = position[1] > self.game_map.data.layout.height or real_big
        if negative or real_big:
            return False

        walled = self.game_map.hasWall( position[0], position[1] )
        fooded = self.game_map.hasFood( position[0], position[1] )
        return not walled and not fooded
        """
    def adjust(self, means):
        alpha = ( self.game_map.data.layout.width / 2, self.game_map.data.layout.height / 2)
        f = lambda x, y, z: int(round(x * z + y * (1 - z)))

        for i in range(len(means)):
            counter = 0.99
            while not self.valid(means[i]):
                    means[i] = ( f(means[i][0], alpha[0], counter), f(means[i][1], alpha[1], counter) )
                    counter -= 0.01
        return means

    def k_means(self, distance_f, mean_f, items, clusters, trials=0):
        clusters = [x for x in clusters if x]
        means = [ mean_f(cluster) for cluster in clusters ]

        means = self.adjust(means)
        distances = [ [distance_f(item, mean) for mean in means] for item in items]

        closest = [ argmin(distance_set) for distance_set in distances ]
        clusters_prime = [ [items[i] for i in range(len(items)) if closest[i] == j] for j in range(len(clusters)) ]

        differences = [[x for x in clusters[i] if not x in clusters_prime[i]] + [x for x in clusters_prime[i] if not x in clusters[i]] for i in range(len(clusters))]
        difference = any(differences)

        if not difference or trials > 10:
            return clusters_prime

        return self.k_means(distance_f, mean_f, items, clusters_prime, trials + 1)

    def splitFoods(self, foods, x):
        f, n = len(foods), (len(foods) + (x - 1)) // x
        split = [foods[i:i + n] for i in range(0, f, n)]
        return split
