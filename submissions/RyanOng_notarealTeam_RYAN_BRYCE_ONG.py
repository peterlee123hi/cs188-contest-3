from captureAgents import CaptureAgent
import distanceCalculator
import util, random
from game import Directions
from capsule import ScareCapsule, GrenadeCapsule, ArmourCapsule, SonarCapsule, JuggernautCapsule
import contestPowersBayesNet
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
    return null
  def chooseCapsules(self, gameState, capsuleLimit, opponentEvidences):
    enemyAgentMarginals = []
    for evidence in opponentEvidences:
        marginals = contestPowersBayesNet.computeMarginals(evidence)
        enemyAgentMarginals.append(marginals)
    return null
