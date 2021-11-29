from agents.DQN.dqn_agent import DQN_Agent
from agents.Rainbow_DQN.rainbow_dqn_agent import RainbowDQNAgent

class AgentFactory:
    def get_agent(env, type_agent="dqn"):
        agents = {
            "dqn": DQN_Agent,
            "rainbowDQN": RainbowDQNAgent
        }

        return agents[type_agent](env)