import os
from argparse import ArgumentParser, Namespace
from agent_factory import AgentFactory
from dataset import Dataset
from environment import Environment

def parse_command_args():
    parser = ArgumentParser()

    parser.add_argument("-dp", "--data_path", dest="data_path",required=True,
                        help="specify the path to the data file stored in CSV format")
    parser.add_argument("-m", "--model", dest="model", required=True)
    parser.add_argument("-ws", "--window_size", dest="window_size", default=30)
    parser.add_argument("-ed", "--epsilon_decay", dest="epsilon_decay", default=0.001)

    args = parser.parse_args()

    return args

def create_simulation(args):
    location_id = 15356
    dataset = Dataset(args.data_path, location_id)
    env = Environment(dataset)
    env.train_environment()

    agent = AgentFactory.get_agent(env, args.model)

    agent.train(151)
    

if __name__ == '__main__':
    args = parse_command_args()
    create_simulation(args)
    