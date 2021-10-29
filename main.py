import os
from argparse import ArgumentParser, Namespace
from dataset import Dataset
from environment import Environment
from agent import Agent
from simulator import Simulator

def parse_command_args():
    parser = ArgumentParser()

    parser.add_argument("-dp", "--data_path", dest="data_path",required=True,
                        help="specify the path to the data file stored in CSV format")
    parser.add_argument("-ws", "--window_size", dest="window_size", default=30)
    parser.add_argument("-ed", "--epsilon_decay", dest="epsilon_decay", default=0.001)

    args = parser.parse_args()

    return args

def create_simulation(args):
    location_id = 14922
    dataset = Dataset(args.data_path, location_id)
    env = Environment(dataset)

    agent = Agent()
    
    simulator = Simulator(agent, env=env)

    for i in range(20):
        simulator.play_one_episode(i)

if __name__ == '__main__':
    args = parse_command_args()
    create_simulation(args)
    