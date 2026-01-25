"""
POLARIS Environment Server
Minimal server wrapper following modern OpenReward pattern.
"""
from openreward.environments import Server
from polaris import Polaris

if __name__ == "__main__":
    Server([Polaris]).run()
