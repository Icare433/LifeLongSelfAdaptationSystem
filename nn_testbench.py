import json

import Q_learner

agent = Q_learner.Agent(10, 8, 15)
experience_file = open("experiences_first.txt", "r")
for line in experience_file.readlines():
    experience = json.loads(line)
    agent.memory.add(experience)
for _ in range(500):
    agent.train_network()
