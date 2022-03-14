import math

import numpy as np

import Q_learner

import json


class TaskManager:

    def __init__(self, knowledge, knowledge_manager, learning_manager, task_based_knowledge_miner):

        self.task_based_knowledge_miner = task_based_knowledge_miner
        self.knowledge_manager = knowledge_manager
        self.learning_manager = learning_manager
        self.knowledge = knowledge

    def update_goal_model(self, new_goal_model):
        self.knowledge.goal_model = new_goal_model
        self.knowledge_manager.goalModel = new_goal_model
        self.knowledge_manager.save_file.write("changed\n")
        self.task_based_knowledge_miner.update_rewards()
        new_model = self.learning_manager.retrain()
        self.learning_manager.update_learning_model(new_model, self.knowledge_manager.get_experiences())


class TaskBasedKnowledgeMiner:

    def __init__(self, knowledge_manager):
        self.knowledge_manager = knowledge_manager

    def update_rewards(self):
        goal_model = self.knowledge_manager.get_current_goals()
        for experience in self.knowledge_manager.experiences:
            experience.reward = goal_model.evaluate(experience.next_state)


class KnowledgeManager:

    def __init__(self, learning_model, goal_model, cluster_detection, reclustering_interval = 10):

        self.experiences = list()
        self.learning_model = learning_model
        self.goal_model = goal_model
        self.last_experiences = dict()
        self.data_per_mote =dict()
        self.save_file = open("results.txt", "a")
        self.cluster_detection = cluster_detection
        self.reclustering_interval = reclustering_interval

    def get_current_goals(self):
        return self.goal_model

    def get_current_model(self):
        return self.learning_model

    def observe_state(self, mote, state, reward):
        self.save_file.write(json.dumps({mote:state}))
        self.save_file.write("\n")
        experience = Experience(state, reward, mote)
        if self.last_experiences.get(mote) is not None:
            self.last_experiences.get(mote).add_next_state(experience.state, experience.reward)
            self.data_per_mote[mote] = self.data_per_mote[mote] + 1
            self.experiences.append(self.last_experiences.get(mote))
        else:
            self.data_per_mote[mote] = 0
        self.last_experiences[mote] = experience

        recluster = False
        for mote in self.data_per_mote:
            if self.data_per_mote[mote] % self.reclustering_interval == 0:
                recluster = True
            else:
                recluster = False

        if recluster:
            self.cluster_detection.check_analysis()

    def observe_action(self, mote, action):
        if self.last_experiences.get(mote) is not None:
            self.last_experiences.get(mote).add_action(action)

    def get_experiences(self):
        memory = Q_learner.Memory(100000)
        for experience in self.experiences:
            state_vector = list()
            for transmission in experience.state:
                for value in list(transmission.values()):
                    state_vector.append(value)
            state_vector = np.array(state_vector, dtype=object)
            next_state_vector = list()
            for transmission in experience.next_state:
                for value in list(transmission.values()):
                    next_state_vector.append(value)
            next_state_vector = np.array(state_vector, dtype=object)
            memory.add({
                    "state": state_vector,
                    "action": experience.action,
                    "reward": experience.reward,
                    "next_state": next_state_vector
                })
        return memory


class Experience:

    def __init__(self, state, reward, mote):
        self.mote = mote
        self.state = state
        self.reward = reward
        self.action = None
        self.next_state = None

    def add_next_state(self,state,reward):
        self.next_state = state
        self.reward = reward

    def add_action(self,action):
        self.action = action


class LearningManager:

    def __init__(self, knowledge_manager,decision_making_component):
        self.knowledge_manager = knowledge_manager
        self.decision_making_component = decision_making_component

    def retrain(self, batch_size=32, discount=0.99):
        current_model = self.knowledge_manager.get_current_model()
        new_model = Q_learner.Network(current_model.input_size, current_model.output_size)

        experiences = self.knowledge_manager.get_experiences()
        for i in range(math.floor(len(experiences)/batch_size)):
            """Update online network weights."""
            batch = experiences.sample(batch_size)
            inputs = np.array([b["state"] for b in batch], dtype=np.float64)
            actions = np.array([b["action"] for b in batch], dtype=int)
            rewards = np.array([b["reward"] for b in batch], dtype=np.float64)
            next_inputs = np.array([b["next_state"] for b in batch], dtype=np.float64)

            actions_one_hot = np.eye(current_model.output_size)[actions]
            next_qvalues = np.squeeze(new_model.model(next_inputs))
            targets = rewards + discount * np.amax(next_qvalues, axis=-1)

            new_model.train_step(inputs, targets, actions_one_hot)
        return new_model

    def update_learning_model(self, model,memory):
        self.decision_making_component.push_new_model(model,memory)




