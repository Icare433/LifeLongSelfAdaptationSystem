import math

import numpy as np
from sklearn.cluster import Birch, OPTICS, SpectralClustering
from sklearn.manifold import MDS, LocallyLinearEmbedding
from sklearn.mixture import GaussianMixture

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

    def check_for_tasks(self):
        self.knowledge_manager


class TaskBasedKnowledgeMiner:

    def __init__(self, knowledge_manager):
        self.knowledge_manager = knowledge_manager

    def update_rewards(self):
        goal_model = self.knowledge_manager.get_current_goals()
        for experience in self.knowledge_manager.experiences:
            experience.reward = goal_model.evaluate(experience.next_state)


class KnowledgeManager:

    def __init__(self, learning_model, goal_model, reclustering_interval = 30, reclustering_delay = 20):

        self.experiences = list()
        self.learning_model = learning_model
        self.goal_model = goal_model
        self.last_experiences = dict()
        self.new_data_per_mote =dict()
        self.save_file = open("results.txt", "a")
        self.reclustering_interval = reclustering_interval
        self.reclustering_delay = reclustering_delay
        self.reclusterd = False

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
            self.experiences.append(self.last_experiences.get(mote))
        else:
            self.new_data_per_mote[mote] = -self.reclustering_delay
        self.new_data_per_mote[mote] = self.new_data_per_mote[mote] + 1

        self.last_experiences[mote] = experience

        recluster = False
        for mote in self.new_data_per_mote:
            if self.new_data_per_mote[mote] > self.reclustering_interval:
                recluster = True

            else:
                recluster = False

        if recluster and (self.clusteringManager is not None):
            self.clusteringManager.recluster()
            self.reclusterd = True

    def observe_action(self, mote, action):
        if self.last_experiences.get(mote) is not None:
            self.save_file.write(str(action))
            self.save_file.write("\n")
            self.last_experiences.get(mote).add_action(action)

    def get_experiences(self, size, motes):
        memory = Q_learner.Memory(100000)
        index = len(self.experiences) - 1
        while memory.size() < size & index >= 0:
            experience = self.experiences[index]
            index -= 1
            if experience.mote in motes:
                state_vector = list()
                for transmission in experience.state:
                    for value in list(transmission.values()):
                        state_vector.append(value)
                state_vector = np.array(state_vector, dtype=object)
                next_state_vector = list()
                for transmission in experience.next_state:
                    for value in list(transmission.values()):
                        next_state_vector.append(value)
                next_state_vector = np.array(next_state_vector, dtype=object)
                memory.add({
                    "state": state_vector,
                    "action": experience.action,
                    "reward": experience.reward,
                    "next_state": next_state_vector
                    })
        return memory

    def cluster_data(self):
        data = dict()
        datasize = 0
        for mote in self.new_data_per_mote:
            datasize += self.new_data_per_mote[mote]
        for experience in self.experiences[-datasize:]:
            if data.get(experience.mote) is None:
                data[experience.mote] = list()
            for transmission in experience.state:
                for value in list(transmission.values()):
                    data[experience.mote].append(value)

            data[experience.mote].append(experience.action)

            for transmission in experience.next_state:
                for value in list(transmission.values()):
                    data[experience.mote].append(value)
        data_array = list()
        minimal_data = self.reclustering_interval*self.reclustering_interval*15
        for mote in data:
            if minimal_data > len(data[mote]):
                minimal_data = len(data[mote])

        mote_array = list()
        for mote in data:
            mote_array.append(mote)
            data_array.append(np.array(data[mote][-minimal_data:], dtype=object))
        data_array = np.array(data_array)
        for mote in self.new_data_per_mote:
            self.new_data_per_mote[mote] = 0
        p = np.random.permutation(len(mote_array))
        randomized_mote_array = list()
        randomized_data_array = list()
        for index in p:
            randomized_mote_array.append(mote_array[index])
            randomized_data_array.append(data_array[index])
        return [randomized_mote_array, randomized_data_array]

    def addCLusteringManager(self, clustering_manager):
        self.clusteringManager = clustering_manager


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

    def retrain(self, current_model, experiences, batch_size=32, discount=0.99):
        new_model = Q_learner.Network(current_model.input_size, current_model.output_size)

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

    def update_learning_model(self, clusters, models, memories):
        self.decision_making_component.push_new_model(clusters, models, memories)


class ClusteringManager:

        def __init__(self, knowledgeManager, cluster_detection, LearningManager, time_window = 25000):
            self.knowledgeManager = knowledgeManager
            self.cluster_detection = cluster_detection
            self.time_window = time_window
            self.LearningManager = LearningManager

        def recluster(self):
            print("recluster")
            [mote_array, data_for_clustering] = self.knowledgeManager.cluster_data()
            clusters = self.cluster_detection.determine_clustering(mote_array, data_for_clustering,
                                                                   reducer=LocallyLinearEmbedding(n_components=5, method='modified'), clusterer=SpectralClustering)

            self.knowledgeManager.save_file.write("recluster: "+str(clusters)+"\n")
            print(clusters)
            models = list()
            experiences = list()
            for index in clusters:
                experience = self.knowledgeManager.get_experiences(self.time_window, clusters[index])
                model = self.knowledgeManager.get_current_model()
                models.append(self.LearningManager.retrain(model, experience))
                experiences.append(experience)
            self.LearningManager.update_learning_model(clusters, models, experiences)




