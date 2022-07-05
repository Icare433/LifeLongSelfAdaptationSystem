
#import os
#os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.5/bin")

import time
import queue
import json
import numpy as np
import Q_learner as Q_learner_model
import tensorflow as tf
import math

import logging

logging.basicConfig(filename='shapes.log', level=logging.DEBUG)

q=queue.Queue()
#later in code

#number of episode we will run
n_episodes = 10000

#maximum of iteration per episode
max_iter_episode = 100

#initialize the exploration probability to 1
exploration_proba = 1

#exploartion decreasing decay for exponential decreasing
exploration_decreasing_decay = 0.001

# minimum of exploration proba
min_exploration_proba = 0.01

#discounted factor
gamma = 0.99

#learning rate
lr = 0.1



def get_data(message):
    datadict = json.loads(message).get("transmission")
    data = dict()
    if (datadict.get("content").get("payload")[-5]) < 0:
        data["transmission_interval"] = (256 + (datadict.get("content").get("payload")[-5]))*5
    else:
        data["transmission_interval"] = (datadict.get("content").get("payload")[-5])*5

    if datadict.get("content").get("payload")[-4] < 0:
        data["expiration_time"] = (256 + datadict.get("content").get("payload")[-4]) * 5
    else:
        data["expiration_time"] = datadict.get("content").get("payload")[-4] * 5

    data["transmission_power_setting"] = datadict.get("content").get("payload")[-3]
    data["transmission_number"] = datadict.get("content").get("payload")[-2]
    data["latency"] = datadict.get("content").get("payload")[-1]
    data["transmission_power"] = datadict.get("transmissionPower")
    for feature in data:
        if data[feature] < 0:
            data[feature] = data[feature] + 255
    time = datadict.get("departureTime")
    time = time.get("time")
    departure_time = time.get("hour") * 3600
    departure_time += time.get("minute") * 60
    departure_time += time.get("second")
    data["departure_time"] = departure_time
    data["receiver"] = datadict.get("receiver")
    return data


def create_lost_data(number,transmission_interval,departure_time):
        lost_data = dict()
        lost_data["transmission_interval"] = transmission_interval
        lost_data["expiration_time"] = -1000
        lost_data["transmission_power_setting"] = -1000
        lost_data["transmission_number"] = number
        lost_data["latency"] = -1000
        lost_data["transmission_power"] = -2000
        lost_data["departure_time"] = departure_time
        lost_data["receiver"] = -1000
        return lost_data


class Knowledge:

    def __init__(self,goal_model,number_of_datapoints_per_state = 10, number_of_features = 8, number_of_actions = 15, knowledgeManager = None):
        self.number_of_datapoints_per_state = number_of_datapoints_per_state
        self.number_of_features = number_of_features
        self.number_of_actions = number_of_actions
        self.goal_model = goal_model
        self.motes = dict()
        self.datapoints = dict()
        self.lastdatapoint = dict()
        self.knowledgeManager = knowledgeManager
        self.expected_next_transmission = dict()
        self.maximum_time_between_transmissions = 300

    def add_data(self, mote_id, data):
        motes_for_analysis = []
        self.expected_next_transmission[mote_id] = (data.get("departure_time") + self.maximum_time_between_transmissions)%86400

        for mote in self.expected_next_transmission:
            if self.expected_next_transmission[mote] < int(data.get("departure_time")) and int(data.get("departure_time")) - self.expected_next_transmission[mote] < 1000:
                number = (self.lastdatapoint.get(mote).get("transmission_number") + 1) % 100
                lost_data = create_lost_data(number, self.lastdatapoint[mote].get("transmission_interval"),data.get("departure_time"))
                self.datapoints.get(mote).append(lost_data)
                self.expected_next_transmission[mote] = (int(data.get("departure_time")) + self.maximum_time_between_transmissions)%86400
                self.lastdatapoint[mote] = lost_data
                motes_for_analysis.append(mote)
        if mote_id not in self.datapoints:
            self.datapoints[mote_id] = list()
            self.datapoints.get(mote_id).append(data)
            self.lastdatapoint[mote_id] = data
            return motes_for_analysis
        prev_data = self.lastdatapoint.get(mote_id)
        if prev_data.get("transmission_number") == data.get("transmission_number"):
            if prev_data.get("transmission_power") < data.get("transmission_power"):
                self.datapoints.get(mote_id)[-1] = data
                self.lastdatapoint[mote_id] = data
                return motes_for_analysis

        elif (prev_data.get("transmission_number") + 1) % 100 == data.get("transmission_number"):
            self.datapoints.get(mote_id).append(data)
            self.lastdatapoint[mote_id] = data
            motes_for_analysis.append(mote_id)
            return motes_for_analysis

        elif data.get("transmission_number") > prev_data.get("transmission_number"):
            for number in range(prev_data.get("transmission_number"), data.get("transmission_number")):
                self.datapoints.get(mote_id).append(create_lost_data(number, prev_data.get("transmission_interval"),data.get("departure_time")))
            self.datapoints.get(mote_id).append(data)
            self.lastdatapoint[mote_id] = data
            motes_for_analysis.append(mote_id)
            return motes_for_analysis

        elif prev_data.get("transmission_number") - data.get("transmission_number")-10 > 0:

            for number in range(prev_data.get("transmission_number"), 100):
                self.datapoints.get(mote_id).append(create_lost_data(number, prev_data.get("transmission_interval"),data.get("departure_time")))

            for number in range(0, data.get("transmission_number")):
                self.datapoints.get(mote_id).append(create_lost_data(number, prev_data.get("transmission_interval"),data.get("departure_time")))

            self.datapoints.get(mote_id).append(data)
            self.lastdatapoint[mote_id] = data
            motes_for_analysis.append(mote_id)
            return motes_for_analysis

        return motes_for_analysis

    def get_last_state_vector(self,mote_id):
        return list(self.datapoints.get(mote_id))[-self.number_of_datapoints_per_state:]

    def getKnowledgeManager(self):
        return self.knowledgeManager

    def addKnowledgeManager(self, knowledgeManager):
        self.knowledgeManager = knowledgeManager

class Analyser:

    def __init__(self, knowledge, decision_making):
        self.gateway = None
        self.knowledge = knowledge
        self.decision_making = decision_making
        self.new_datapoint_counter = dict()

    def analyse_new_datapoint(self, mote):
        if self.new_datapoint_counter.get(mote) is None:
            self.new_datapoint_counter[mote] = 0
        # A new datapoint for a mote is available
        self.new_datapoint_counter[mote] = self.new_datapoint_counter[mote] + 1
        # If enough points are available to create
        if self.new_datapoint_counter[mote] > self.knowledge.number_of_datapoints_per_state:
            self.new_datapoint_counter[mote] = 0

            state = self.knowledge.get_last_state_vector(mote)
            state_vector = list()
            for transmission in state:
                if transmission.get("transmission_power_setting") != -1000:
                    self.gateway = transmission.get("receiver")
                for value in list(transmission.values()):
                    state_vector.append(value)
            state_vector = np.array(state_vector, dtype=np.float64)
            reward = self.knowledge.goal_model.evaluate(state)
            if self.knowledge.getKnowledgeManager() is not None:
                self.knowledge.getKnowledgeManager().observe_state(mote,state,reward)

            self.decision_making.determine_action(mote, state_vector, reward)



class DecisionMaking:

    def __init__(self, planner, knowledge, nb_agents):
        self.clustering = dict()
        self.planner = planner
        self.knowledge = knowledge
        self.agents = list()
        for n in range(nb_agents):
            self.agents.append(Q_learner_model.Agent(self.knowledge.number_of_datapoints_per_state * self.knowledge.number_of_features, self.knowledge.number_of_actions))
            self.agents[n].handle_episode_start()


    def determine_action(self, mote, state, objective_function):
        observation = Observation(state, objective_function)
        cluster = self.cluster_of_mote(mote)
        action = self.agents[cluster].step(mote, observation)
        if self.knowledge.getKnowledgeManager() is not None:
            self.knowledge.getKnowledgeManager().observe_action(mote, action)
        self.planner.plan(mote, action)


    def push_new_model(self, clusters, models, memories):
        self.agents = list()
        index = 0
        self.clustering = dict()
        while index < len(models):
            self.agents.append(Q_learner_model.Agent(self.knowledge.number_of_datapoints_per_state * self.knowledge.number_of_features, self.knowledge.number_of_actions))
            self.agents[len(self.agents)-1].learning_model = models[index]
            self.agents[len(self.agents) - 1].memory = memories[index]
            self.agents[len(self.agents)-1].handle_episode_start()

            for mote in clusters[index]:
                self.clustering[mote] = index
            index += 1
        print(self.clustering)



    def cluster_of_mote(self, mote):
        return self.clustering.get(mote, 0)





class Monitor:
    def __init__(self,knowledge,analyser, queue):
        self.knowledge = knowledge
        self.analyser = analyser
        self.queue = queue
        self.timefile= open("times.txt","a")

    def monitor(self):

            literal_message = self.queue.recv(4096).decode('utf-8')

            if literal_message is None:
                return
            try:
                message_topic = literal_message.split("|")[0]

                message_payload = literal_message.split("|")[1]
            except IndexError:
                return

            should_analyse = []
            try:
                mote = int(message_topic.split("/")[1])
                should_analyse = self.knowledge.add_data(mote, get_data(str(message_payload)))
            except AttributeError:
                return

            except ValueError:
                return
            except IndexError:
                return

            if len(should_analyse) > 0:
                for mote_id in should_analyse:
                    self.analyser.analyse_new_datapoint(mote_id)


class Planner:

    actionTable = {0:[0,0,0],1:[1,0,0],2:[0,5,0],3:[0,0,5],4:[1,5,0],5:[1,0,5],6:[0,5,5],7:[1,5,5],
                   8:[-1,-5,-5],9:[-1,0,0],10:[0,-5,0],11:[0,0,-5],12:[-1,-5,0],13:[-1,0,-5],14:[0,-5,-5]}

    def __init__(self, executer):
        self.executer = executer

    def plan(self, mote, action_number):
        if action_number != 0:
            action = self.actionTable.get(action_number)
            self.executer.execute_change(mote, action[0], action[1], action[2])

class Executor:

    def __init__(self,mqtt_client):
        self.mqtt_client = mqtt_client

    def execute_change(self,moteid,transmission_power,expiration_time,transmission_interval):
        print("made a choice " + str(moteid))
        self.mqtt_client.send(("node/"+str(moteid)+"/application/1/tx"+"|"+'{"data":['+str(transmission_power)+','+str(expiration_time)+','+str(transmission_interval)+'],"macCommands":[]}'+"\n").encode('utf-8'))



class LatencyGoal:

    def __init__(self, value):
        self.value = value

    def evaluate(self, state):
        satisfaction = 0
        counter= 0
        for transmission in state:
            if transmission.get("expiration_time") != -1000:
                satisfaction += 1 - max(0, (transmission.get("latency")*transmission.get("expiration_time")/100-self.value))\
                               / (transmission.get("latency")*transmission.get("expiration_time")/100+1)
                counter += 1
        if counter > 0:
            satisfaction = satisfaction / counter
        else:
            satisfaction = 0
        return satisfaction


class AvailabiltyGoal:

    def __init__(self, value):
        self.value = value

    def evaluate(self, state):
        satisfaction = 0
        counter= 0
        last_transmitted = None
        for transmission in state:
            if transmission["transmission_power_setting"] != -1000:
                if last_transmitted is not None and transmission.get("departure_time") - last_transmitted > 0 and transmission.get("departure_time") - last_transmitted < 400:
                    satisfaction += 1 - max(0, (transmission.get("departure_time") - last_transmitted -self.value))\
                               / (transmission.get("departure_time") - last_transmitted + 1)
                    counter += 1
                last_transmitted = transmission.get("departure_time")

        if counter > 0:
            satisfaction = satisfaction / counter
        else:
            satisfaction = 0
        return satisfaction


class PacketlossGoal:

    def __init__(self,value):
        self.value = value

    def evaluate(self,state):
        satisfaction = 0.0
        for transmission in state:
            if transmission.get("expiration_time") == -1000:
                satisfaction+= 1.0
        satisfaction = satisfaction / len(state)
        satisfaction = min(max(1.0 - (satisfaction-self.value)*4,0),1.0)

        return satisfaction

class EnergyconsumptionGoal:

    def evaluate(self,state):
        satisfaction = 0.0
        for transmission in state:
            if transmission.get("expiration_time") != -1000:
                satisfaction = 1 - math.pow(transmission.get("transmission_power_setting")/14.0,2)
        satisfaction = max(satisfaction,0.0)
        return satisfaction

class Observation:

    def __init__(self,state,reward):
        self.state = state
        self.reward = reward

class ListGoalModel:

    def __init__(self,goals,weights):
        self.goals = goals
        self.weights = weights

    def evaluate(self,state):
        reward = 0.0
        for goal in range(len(self.goals)):
            reward +=self.goals[goal].evaluate(state)*self.weights[goal]
        return reward




