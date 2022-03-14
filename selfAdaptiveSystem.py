import paho.mqtt.client as mqtt
import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.5/bin")

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
    data["transmission_interval"] = datadict.get("content").get("payload")[-5]*5
    data["expiration_time"] = datadict.get("content").get("payload")[-4]*5
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


def create_lost_data(number,mote):
        lost_data = dict()
        lost_data["transmission_interval"] = -1000
        lost_data["expiration_time"] = -1000
        lost_data["transmission_power_setting"] = -1000
        lost_data["transmission_number"] = number
        lost_data["latency"] = -1000
        lost_data["transmission_power"] = -2000
        lost_data["departure_time"] = -1000
        lost_data["receiver"] = -1000
        return lost_data

class Knowledge:



    def __init__(self,goal_model,number_of_datapoints_per_state = 5, number_of_features = 8, number_of_actions = 15, knowledgeManager = None):
        self.number_of_datapoints_per_state = number_of_datapoints_per_state
        self.number_of_features = number_of_features
        self.number_of_actions = number_of_actions
        self.goal_model = goal_model
        self.motes = dict()
        self.datapoints = dict()
        self.lastdatapoint = dict()
        self.knowledgeManager = knowledgeManager



    def add_data(self,mote_id,data):
        if mote_id not in self.datapoints:
            self.datapoints[mote_id] = list()
            self.datapoints.get(mote_id).append(data)
            self.lastdatapoint[mote_id] = data;
            return False


        prev_data = self.lastdatapoint.get(mote_id)
        if prev_data.get("transmission_number") == data.get("transmission_number"):
            if prev_data.get("transmission_power") < data.get("transmission_power"):
                self.datapoints.get(mote_id)[-1] = data
                self.lastdatapoint[mote_id] = data
                return False

        elif (prev_data.get("transmission_number") + 1) %100 == data.get("transmission_number"):

            self.datapoints.get(mote_id).append(data)
            self.lastdatapoint[mote_id] = data
            return True

        elif data.get("transmission_number") > prev_data.get("transmission_number"):
            for number in range(prev_data.get("transmission_number"),data.get("transmission_number")):
                self.datapoints.get(mote_id).append(create_lost_data(number,mote_id))
            self.datapoints.get(mote_id).append(data)
            self.lastdatapoint[mote_id] = data
            return True


        elif prev_data.get("transmission_number") - data.get("transmission_number")-10 > 0:
            datapoint_complete = True
            for number in range(prev_data.get("transmission_number"),100):
                self.datapoints.get(mote_id).append(create_lost_data(number,mote_id))

            for number in range(0,data.get("transmission_number")):
                self.datapoints.get(mote_id).append(create_lost_data(number,mote_id))

            self.datapoints.get(mote_id).append(data)
            self.lastdatapoint[mote_id] = data
            return True


    def get_last_state_vector(self,mote_id):
        return list(self.datapoints.get(mote_id))[-self.number_of_datapoints_per_state:]

    def getKnowledgeManager(self):
        return self.knowledgeManager

    def addKnowledgeManager(self,knowledgeManager):
        self.knowledgeManager =knowledgeManager

class Analyser:

    def __init__(self,knowledge, decision_making):
        self.knowledge = knowledge
        self.decision_making = decision_making
        self.new_datapoint_counter = dict()

    def analyse_new_datapoint(self,mote):
        #If we get first datapoint for a mote, prepare a counter
        if self.new_datapoint_counter.get(mote) is None:
            self.new_datapoint_counter[mote] = 0
        # A new datapoint for a mote is available
        self.new_datapoint_counter[mote] = self.new_datapoint_counter.get(mote) + 1
        # If enough points are available to create
        if self.new_datapoint_counter.get(mote) > self.knowledge.number_of_datapoints_per_state:
            self.new_datapoint_counter[mote] = 0

            state = self.knowledge.get_last_state_vector(mote)
            state_vector = list()
            for transmission in state:
                for value in list(transmission.values()):
                    state_vector.append(value)
            state_vector = np.array(state_vector,dtype=np.float64)
            reward = self.knowledge.goal_model.evaluate(state)
            if self.knowledge.getKnowledgeManager() is not None:
                self.knowledge.getKnowledgeManager().observe_state(mote,state,reward)

            self.decision_making.determine_action(mote,state_vector,reward)



class DecisionMaking:

    def __init__(self, planner, knowledge, nb_agents):
        self.clustering = dict()
        self.planner = planner
        self.knowledge = knowledge
        self.agents = list()
        for n in range(nb_agents):
            self.agents.append(Q_learner_model.Agent(self.knowledge.number_of_datapoints_per_state * self.knowledge.number_of_features, self.knowledge.number_of_actions))
            self.agents[n].handle_episode_start()


    def determine_action(self,mote,state,objective_function):
        observation = Observation(state,objective_function)
        cluster = cluster_of_mote(mote)
        action = self.agents[cluster].step(mote,observation)
        self.planner.plan(mote, action)

    def push_new_model(self,model,memory):
        self.agent.learning_model = model
        self.agent.memory = memory
        self.agent.steps =(1.0/self.agent.anneal_rate)/2

    def cluster_of_mote(self, mote):
        return self.clustering.get(mote,0)

    def switch_mote_cluster(self,mote,cluster):
        self.clustering[mote] = cluster




class Monitor:
    def __init__(self,knowledge,analyser, queue):
        self.knowledge = knowledge
        self.analyser = analyser
        self.queue = queue

    def monitor(self):
            try:
                message = self.queue.get_nowait()
            except queue.Empty:
                return
            should_analyse = False
            if message is None:
                return

            try:
                mote = int(message.topic.split("/")[1])
                should_analyse = self.knowledge.add_data(mote, get_data(str(message.payload.decode("utf-8"))))

            except AttributeError:
                return

            except ValueError:
                return
            if should_analyse:
                self.analyser.analyse_new_datapoint(mote)

class Planner:

    actionTable = {0:[0,0,0],1:[1,0,0],2:[0,5,0],3:[0,0,5],4:[1,5,0],5:[1,0,5],6:[0,5,5],7:[1,5,5],
                   8:[-1,-5,-5],9:[-1,0,0],10:[0,-5,0],11:[0,0,-5],12:[-1,-5,0],13:[-1,0,-5],14:[0,-5,-5]}

    def __init__(self,executer):
        self.executer = executer

    def plan(self,mote,action_number):
        if action_number != 0:
            action = self.actionTable.get(action_number)
            self.executer.execute_change(mote,action[0],action[1],action[2])

class Executor:

    def __init__(self,mqtt_client):
        self.mqtt_client = mqtt_client

    def execute_change(self,moteid,transmission_power,expiration_time,transmission_interval):
        self.mqtt_client.publish("node/"+str(moteid)+"/application/1/tx",'{"data":['+str(transmission_power)+','+str(expiration_time)+','+str(transmission_interval)+'],"macCommands":[]}')


class LatencyGoal:

    def __init__(self,value):
        self.value = value

    def evaluate(self,state):
        satisfaction = 0
        counter= 0
        for transmission in state:
            if transmission.get("expiration_time") != -1000:
                satisfaction+= max(0,(transmission.get("latency")*transmission.get("expiration_time")/100 - self.value))/(transmission.get("latency")*transmission.get("expiration_time")/100+1)
                counter += 1
        satisfaction = satisfaction / counter
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
        return reward / len(self.goals)




