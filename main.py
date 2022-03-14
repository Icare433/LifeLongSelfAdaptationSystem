import ClusteringDetection
import selfAdaptiveSystem
import paho.mqtt.client as mqtt
import queue

from lifeLongSystem import *

q = queue.Queue()


def on_message(client, userdata, message):
    q.put(message)


client = mqtt.Client("life-long-learning-on-dingnet")
client.connect("localhost", port=1883)
client.subscribe("node/#")

client.on_message = on_message  # attach function to callback


def main():
    client.loop_stop()
    client.loop_start()
    latency_goal = selfAdaptiveSystem.LatencyGoal(2)
    packet_loss_goal = selfAdaptiveSystem.PacketlossGoal(0.1)
    energy_goal = selfAdaptiveSystem.EnergyconsumptionGoal()
    goal_list = list()
    goal_list.append(packet_loss_goal)
    goal_list.append(latency_goal)
    goal_model = selfAdaptiveSystem.ListGoalModel(goal_list,[0.8,0.2])
    new_goal_list = list()
    new_goal_list.append(packet_loss_goal)
    new_goal_list.append(latency_goal)
    new_goal_list.append(energy_goal)
    new_goal_model = selfAdaptiveSystem.ListGoalModel(new_goal_list,[0.4,0.4,0.2])

    knowledge = selfAdaptiveSystem.Knowledge(goal_model)
    executor = selfAdaptiveSystem.Executor(client)
    planner = selfAdaptiveSystem.Planner(executor)
    q_learner = selfAdaptiveSystem.Q_learner(planner, knowledge)
    cluster_detection = ClusteringDetection.ClusterDetection()
    knowledge_manager = KnowledgeManager(q_learner.agent.learning_model, goal_model,cluster_detection)
    knowledge.addKnowledgeManager(knowledge_manager)
    analyser = selfAdaptiveSystem.Analyser(knowledge, q_learner)
    monitor = selfAdaptiveSystem.Monitor(knowledge, analyser, q)
    #learning_manager = LearningManager(knowledge_manager, q_learner)
    #task_based_knowledge_miner = TaskBasedKnowledgeMiner(knowledge_manager)
    #task_manager = TaskManager(knowledge, knowledge_manager, learning_manager, task_based_knowledge_miner)
    #change = False
    #changed = False
    while True:

        monitor.monitor()
        #for mote in knowledge.datapoints:
        #    mote_data = knowledge.datapoints.get(mote)
        #    if len(mote_data) > 8000:

        #        change = True

        #if change and not changed:
        #    changed = True
        #    print("changed!")
        #    task_manager.update_goal_model(new_goal_model)

    client.loop_stop()


main()
