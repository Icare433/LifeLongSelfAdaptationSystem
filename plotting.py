import json
import math

import plotly.graph_objects as go
import numpy as np
from scipy import stats
from plotly.subplots import make_subplots
import plotly.express as px
import statistics
import datetime

from scipy.stats import linregress


def readData(resultsfileName):
    data = {"cycle": [dict()], "packet_loss": [dict()], "energy_consumption": [dict()],
            "freshness": [dict()], "utility": [dict()], "cluster": [dict()], "transmission_interval":[dict()],"action":[dict()], "received_transmission_interval":[dict()]}
    results_file = open(resultsfileName, "r")
    clustering = dict()
    last_performed_action = dict()
    last_mote = 0
    cycle = [dict()]
    cluster = 0
    counter = dict()
    cluster_time_interval = 10000
    next_cluster_time = cluster_time_interval
    changes = {"to_0":list(), "to_1":list()}
    for line in results_file.readlines():
        if line == "changed\n":
            cluster = 1
            counter = dict()
        elif line[:len("recluster: ")] == "recluster: ":
            #cluster += 1
            #cycle.append(dict())
            #for feature in data:
            #    data[feature].append(dict())

            to_search = line[len("recluster: "):-1]
            correcting_index = to_search.find(": [")
            dictstring = ""
            while correcting_index > -1:
                if to_search[correcting_index - 2] == " " or to_search[correcting_index - 2] == "{":
                    dictstring += to_search[:correcting_index - 1]
                    dictstring += '"'
                    dictstring += to_search[correcting_index - 1]
                    dictstring += '"'
                else:
                    dictstring += to_search[:correcting_index - 2]
                    dictstring += '"'
                    dictstring += to_search[correcting_index - 2]
                    dictstring += to_search[correcting_index - 1]
                    dictstring += '"'
                dictstring += to_search[correcting_index:correcting_index + 1]
                to_search = to_search[correcting_index + 1:]
                correcting_index = to_search.find(": [")
            dictstring += to_search
            cluster_data = json.loads(dictstring)
            set_size = {"same": list(), "change" : list()}
            for cluster_motes in cluster_data:
                for mote_to_cluster in cluster_data[cluster_motes]:
                    if clustering.get(mote_to_cluster) is None or clustering[mote_to_cluster] == int(cluster_motes):
                        set_size["same"].append(mote_to_cluster)
                    else:
                        set_size["change"].append(mote_to_cluster)
                    clustering[mote_to_cluster] = int(cluster_motes)


            if len(set_size["same"]) < len(set_size["change"]):
                for mote_to_cluster in clustering:
                    clustering[mote_to_cluster] = (clustering[mote_to_cluster]+1)%2
                for changed_mote in set_size["same"]:
                    if data["packet_loss"][cluster - 1].get(str(changed_mote)) is not None and data["packet_loss"][cluster].get(str(changed_mote)) is not None:
                        if clustering[changed_mote] == 1:
                            changes["to_1"].append(np.mean(data["packet_loss"][cluster].get(str(changed_mote))))
                        else:
                            changes["to_0"].append(
                                np.mean(data["packet_loss"][cluster].get(str(changed_mote))))


            else:
                for changed_mote in set_size["change"]:
                    if data["packet_loss"][cluster - 1].get(str(changed_mote)) is not None and data["packet_loss"][cluster].get(str(changed_mote)) is not None:
                        if clustering[changed_mote] == 1:
                            changes["to_1"].append(
                                np.mean(data["packet_loss"][cluster].get(
                                str(changed_mote))))
                        else:
                            changes["to_0"].append(
                                np.mean(data["packet_loss"][cluster].get(
                                    str(changed_mote))))
            for i in range(0):
                cluster += 1
                cycle.append(dict())
                for feature in data:
                    data[feature].append(dict())

        elif all(c.isdigit() for c in line[:-2]):
            last_performed_action[last_mote] = int(line)

        else:
            read_data = json.loads(line)
            motenumber = list(read_data.keys())[0]
            mote = motenumber
            last_mote = motenumber
            if counter.get(mote) is None:
                counter[mote] = 0

            counter[mote] = counter[mote] + 1

            last_arrived_signal = 0
            for transmission in read_data.get(motenumber):
                if transmission["departure_time"] > 0:
                    last_arrived_signal = transmission["departure_time"]


            if last_arrived_signal < next_cluster_time - cluster_time_interval:
                next_cluster_time = cluster_time_interval

            if last_arrived_signal > next_cluster_time:
                next_cluster_time += cluster_time_interval
                cluster += 1
                cycle.append(dict())
                for feature in data:
                    data[feature].append(dict())

            if counter.get(mote) > 0:

                features = {"packet_loss": 0, "energy_consumption": 0, "freshness": 0, "utility": [0, 0, 0, 0], "cluster": clustering.get(int(mote),0), "transmission_interval": 0, "received_transmission_interval":0}
                if cycle[cluster].get(mote) is None:
                    data["cycle"][cluster][mote] = list()
                    data["action"][cluster][mote] = list()
                    for feature in features:
                        data[feature][cluster][mote] = list()
                    cycle[cluster][mote] = 0

                last_transmitted = None
                for transmission in read_data.get(motenumber):

                    if transmission["transmission_power_setting"] != -1000:
                        if last_transmitted is not None and transmission.get("departure_time") - last_transmitted > 0 and transmission.get("departure_time") - last_transmitted < 400:
                            features["received_transmission_interval"] = transmission.get("departure_time") - last_transmitted
                            features["utility"][0] += 1 - max(0, (
                                        transmission.get("departure_time") - last_transmitted - 180)) \
                                                     / (transmission.get("departure_time") - last_transmitted + 1)
                        features["transmission_interval"] = transmission.get("transmission_interval")
                        last_transmitted = transmission.get("departure_time")



                        features["energy_consumption"] = math.pow(10, (
                                transmission["transmission_power_setting"] - 30) / 10)
                        if transmission.get("expiration_time") < 0:
                            transmission["expiration_time"] = (256+ transmission["expiration_time"]/5)*5
                        features["freshness"] = features.get("freshness") + transmission["latency"] * transmission.get(
                            "expiration_time") / 100
                        features["utility"][1] = features["utility"][1] + 1 - max(0, (
                                transmission.get("latency") * transmission.get("expiration_time") / 100 - 120)) / (
                                                         transmission.get("latency") * transmission.get(
                                                     "expiration_time") / 100 + 1)
                        features["utility"][3] += 1 - math.pow(transmission.get("transmission_power_setting")/14.0,2)


                    else:
                        features["packet_loss"] = features.get("packet_loss") + 1
                        last_transmitted = None
                if(20 - features["packet_loss"] > 0):
                    features["freshness"] = features.get("freshness") / (20 - features["packet_loss"])
                    features["utility"][0] = features["utility"][0] / (20 - features["packet_loss"])
                    features["utility"][1] = features["utility"][1] / (20 - features["packet_loss"])
                    features["utility"][3] = features["utility"][3] / (20 - features["packet_loss"])
                    features["utility"][2] = min(max(1.0 - (features.get("packet_loss") / 20 - 0.1) * 4, 0), 1.0)
                else:
                    features["freshness"] = 20*500
                    features["utility"][0] = 0
                    features["utility"][1] = 0
                    features["utility"][3] = 0
                    features["utility"][2] = 0


                if False:  # cycle[cluster][mote] > 0:
                    features["packet_loss"] = data["packet_loss"][cluster][mote][-1] * cycle[cluster][mote] / (
                            cycle[cluster][mote] + 1) + features.get("packet_loss") / (
                                                      10 * (cycle[cluster][mote] + 1))
                else:
                    features["packet_loss"] = features.get("packet_loss") / 20

                if True:  # cluster == 1:
                    features["utility"] = features["utility"][0] * 0.25 + features["utility"][1] * 0.25 + features["utility"][2] * 0.25 + features["utility"][3] * 0.25
                else:
                    features["utility"] = sum(features["utility"]) / 3

                for feature in features:
                    data[feature][cluster][mote].append(features[feature])
                data["action"][cluster][mote].append(last_performed_action.get(mote, 0))
                data["cycle"][cluster][mote].append(cycle[cluster][mote])
                cycle[cluster][mote] = cycle[cluster][mote] + 1
    for cluster_change in changes:
        if len(changes.get(cluster_change)) > 0:
            fig = px.bar(x=range(len(changes.get(cluster_change))), y=changes.get(cluster_change))
            #fig.show()
            print(cluster_change +" "+str(np.mean(changes.get(cluster_change))))
    return data


def readDataJava(resultsfileName):
    data = {"cycle": [dict()], "packet_loss": [dict()], "energy_consumption": [dict()],
            "freshness": [dict()], "utility": [dict()], "cluster": [dict()], "transmission_interval":[dict()], "received_transmission_interval":[dict()]}
    results_file = open(resultsfileName, "r")
    cycle = [dict()]
    cluster = 0
    counter = dict()
    cluster_time_interval = 1000
    next_cluster_time = cluster_time_interval
    last_transmitted = None
    for line in results_file.readlines():

        read_data = json.loads(line)
        motenumber = list(read_data.keys())[0]
        mote = motenumber
        if counter.get(mote) is None:
            counter[mote] = 0

        counter[mote] = counter[mote] + 1

        last_arrived_signal = 0
        for transmission in read_data.get(motenumber):
            try:
                departure_time = datetime.datetime.strptime(transmission["departure_time"], "%Y-%m-%dT%H:%M:%S")
                departure_time_seconds = (departure_time.hour * 60 + departure_time.minute) * 60 + departure_time.second
            except:
                departure_time = datetime.datetime.strptime(transmission["departure_time"], "%Y-%m-%dT%H:%M")
                departure_time_seconds = (departure_time.hour * 60 + departure_time.minute) * 60
            if departure_time_seconds > 0:
                last_arrived_signal = departure_time_seconds

        if last_arrived_signal > next_cluster_time:
            next_cluster_time += cluster_time_interval
            for feature in data:
                for mote_to_check in data[feature][cluster]:
                    if feature == "utility":
                        state_length = len(data[feature][cluster][mote_to_check])
                        data[feature][cluster][mote_to_check] = np.sum(data[feature][cluster][mote_to_check],axis=0)
                        if(state_length > data[feature][cluster][mote_to_check][2]):
                            data[feature][cluster][mote_to_check][0] = data[feature][cluster][mote_to_check][0]/ (
                                    state_length-data[feature][cluster][mote_to_check][2])
                            data[feature][cluster][mote_to_check][1] = data[feature][cluster][mote_to_check][1] / (
                                    state_length - data[feature][cluster][mote_to_check][2])
                            data[feature][cluster][mote_to_check][3] = data[feature][cluster][mote_to_check][3] / (
                                    state_length - data[feature][cluster][mote_to_check][2])
                            data[feature][cluster][mote_to_check][2] = data[feature][cluster][mote_to_check][2] / state_length
                        else:
                            data[feature][cluster][mote_to_check][0] = 0
                            data[feature][cluster][mote_to_check][1] = 0
                            data[feature][cluster][mote_to_check][3] = 0
                            data[feature][cluster][mote_to_check][2] = 1.0
                        data[feature][cluster][mote_to_check][2] = min(max(1.0 - (data[feature][cluster][mote_to_check][2] - 0.05) * 4, 0), 1.0)
                        data[feature][cluster][mote_to_check] = [np.mean(data[feature][cluster][mote_to_check])]
                    else:
                        data[feature][cluster][mote_to_check] = [np.mean(data[feature][cluster][mote_to_check])]

            cluster += 1
            cycle.append(dict())
            for feature in data:
                data[feature].append(dict())
        if last_arrived_signal < next_cluster_time-cluster_time_interval:
            next_cluster_time = 0

        if counter.get(mote) > 0:

            features = {"packet_loss": 0, "energy_consumption": 0, "freshness": 0, "utility": [0, 0, 0, 0], "transmission_interval": 0, "received_transmission_interval":0}
            if cycle[cluster].get(mote) is None:
                data["cycle"][cluster][mote] = list()
                for feature in features:
                    data[feature][cluster][mote] = list()
                cycle[cluster][mote] = 0


            for transmission in read_data.get(motenumber):

                try:
                    departure_time = datetime.datetime.strptime(transmission["departure_time"], "%Y-%m-%dT%H:%M:%S")
                    departure_time_seconds = (departure_time.hour * 60 + departure_time.minute) * 60 + departure_time.second
                except:
                    departure_time = datetime.datetime.strptime(transmission["departure_time"], "%Y-%m-%dT%H:%M")
                    departure_time_seconds = (departure_time.hour * 60 + departure_time.minute) * 60

                if bool(transmission["collided"]) == False:
                    if last_transmitted is not None and departure_time_seconds - last_transmitted > 0 and departure_time_seconds - last_transmitted < 400:
                        features["received_transmission_interval"] = departure_time_seconds - last_transmitted
                        features["utility"][0] += 1 - max(0, (
                                        departure_time_seconds - last_transmitted - 100)) \
                                                     / (departure_time_seconds - last_transmitted + 1)
                    features["transmission_interval"] = transmission.get("transmission_interval")
                    last_transmitted = departure_time_seconds

                    features["energy_consumption"] = math.pow(10, (
                                int(transmission["transmission_power_setting"]) - 30) / 10)

                    transmission["expiration_time"] = transmission["expiration_time"]

                    features["freshness"] = features.get("freshness") + transmission["latency"] * transmission.get(
                            "expiration_time") / 100
                    features["utility"][1] = features["utility"][1] + 1 - max(0, (
                                transmission.get("latency") * transmission.get("expiration_time") / 100 - 60)) / (
                                                         transmission.get("latency") * transmission.get(
                                                     "expiration_time") / 100 + 1)
                    features["utility"][3] += 1 - math.pow(transmission.get("transmission_power_setting")/14.0,2)


                else:
                    features["packet_loss"] = features.get("packet_loss") + 1
                    features["utility"][2] += 1


            for feature in features:
                data[feature][cluster][mote].append(features[feature])
            data["cycle"][cluster][mote].append(cycle[cluster][mote])
            cycle[cluster][mote] = cycle[cluster][mote] + 1

    for feature in data:
        for mote_to_check in data[feature][cluster]:
            if feature == "utility":
                state_length = len(data[feature][cluster][mote_to_check])
                data[feature][cluster][mote_to_check] = np.sum(data[feature][cluster][mote_to_check], axis=0)
                if (state_length > data[feature][cluster][mote_to_check][2]):
                    data[feature][cluster][mote_to_check][0] = data[feature][cluster][mote_to_check][0] / (
                            state_length - data[feature][cluster][mote_to_check][2])
                    data[feature][cluster][mote_to_check][1] = data[feature][cluster][mote_to_check][1] / (
                            state_length - data[feature][cluster][mote_to_check][2])
                    data[feature][cluster][mote_to_check][3] = data[feature][cluster][mote_to_check][3] / (
                            state_length - data[feature][cluster][mote_to_check][2])
                    data[feature][cluster][mote_to_check][2] = data[feature][cluster][mote_to_check][2] / state_length
                else:
                    data[feature][cluster][mote_to_check][0] = 0
                    data[feature][cluster][mote_to_check][1] = 0
                    data[feature][cluster][mote_to_check][3] = 0
                    data[feature][cluster][mote_to_check][2] = 1.0
                data[feature][cluster][mote_to_check][2] = min(
                    max(1.0 - (data[feature][cluster][mote_to_check][2] - 0.05) * 4, 0), 1.0)
                data[feature][cluster][mote_to_check] = [np.mean(data[feature][cluster][mote_to_check])]
            else:
                data[feature][cluster][mote_to_check] = [np.mean(data[feature][cluster][mote_to_check])]
    return data

def readDataMetricsFile(resultsfileName):
    data = {"cycle": [dict()], "packet_loss": [dict()], "energy_consumption": [dict()],
            "freshness": [dict()], "utility": [dict()], "cluster": [dict()], "transmission_interval":[dict()],"action":[dict()], "received_transmission_interval":[dict()]}
    results_file = open(resultsfileName, "r")
    clustering = dict()
    cycle = [dict()]
    cluster = 0
    counter = dict()
    cluster_time_interval = 3000
    next_cluster_time = cluster_time_interval
    for line in results_file.readlines():

        read_data = json.loads(line)
        motenumber = list(read_data.keys())[0]
        mote = motenumber
        if counter.get(mote) is None:
            counter[mote] = 0

        counter[mote] = counter[mote] + 1

        last_arrived_signal = 0
        data_for_mote = read_data.get(motenumber)
        if data_for_mote["time"] > 0:
            last_arrived_signal = data_for_mote["time"]

        if last_arrived_signal < next_cluster_time - cluster_time_interval:
            next_cluster_time = cluster_time_interval

        if last_arrived_signal > next_cluster_time:
            next_cluster_time += cluster_time_interval
            cluster += 1
            cycle.append(dict())
            for feature in data:
                data[feature].append(dict())

        if counter.get(mote) > 0:

            features = {"packet_loss": 0, "energy_consumption": 0, "freshness": 0, "utility": [0, 0, 0, 0], "cluster": clustering.get(int(mote),0), "transmission_interval": 0, "received_transmission_interval":0}
            if cycle[cluster].get(mote) is None:
                data["cycle"][cluster][mote] = list()
                for feature in features:
                    data[feature][cluster][mote] = list()
                cycle[cluster][mote] = 0

            data_from_mote =  read_data.get(motenumber)

            features["utility"] = data_from_mote["utility"]

            for feature in features:
                data[feature][cluster][mote].append(features[feature])

        data["cycle"][cluster][mote].append(cycle[cluster][mote])
        cycle[cluster][mote] = cycle[cluster][mote] + 1

    for feature in data:
        for period in range(len(data[feature])):
            for mote in counter:
                if mote not in data[feature][period].keys():
                    if period > 0:
                        data[feature][period][mote] = data[feature][period-1][mote]
                    else:
                        data[feature][period][mote] = [0]

    return data


def readDataPowerSetting(resultsfileName):
    data = {"cluster": [], "mote": [], "cycle": [], "transmission_interval": [], "power_setting": []}
    results_file = open(resultsfileName, "r")
    clustering = dict()
    cycle = [dict()]
    cluster = 0
    counter = dict()
    cluster_time_interval = 10000
    next_cluster_time = cluster_time_interval
    for line in results_file.readlines():
        if line == "changed\n":
            cluster = 1
            counter = dict()
        elif line[:len("recluster: ")] == "recluster: ":
            #cluster += 1
            #cycle.append(dict())
            #for feature in data:
            #    data[feature].append(dict())

            to_search = line[len("recluster: "):-1]
            correcting_index = to_search.find(": [")
            dictstring = ""
            while correcting_index > -1:
                if to_search[correcting_index - 2] == " " or to_search[correcting_index - 2] == "{":
                    dictstring += to_search[:correcting_index - 1]
                    dictstring += '"'
                    dictstring += to_search[correcting_index - 1]
                    dictstring += '"'
                else:
                    dictstring += to_search[:correcting_index - 2]
                    dictstring += '"'
                    dictstring += to_search[correcting_index - 2]
                    dictstring += to_search[correcting_index - 1]
                    dictstring += '"'
                dictstring += to_search[correcting_index:correcting_index + 1]
                to_search = to_search[correcting_index + 1:]
                correcting_index = to_search.find(": [")
            dictstring += to_search
            cluster_data = json.loads(dictstring)
            for cluster_motes in cluster_data:
                for mote_to_cluster in cluster_data[cluster_motes]:
                    clustering[mote_to_cluster] = int(cluster_motes)

        else:
            read_data = json.loads(line)
            motenumber = list(read_data.keys())[0]
            mote = motenumber
            if counter.get(mote) is None:
                counter[mote] = 0

            counter[mote] = counter[mote] + 1

            if counter.get(mote) > 1000:

                features = {"cluster": 0, "mote": 0, "cycle": 0, "transmission_interval": 0, "power_setting": -1}

                for transmission in read_data.get(motenumber):
                    if transmission["transmission_power_setting"] != -1000:
                        if transmission.get("transmission_interval") < 0:
                            transmission["transmission_interval"] = (256 + transmission["transmission_interval"]/5)*5

                        features["power_setting"] = transmission.get("transmission_power_setting")
                        features["transmission_interval"] = transmission.get("transmission_interval")

                if cycle[cluster].get(mote) is None:
                    cycle[cluster][mote] = 0
                features["cluster"] = cluster
                features["mote"] = mote
                features["cycle"] = cycle[cluster][mote]
                for feature in features:
                    data[feature].append(features[feature])
                cycle[cluster][mote] = cycle[cluster][mote] + 1



    return data


def readDataGlobal(resultsfileName):
    data = {"cycle": [dict()], "packet_loss": [dict()], "energy_consumption": [dict()],
            "freshness": [dict()], "utility": [dict()]}
    results_file = open(resultsfileName, "r")
    clustering = dict()
    cycle = [dict()]
    cluster = 0
    counter = dict()
    cluster_time_interval = 10000
    next_cluster_time = cluster_time_interval
    for line in results_file.readlines():
        if line == "changed\n":
            before = 1
            counter = dict()
        elif line[:len("recluster: ")] == "recluster: ":
            # cluster += 1
            # cycle.append(dict())
            # for feature in data:
            #    data[feature].append(dict())

            to_search = line[len("recluster: "):-1]
            correcting_index = to_search.find(": [")
            dictstring = ""
            while correcting_index > -1:
                if to_search[correcting_index - 2] == " " or to_search[correcting_index - 2] == "{":
                    dictstring += to_search[:correcting_index - 1]
                    dictstring += '"'
                    dictstring += to_search[correcting_index - 1]
                    dictstring += '"'
                else:
                    dictstring += to_search[:correcting_index - 2]
                    dictstring += '"'
                    dictstring += to_search[correcting_index - 2]
                    dictstring += to_search[correcting_index - 1]
                    dictstring += '"'
                dictstring += to_search[correcting_index:correcting_index + 1]
                to_search = to_search[correcting_index + 1:]
                correcting_index = to_search.find(": [")
            dictstring += to_search
            cluster_data = json.loads(dictstring)
            set_size = {"same": 0, "change": 0}
            for cluster_motes in cluster_data:
                for mote_to_cluster in cluster_data[cluster_motes]:
                    if clustering.get(mote_to_cluster) is None or clustering[mote_to_cluster] == int(cluster_motes):
                        set_size["same"] = set_size["same"] + 1
                    else:
                        set_size["change"] = set_size["change"] + 1
                    clustering[mote_to_cluster] = int(cluster_motes)
            if set_size["same"] < set_size["change"]:
                for mote_to_cluster in clustering:
                    clustering[mote_to_cluster] = (clustering[mote_to_cluster] + 1) % 2


        elif not all(c.isdigit() for c in line[:-2]):
            read_data = json.loads(line)
            motenumber = list(read_data.keys())[0]
            mote = "global"

            if counter.get(motenumber) is None:
                counter[motenumber] = 0
            counter[motenumber] = counter[motenumber] + 1

            first_arrived_signal = 0
            for transmission in read_data.get(motenumber):
                if transmission["departure_time"] > 0:
                    first_arrived_signal = transmission["departure_time"]

            if first_arrived_signal < next_cluster_time - cluster_time_interval:
                next_cluster_time = cluster_time_interval

            if first_arrived_signal > next_cluster_time:
                next_cluster_time += cluster_time_interval
                cluster += 1
                cycle.append(dict())
                for feature in data:
                    data[feature].append(dict())

            if counter.get(motenumber) > 0:

                features = {"packet_loss": 0, "energy_consumption": 0, "freshness": 0, "utility": [0, 0, 0]}
                if cycle[cluster].get(mote) is None:
                    data["cycle"][cluster][mote] = list()
                    for feature in features:
                        data[feature][cluster][mote] = list()
                    cycle[cluster][mote] = 0

                last_transmitted = None
                for transmission in read_data.get(list(read_data.keys())[0]):
                    if transmission["transmission_power_setting"] != -1000:
                        if last_transmitted is not None and transmission.get("departure_time") - last_transmitted > 0 and transmission.get("departure_time") - last_transmitted < 400:
                            features["utility"][0] += 1 - max(0, (
                                        transmission.get("departure_time") - last_transmitted - 180)) \
                                                     / (transmission.get("departure_time") - last_transmitted + 1)

                        last_transmitted = transmission.get("departure_time")

                        if transmission.get("expiration_time") < 0:
                            transmission["expiration_time"] = (256+ transmission["expiration_time"]/5)*5

                        features["utility"][1] = features["utility"][1] + 1 - max(0, (
                                transmission.get("latency") * transmission.get("expiration_time") / 100 - 15) /
                                                                              (transmission.get(
                                                                                  "latency") * transmission.get(
                                                                                  "expiration_time") / 100 + 1))

                        features["energy_consumption"] = math.pow(10, (
                                transmission["transmission_power_setting"] - 30) / 10)
                        features["freshness"] = features.get("freshness") + transmission["latency"] * transmission.get(
                            "expiration_time") / 100
                    else:
                        features["packet_loss"] = features.get("packet_loss") + 1
                        last_transmitted = None

                features["freshness"] = features.get("freshness") / (10 - features["packet_loss"])

                features["utility"][0] = features["utility"][0] / (10 - features["packet_loss"])
                features["utility"][1] = features["utility"][1] / (10 - features["packet_loss"])

                features["utility"][2] = min(max(1.0 - (features.get("packet_loss") / 10 - 0.1) * 4, 0), 1.0)
                if False:  # cycle[cluster][mote] > 0:
                    features["packet_loss"] = data["packet_loss"][cluster][mote][-1] * cycle[cluster][mote] / (
                            cycle[cluster][mote] + 1) + features.get("packet_loss") / (
                                                      10 * (cycle[cluster][mote] + 1))
                else:
                    features["packet_loss"] = features.get("packet_loss") / 10

                if True:  # cluster == 1:
                    features["utility"] = features["utility"][0] * 0.3 + features["utility"][1] * 0.3 + features["utility"][2] * 0.4
                else:
                    features["utility"] = sum(features["utility"]) / 3
                for feature in features:
                    data[feature][cluster][mote].append(features[feature])
                data["cycle"][cluster][mote].append(cycle[cluster][mote])
                cycle[cluster][mote] = cycle[cluster][mote] + 1

    return data


# for mote in data:
#     df = pd.DataFrame.from_dict(data[mote])
#     figures.append(px.line(df, x="cycle", y="energy_consumption", title="switching_goals"))
#
# for figure in figures:
#     figure.show()


def plot_metrics(metrics, plot_names, cycles, metric_name, shape_name, is_global=False, y_range=None, line_colors=None):
    fig = make_subplots(rows=1, cols=len(metrics))
    maxlist = list()
    for cluster in range(len(metrics)):
        if len(metrics[cluster]) > 0:
            for sublist in metrics[cluster].values():
                maxlist.append(np.amax(sublist))

    max_val = np.amax(maxlist)

    for cluster in range(len(metrics)):
        for mote in range(len(metrics[cluster])):
            if line_colors is not None:
                fig.add_trace(go.Scatter(x=cycles[cluster].get(plot_names[mote]),
                                         y=metrics[cluster][list(metrics[cluster].keys())[mote]], name=plot_names[mote],
                                         line_shape="spline", line_width=2), row=1, col=cluster + 1)
            else:
                fig.add_trace(go.Scatter(x=cycles[cluster].get(plot_names[mote]),
                                         y=metrics[cluster][list(metrics[cluster].keys())[mote]],
                                         name=plot_names[mote]), row=1, col=cluster + 1)

    # for i in range(len(metrics[1])):
    #    if line_colors is not None:
    #        fig.add_trace(go.Scatter(x=cycles[1].get(plot_names[i]), y=metrics[1][i], name=plot_names[i],
    #                                 line=dict(color=line_colors[i]), line_shape="spline", line_width=2), row=1, col=2)
    #    else:
    #        fig.add_trace(go.Scatter(x=cycles[1].get(plot_names[i]), y=metrics[1][i], name=plot_names[i]), row=1, col=2)

    if y_range is None:
        y_range = [0, max_val + 5.0 * max_val / 100.0]
    fig.update_layout(
        yaxis=dict(
            title=metric_name,
            range=y_range
        ),
        xaxis=dict(
            title='Adaptation cycle',
            # tickvals = years_count[cYear]
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.7
        ),
        # title={
        # 'text': text_title,
        # 'y':.99,
        # 'x':0.5,
        # 'xanchor': 'center',
        # 'yanchor': 'top'}
    )

    # fig.write_image("./figures/" + shape_name + ".pdf")
    fig.write_html("./figures/" + shape_name + ".html")


def heatmap_metric(metrics, metrics_2, none_value, shape_name, colorscale= [[0, 'green'], [1, 'red']], y_range=None):

    maxlist = list()
    for metric in metrics:
        for cluster in range(len(metric)):
            if len(metric[cluster]) > 0:
                for sublist in metric[cluster].values():
                    maxlist.append(np.amax(sublist))

    max_val = np.amax(maxlist)

    data_matrix_mean = []
    for cluster in range(len(metrics[0][0])):
        data_matrix_mean.append(list())
    for cluster in range(1, len(metrics[0])):
        i = 0

        for mote in metrics[0][0]:
            metric_mean = list()
            for metric in metrics:
                if metric[cluster].get(mote) is not None:
                    metric_mean.append(np.mean(metric[cluster][mote]))

                else:
                    metric_mean.append(none_value)
            if len(metric_mean) > 0:
                data_matrix_mean[i].append(np.median(metric_mean))
            else:
                data_matrix_mean[i].append(None)
            i += 1



    if metrics_2 is not None:
        fig = make_subplots(2, 1)

        data_2_matrix_mean = []
        for cluster in range(len(metrics_2[0][0])):
            data_2_matrix_mean.append(list())
        for cluster in range(1, len(metrics_2[0])):
            i = 0
            for mote in metrics_2[0][0]:
                metric_mean = list()
                for metric in metrics_2:
                    if metric[cluster].get(mote) is not None:
                        metric_mean.append(np.mean(metric[cluster][mote]))
                    else:
                        metric_mean.append(none_value)
                if len(metric_mean) > 0:
                    data_2_matrix_mean[i].append(np.median(metric_mean))
                else:
                    data_2_matrix_mean[i].append(None)
                i += 1

        fig.add_trace(go.Heatmap(
            z=data_matrix_mean, hoverongaps=False, coloraxis = "coloraxis"), 1, 1)
        fig.add_trace(
            go.Heatmap(
                z=data_2_matrix_mean, hoverongaps=False, coloraxis = "coloraxis"), 2, 1)
    else:
        fig = go.Figure(data=go.Heatmap(
            z=data_matrix_mean, colorscale=colorscale, hoverongaps=False))
    if y_range is None:
        y_range = [0, max_val + 5.0 * max_val / 100.0]
    fig.update_layout(
        yaxis=dict(
            title='motes'
        ),
        xaxis=dict(
            title='10,000 seconds',
            # tickvals = years_count[cYear]
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.7
        ),
        coloraxis={'colorscale': colorscale}
        # title={
        # 'text': text_title,
        # 'y':.99,
        # 'x':0.5,
        # 'xanchor': 'center',
        # 'yanchor': 'top'}
    )

    # fig.write_image("./figures/" + shape_name + ".pdf")
    fig.write_html("./figures/" + shape_name + "_mean.html")

    data_matrix_max = []
    for cluster in range(len(metrics[0][0])):
        data_matrix_max.append(list())
    for cluster in range(1, len(metrics[0])):
        i = 0

        for mote in metrics[0][0]:
            metric_max = list()
            for metric in metrics:
                if metric[cluster].get(mote) is not None:
                    metric_max.append(np.max(metric[cluster][mote]))
                else:
                    metric_max.append(none_value)
            if len(metric_max) > 0:
                data_matrix_max[i].append(np.median(metric_max))
            else:
                data_matrix_max[i].append(None)
            i += 1
    if metrics_2 is not None:
        fig = make_subplots(2, 1)

        data_2_matrix_max = []
        for cluster in range(len(metrics_2[0][0])):
            data_2_matrix_max.append(list())
        for cluster in range(1, len(metrics_2[0])):
            i = 0

            for mote in metrics_2[0][0]:
                metric_max = list()
                for metric in metrics_2:
                    if metric[cluster].get(mote) is not None:
                        metric_max.append(np.max(metric[cluster][mote]))
                    else:
                        metric_max.append(none_value)
                if len(metric_max) > 0:
                    data_2_matrix_max[i].append(np.median(metric_max))
                else:
                    data_2_matrix_max[i].append(None)
                i += 1

        fig.add_trace(go.Heatmap(
            z=data_matrix_max, hoverongaps=False, coloraxis="coloraxis"), 1, 1)
        fig.add_trace(
            go.Heatmap(
                z=data_2_matrix_max, hoverongaps=False, coloraxis="coloraxis"), 2, 1)
    else:
        fig = go.Figure(data=go.Heatmap(
            z=data_matrix_max, colorscale=colorscale, hoverongaps=False))
    if y_range is None:
        y_range = [0, max_val + 5.0 * max_val / 100.0]
    fig.update_layout(
        yaxis=dict(
            title='motes'
        ),
        xaxis=dict(
            title='10,000 seconds',
            # tickvals = years_count[cYear]
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.7
        ),
        coloraxis={'colorscale': colorscale}
        # title={
        # 'text': text_title,
        # 'y':.99,
        # 'x':0.5,
        # 'xanchor': 'center',
        # 'yanchor': 'top'}
    )

    # fig.write_image("./figures/" + shape_name + ".pdf")
    fig.write_html("./figures/" + shape_name + "_max.html")

    data_matrix_min = []
    for cluster in range(len(metrics[0][0])):
        data_matrix_min.append(list())
    for cluster in range(1, len(metrics[0])):
        i = 0

        for mote in metrics[0][0]:
            metric_min = list()
            for metric in metrics:
                if metric[cluster].get(mote) is not None:
                    metric_min.append(np.min(metric[cluster][mote]))
                else:
                    metric_min.append(none_value)
            if len(metric_min) > 0:
                data_matrix_min[i].append(np.median(metric_min))
            else:
                data_matrix_min[i].append(None)
            i += 1

    if metrics_2 is not None:
        fig = make_subplots(2, 1)

        data_2_matrix_min = []
        for cluster in range(len(metrics_2[0][0])):
            data_2_matrix_min.append(list())
        for cluster in range(1, len(metrics_2[0])):
            i = 0

            for mote in metrics_2[0][0]:
                metric_min = list()
                for metric in metrics_2:
                    if metric[cluster].get(mote) is not None:
                        metric_min.append(np.min(metric[cluster][mote]))
                    else:
                        metric_min.append(none_value)
                if len(metric_min) > 0:
                    data_2_matrix_min[i].append(np.median(metric_min))
                else:
                    data_2_matrix_min[i].append(None)
                i += 1

        fig.add_trace(go.Heatmap(
            z=data_matrix_min, hoverongaps=False, coloraxis="coloraxis"), 1, 1)
        fig.add_trace(
            go.Heatmap(
                z=data_2_matrix_min, hoverongaps=False, coloraxis="coloraxis"), 2, 1)
    else:
        fig = go.Figure(data=go.Heatmap(
            z=data_matrix_min, colorscale=colorscale, hoverongaps=False))
    if y_range is None:
        y_range = [0, max_val + 5.0 * max_val / 100.0]
    fig.update_layout(
        yaxis=dict(
            title='motes'
        ),
        xaxis=dict(
            title='10,000 seconds',
            # tickvals = years_count[cYear]
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.7
        ),
        coloraxis={'colorscale': colorscale}
        # title={
        # 'text': text_title,
        # 'y':.99,
        # 'x':0.5,
        # 'xanchor': 'center',
        # 'yanchor': 'top'}
    )

    # fig.write_image("./figures/" + shape_name + ".pdf")
    fig.write_html("./figures/" + shape_name + "_min.html")

    data_matrix_median = []
    for cluster in range(len(metrics[0][0])):
        data_matrix_median.append(list())
    for cluster in range(1, len(metrics[0])):
        i = 0

        for mote in metrics[0][0]:
            metric_median = list()
            for metric in metrics:
                if metric[cluster].get(mote) is not None:
                    metric_median.append(np.median(metric[cluster][mote]))
                else:
                    metric_median.append(none_value)
            if len(metric_median) > 0:
                data_matrix_median[i].append(np.median(metric_median))
            else:
                data_matrix_median[i].append(None)
            i += 1

    if metrics_2 is not None:
        fig = make_subplots(2, 1)

        data_2_matrix_median = []
        for cluster in range(len(metrics_2[0][0])):
            data_2_matrix_median.append(list())
        for cluster in range(1, len(metrics_2[0])):
            i = 0

            for mote in metrics_2[0][0]:
                metric_median = list()
                for metric in metrics_2:
                    if metric[cluster].get(mote) is not None:
                        metric_median.append(np.median(metric[cluster][mote]))
                    else:
                        metric_median.append(none_value)
                if len(metric_median) > 0:
                    data_2_matrix_median[i].append(np.median(metric_median))
                else:
                    data_2_matrix_median[i].append(None)
                i += 1

        fig.add_trace(go.Heatmap(
            z=data_matrix_median, hoverongaps=False, coloraxis="coloraxis"), 1, 1)
        fig.add_trace(
            go.Heatmap(
                z=data_2_matrix_median, hoverongaps=False, coloraxis="coloraxis"), 2, 1)
    else:
        fig = go.Figure(data=go.Heatmap(
            z=data_matrix_max, colorscale=colorscale, hoverongaps=False))
    if y_range is None:
        y_range = [0, max_val + 5.0 * max_val / 100.0]
    fig.update_layout(
        yaxis=dict(
            title='motes'
        ),
        xaxis=dict(
            title='10,000 seconds',
            # tickvals = years_count[cYear]
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.7
        ),
        coloraxis={'colorscale': colorscale}
        # title={
        # 'text': text_title,
        # 'y':.99,
        # 'x':0.5,
        # 'xanchor': 'center',
        # 'yanchor': 'top'}
    )

    # fig.write_image("./figures/" + shape_name + ".pdf")
    fig.write_html("./figures/" + shape_name + "_median.html")


def plot_metrics_single(metrics, plot_names, cycles, metric_name, shape_name, y_range=None, line_colors=None):
    plots = []
    maxlist = list()
    for subslist in metrics[0]:
        maxlist.append(np.amax(subslist))

    max_val = np.amax(maxlist)

    for i in range(len(metrics)):
        if line_colors is not None:
            plots.append(go.Scatter(x=cycles[0].get(plot_names[i]), y=metrics[i], name=plot_names[i],
                                    line_shape="spline", line_width=2))
        else:
            plots.append(go.Scatter(x=cycles[0].get(plot_names[i]), y=metrics[i], name=plot_names[i]))

    fig = go.Figure(plots)
    if y_range is None:
        y_range = [0, max_val + 5.0 * max_val / 100.0]
    fig.update_layout(
        yaxis=dict(
            title=metric_name,
            range=y_range
        ),
        xaxis=dict(
            title='Adaptation cycle',
            # tickvals = years_count[cYear]
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.7
        ),
        # title={
        # 'text': text_title,
        # 'y':.99,
        # 'x':0.5,
        # 'xanchor': 'center',
        # 'yanchor': 'top'}
    )

    # fig.write_image("./figures/" + shape_name + ".pdf")
    fig.write_html("./figures/" + shape_name + ".html")


def plot_box_new(metrics, plot_names, cycles, metric_name, shape_name, is_global=True, line_colors=None):

    units = {"cycle": "", "global packet_loss": "(%)", "global energy_consumption": "(W/byte)",
             "global freshness": "(s)", "global utility": ""}

    if not is_global:
        fig = make_subplots(rows=1, cols=1, shared_yaxes=True)
        for i in range(len(metrics[0])):

            if line_colors is not None:
                fig.add_trace(go.Box(x=cycles[0].get(plot_names[i]), y=metrics[0][i], name=(plot_names[i]),
                                     ), row=1, col=1)
            else:
                fig.add_trace(go.Box(x=cycles[1].get(plot_names[i]), y=metrics[0][i], name=plot_names[i]), row=1, col=1)


        fig.update_layout(yaxis_title=metric_name)

    else:
        fig = make_subplots(rows=1, cols=3, column_widths=[0.25, 0.25, 0.25])
        for feature in range(len(metrics[0]) - 1):
            if line_colors is not None:
                fig.add_trace(go.Box(x=np.full((len(metrics[0][feature])), " before"), y=metrics[0][feature],
                                     name=(plot_names[feature + 1]), ), row=1, col=feature + 1)
                fig.add_trace(go.Box(x=np.full((len(metrics[1][feature])), " after"), y=metrics[1][feature],
                                     name=(plot_names[feature + 1]), ), row=1, col=feature + 1)
            else:
                fig.add_trace(go.Box(x=cycles[0].get(plot_names[feature + 1]), y=metrics[0][feature],
                                     name=plot_names[feature + 1]), row=1, col=1 + feature)
                fig.add_trace(
                    go.Box(x=cycles[0].get(plot_names[feature + 1]), y=metrics[1][feature],
                           name=plot_names[feature + 1]),
                    row=1, col=1 + feature)

            fig.update_yaxes(title_text=plot_names[feature + 1] + units[plot_names[feature + 1]], row=1,
                             col=feature + 1)

    fig.update_layout(showlegend=False)

    # fig.write_image("./figures/" + shape_name + ".pdf")
    fig.write_html("./figures/" + shape_name + ".html")


def plotAll(data_list, data_2_list = None, name = "trial"):
    feature = list(data_list[0].keys())[0]

    max_freshness = 600
    fifteen = 15/max_freshness
    colorscales = { "cluster": [[0, 'blue'], [0.5, 'blue'], [0.5, 'yellow'], [1, 'yellow']], "packet_loss":  [[0, 'green'], [0.1, 'green'], [0.35, 'red'], [1, 'red']], "energy_consumption":  [[0, 'green'], [1, 'red']], "freshness":  [[0, 'green'], [fifteen, 'green'], [1, 'red']], "utility": [[0, 'red'], [1, 'green']]}
    none_values = {"cluster": 0,
                   "packet_loss": 1,
                   "energy_consumption": 0,
                   "freshness": 0,
                   "utility": 0,
                   "transmission_interval": 0,
                   "action": 0,
                   "received_transmission_interval":0}
    names = list()
    max_motes = 0

    for i in range(len(data_list[0].get(feature))):
        if len(list(data_list[0].get(feature)[i].keys())) > max_motes:
            max_motes = len(list(data_list[0].get(feature)[i].keys()))
    for motenumber in range(max_motes):
        names.append("mote " + str(motenumber + 1))

    for feature in data_list[0]:

        feature_data = list()
        for data in data_list:
            feature_data.append(data.get(feature))
        if feature != "cycle":
            #plot_metrics(feature_data,  names,
            #             data.get("cycle"), feature,
            #             name + "_goals_" + feature)

            if data_2_list is not None:
                feature_data_2 = list()
                for data_2 in data_2_list:
                    feature_data_2.append(data_2.get(feature))
                heatmap_metric(feature_data,feature_data_2, none_values.get(feature),
                            name + "_goals_" + feature + "_heatmap", colorscale= colorscales.get(feature, [[0, 'green'], [1, 'red']]))
            else:
                heatmap_metric(feature_data, None, none_values.get(feature),
                               name + "_goals_" + feature + "_heatmap", colorscale= colorscales.get(feature, [[0, 'green'], [1, 'red']]))
            if False:  # feature == "energy_consumption":

                cluster = 0
                for data_list in data.get(feature):
                    name_index = 0
                    cluster += 1
                    for mote in data_list:
                        data_mote = [data_list.get(mote)]
                        plot_metrics_single(data_mote, [names[name_index]], data.get("cycle"), feature,
                                            name + "_goals_" + feature + " mote " + str(
                                                name_index + 1) + "_phase_" + str(cluster))
                        name_index += 1


def plotDensity(data):

    fig = px.density_heatmap(data, x="transmission_interval", y="power_setting")
    fig.write_html("./figures/"+"density_transmission_interval_transmission_power.html")


def plotAllBox(data, name, is_global=False):
    units = {"cycle": "", "packet_loss": "(%)", "energy_consumption": "(W/byte)", "freshness": "(s)", "utility": ""}
    feature = list(data.keys())[0]
    names = list()
    global_data = [[], []]
    if is_global:
        for feature in data:
            if feature != "cycle":
                global_data[0].append([list(data.get(feature)[0].values())[0]])
    for motenumber in range(len(list(data.get(feature)[0].keys()))):
        if is_global:
            for feature in data:
                names.append("global " + feature)
        else:
            names.append("mote " + str(motenumber + 1))

    if (is_global):
        plot_box_new(global_data, names,
                     data.get("cycle"), "all goals",
                     name + "_goals_box", is_global)
    else:
        for feature in data:
            if feature != "cycle":
                plot_box_new([list(data.get(feature)[0].values())], names,
                             data.get("cycle"), feature + units.get(feature),
                             name + "_goals_box_" + feature, is_global)


# data_two = readData("results.txt")
data_switch = readData("results_paper_recluster_better.txt")
data_no_switch = readData("results_for_action.txt")
data_no = list()
#data_no.append(readData("results_no_0.txt"))
data_no.append(readDataJava("moteData_dynamic gaussian.txt"))

data_nothing = list()
data_nothing.append(readDataJava("moteData_single.txt"))
#data_nothing.append(readData("results_nothing_maybe.txt"))
data_run = list()
data_run.append(readDataJava("moteData_dynamic gaussian.txt"))
data_global =list()
data_global.append(readDataGlobal("results_new_utility_seed.txt"))
data_no_switch_global =list()
data_no_switch_global.append(readDataGlobal("results_new_utility_static_seeded.txt"))

data_power_setting = readDataPowerSetting("results_run_2.txt")
# [data_three, cycle_three] = readData("results_three_two.txt")
#print(stats.spearmanr(data_power_setting["power_setting"],data_power_setting["transmission_interval"]))

utility =list()
boxplot_data = {"utility":list(), "mote":list(), "period": list(), "run":list()}
motes =list()
cluster = list()
packet_loss = list()
freshness = list()
iqr= list()
medians = list()
cluster_0_size = list()
mote_utility = dict()
for data in data_run:
    packet_loss.append(list())
    freshness.append(list())
    period_counter = 0
    packet_loss_period = list()
    freshness_period = list()
    data_utility = list()
    cluster_0_size.append({"size": list(), "period": list(), "transmission_interval": list(), "cluster": list()})
    for period in range(len(data["utility"])):

        for mote in mote_utility:
            mote_utility[mote] = list()

        if period > 0:

            cluster_0_size[-1]["period"].append(period)
            cluster_0_size[-1]["size"].append(0)
            cluster_0_size[-1]["period"].append(period)
            cluster_0_size[-1]["size"].append(0)

            period_counter += 1
            utility_0 = list()
            utility_1 = list()
            if period_counter % 1 == 0:
                packet_loss[len(packet_loss) - 1].append(np.mean(packet_loss_period))
                freshness[len(freshness) - 1].append(np.mean(freshness_period))
                packet_loss_period = list()
                freshness_period = list()

            for mote in data["utility"][period]:

                #if(data["cluster"][period][mote][0] == 0):
                #    cluster_0_size[-1]["size"][-1] += 1
                if mote not in mote_utility:
                    mote_utility[mote] = list()
                for transmission in range(len(data["utility"][period][mote])):
                    packet_loss_period.append(data["packet_loss"][period][mote][transmission])
                    freshness_period.append(data["freshness"][period][mote][transmission])
                    mote_utility[mote].append(data["utility"][period][mote][transmission])
                    utility_0.append(data["received_transmission_interval"][period][mote][transmission])

            utility.append(np.mean(utility_0))
            if len(utility_0) > 0:
                min_0 = np.min(utility_0)
                slope, intercept, r_value, p_value, std_err = linregress(range(len(utility_0)), utility_0)
                cluster_0_size[-1]["transmission_interval"].append(np.mean(utility_0))
            else:
                cluster_0_size[-1]["transmission_interval"].append(0)
            cluster_0_size[-1]["cluster"].append(str(0))
            cluster.append(0)
            if len(utility_1) > 0:

                slope, intercept, r_value, p_value, std_err = linregress(range(len(utility_1)), utility_1)
                cluster_0_size[-1]["transmission_interval"].append(np.mean(utility_1))
            else:
                cluster_0_size[-1]["transmission_interval"].append(0)
            cluster_0_size[-1]["cluster"].append(str(1))
            utility.append(np.mean(utility_1))
            cluster.append(1)

            for mote in mote_utility:
                for cycle_data in mote_utility[mote]:
                    data_utility.append(np.mean(mote_utility[mote]))
                    boxplot_data["utility"].append(cycle_data)
                    boxplot_data["period"].append(period)
                    boxplot_data["mote"].append(mote)
                    boxplot_data["run"].append("dynamic")
        else:
            for mote in data["utility"][period]:
                if mote not in mote_utility:
                    mote_utility[mote] = list()


    #percentiles = np.percentile(data_utility, [75, 25])
    #iqr.append(percentiles[0] - percentiles[1])
    medians.append(np.median(data_utility))
print( "dynamic: median: " + str(np.mean(medians)) + " IQR " + str(np.mean(iqr)))
fig = px.bar(cluster_0_size[0], x='period', y='size')
fig.show()


fig = px.bar(cluster_0_size[0], x='period', y='transmission_interval', color="cluster",barmode='group')
fig.show()

lastmote = 1
mote_names = dict()
new_motes_list = list()
for entry in boxplot_data["mote"]:
    if mote_names.get(entry) is None:
        mote_names[entry] = lastmote
        lastmote += 1
    new_motes_list.append(mote_names.get(entry))
boxplot_data["mote"] = new_motes_list

fig = px.box(boxplot_data, x="mote", y="utility")
fig.show()

utility_no =list()
cluster_no = list()
packet_loss_no = list()
freshness_no = list()
iqr_no= list()
medians_no =list()
boxplot_data_no = {"utility": list(), "mote": list(), "period":list(), "run":list()}
for data in data_no:
    data_utility = list()
    packet_loss_no.append(list())
    freshness_no.append(list())
    packet_loss_period = list()
    freshness_period = list()
    period_counter = 0
    for period in range(len(data["utility"])):

        mote_utility = dict()

        if period > 300:
            period_counter += 1
            utility_0 = list()
            utility_1 = list()
            if period_counter % 20 == 0:
                packet_loss_no[len(packet_loss_no) - 1].append(np.mean(packet_loss_period))
                freshness_no[len(freshness_no) - 1].append(np.mean(freshness_period))
                packet_loss_period = list()
                freshness_period = list()
            for mote in data["utility"][period]:
                if mote not in mote_utility:
                    mote_utility[mote] = list()
                for cycle in range(len(data["utility"][period][mote])):
                    packet_loss_period.append(data["packet_loss"][period][mote][cycle])
                    freshness_period.append(data["freshness"][period][mote][cycle])
                    mote_utility[mote].append(data["utility"][period][mote][cycle])

                    utility_0.append(data["utility"][period][mote][cycle])

            utility_no.append(np.mean(utility_0))
            cluster_no.append(0)
            utility_no.append(np.mean(utility_1))
            cluster_no.append(1)

            for mote in mote_utility:
                    data_utility.append(np.mean(mote_utility[mote]))

                    boxplot_data_no["utility"].append(np.mean(mote_utility[mote]))
                    boxplot_data_no["period"].append(period)
                    boxplot_data_no["mote"].append(mote)
                    boxplot_data_no["run"].append("static")

    #percentiles = np.percentile(data_utility, [75, 25])
    #iqr_no.append(percentiles[0] - percentiles[1])
    medians_no.append(np.median(data_utility))
new_motes_list = list()
for entry in boxplot_data_no["mote"]:
    if mote_names.get(entry) is None:
        mote_names[entry] = lastmote
        lastmote += 1
    new_motes_list.append(mote_names.get(entry))
boxplot_data_no["mote"] = new_motes_list
print( "static: median: " + str(np.mean(medians_no)) + " IQR " + str(np.mean(iqr_no)))
utility_nothing =list()
cluster_nothing = list()
packet_loss_nothing = list()
freshness_nothing = list()
boxplot_data_nothing = {"utility": list(), "mote": list(), "period":list(), "run":list()}
iqr_nothing = list()
medians_nothing =list()
for data in data_nothing:
    data_utility = list()
    packet_loss_nothing.append(list())
    freshness_nothing.append(list())
    packet_loss_period = list()
    freshness_period = list()
    period_counter = 0
    for period in range(len(data["utility"])):

        mote_utility = dict()
        total_utiliy = dict()

        if period > 0:
            period_counter += 1
            utility_0 = list()
            utility_1 = list()
            if period_counter % 1 == 0:
                packet_loss_nothing[len(packet_loss_nothing) - 1].append(np.mean(packet_loss_period))
                freshness_nothing[len(freshness_nothing) - 1].append(np.mean(freshness_period))
                packet_loss_period = list()
                freshness_period = list()
            for mote in data["utility"][period]:
                if mote not in mote_utility:
                    mote_utility[mote] = list()
                for transmission in range(len(data["utility"][period][mote])):
                    packet_loss_period.append(data["packet_loss"][period][mote][transmission])
                    freshness_period.append(data["freshness"][period][mote][transmission])
                    mote_utility[mote].append(data["utility"][period][mote][transmission])
                    utility_0.append(data["utility"][period][mote][transmission])
            utility_nothing.append(np.mean(utility_0))
            cluster_nothing.append(0)
            utility_nothing.append(np.mean(utility_1))
            cluster_nothing.append(1)

            for mote in mote_utility:
                    data_utility.append(np.mean(mote_utility[mote]))
                    boxplot_data_nothing["utility"].append(np.mean(mote_utility[mote]))
                    boxplot_data_nothing["period"].append(period)
                    boxplot_data_nothing["mote"].append(mote)
                    boxplot_data_nothing["run"].append("single")
    #percentiles = np.percentile(data_utility,[75 ,25])
    #iqr_nothing.append(percentiles[0]-percentiles[1])
    medians_nothing.append(np.median(data_utility))

new_motes_list = list()
for entry in boxplot_data_nothing["mote"]:
    if mote_names.get(entry) is None:
        mote_names[entry] = lastmote
        lastmote += 1
    new_motes_list.append(mote_names.get(entry))
boxplot_data_nothing["mote"] = new_motes_list

fig = px.box(boxplot_data_nothing, x="mote", y="utility")
fig.show()

fairness = 0
for mote in list(mote_names.values()):
    median_mote_dynamic = np.median([boxplot_data["utility"][j] for j in [i for i in range(len(boxplot_data["mote"])) if boxplot_data["mote"][i] == mote]])
    median_mote_single = np.median([boxplot_data_nothing["utility"][j] for j in [i for i in range(len(boxplot_data_nothing["mote"])) if boxplot_data_nothing["mote"][i] == mote]])
    if (len([i for i in range(len(boxplot_data_nothing["mote"])) if boxplot_data_nothing["mote"][i] == mote]) == 0):
        print(list(mote_names.keys())[mote])
    print(str(median_mote_single) +" vs "+str(median_mote_dynamic))
    fairness += (median_mote_single - median_mote_dynamic)

print("fairness: " +str(fairness))

print( "single: median: " + str(np.mean(medians_nothing)) + " IQR " + str(np.mean(iqr_nothing)))

fig = make_subplots(rows=3, cols=1)
fig.add_trace(go.Box( x=boxplot_data_nothing["period"], y=boxplot_data_nothing["utility"], name="single"),row=1,col=1)
fig.add_trace(go.Box( x=boxplot_data_no["period"], y=boxplot_data_no["utility"], name="static"),row=2,col=1)
fig.add_trace(go.Scatter( x=boxplot_data["period"], y=boxplot_data["utility"], name="dynamic", mode='markers',marker=dict(
        size=16,
        color=boxplot_data["mote"], #set color equal to a variable
        colorscale='Viridis', # one of plotly colorscales
        showscale=True
    )),row=3,col=1)
fig.update_layout(yaxis_title="utility")
fig.update_layout(xaxis_title="time unit")
fig.show()
fig = make_subplots(rows=1, cols=3)
fig.add_trace(go.Box( x=boxplot_data_nothing["run"], y=boxplot_data_nothing["utility"], name="single"),row=1,col=1)
fig.add_trace(go.Box( x=boxplot_data_no["run"], y=boxplot_data_no["utility"], name="multi"),row=1,col=2)
fig.add_trace(go.Box( x=boxplot_data["run"], y=boxplot_data["utility"], name="dynamic"),row=1,col=3)
fig.update_layout(yaxis_title="utility")
fig.update_layout(xaxis_title="")
fig.show()
print(stats.ttest_ind(boxplot_data_no["utility"][-len(boxplot_data["utility"]):],boxplot_data["utility"]))


data_global_diff = {"cycle": [dict(), dict()], "packet_loss": [dict(), dict()], "energy_consumption": [dict(), dict()],
                    "freshness": [dict(), dict()], "utility": [dict(), dict()]}

#for feature in data_switch_global:
#    for cluster in range(len(data_switch_global.get(feature))):
#        print(feature + " cluster " + str(cluster) + " : " + str(sum(data_switch_global.get(feature)[cluster]["global"])
#                                                                 / len(
#            data_switch_global.get(feature)[cluster]["global"]))
#              + " stdev: " + str(statistics.stdev(data_switch_global.get(feature)[cluster]["global"])))

data_global_diff = {"cycle": [dict(), dict()], "packet_loss": [dict(), dict()], "energy_consumption": [dict(), dict()],
                    "freshness": [dict(), dict()], "utility": [dict(), dict()]}

#for feature in data_switch:
#    for cluster in range(len(data_switch.get(feature))):
#        for mote in data_switch.get(feature)[cluster]:
#            print("mote: " + str(mote) + " " + feature + " + cluster " + str(cluster) + " : " + str(
#                sum(data_switch.get(feature)[cluster][mote])
#                / len(data_switch.get(feature)[cluster][mote])))
plotDensity(data_power_setting)
# plotAll(data_two, "two")
##plotAll(data_switch, name="switch")
#plotAll([data_no_switch], name = "information")
plotAll(data_no,data_run, name = "comparision")

plotAll(data_no_switch_global, data_global, name ="global_comparison")
#plotAll(data_power_setting, name="power_setting")
# plotAll(data_three,cycle_three,"three")
# plotAllBox(data_two, "two")
# plotAllBox(data_power_setting, "power_setting")
# plotAllBox(data_switch_global, "switch_global", is_global=True)
# plotAllBox(data_switch_global, "switch_global")
# plotAllBox(data_three,cycle_three,"three")
