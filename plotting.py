import json
import math

import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
import statistics


def readData(resultsfileName):
    data = {"cycle": [dict(), dict()], "packet_loss": [dict(), dict()], "energy_consumption": [dict(), dict()],
            "freshness": [dict(), dict()], "utility": [dict(), dict()]}
    results_file = open(resultsfileName, "r")
    cycle = [dict(), dict()]
    before = 0
    counter = dict()
    for line in results_file.readlines():
        if line == "changed\n":
            for mote in cycle[before]:
                print("mote " + mote + ": " + str(cycle[before].get(mote)))
            before = 1
            counter = dict()
        else:
            read_data = json.loads(line)
            motenumber = list(read_data.keys())[0]
            mote = motenumber
            if counter.get(mote) is None:
                counter[mote] = 0
            counter[mote] = counter[mote] + 1
            if counter.get(mote) > 0:

                features = {"packet_loss": 0, "energy_consumption": 0, "freshness": 0, "utility": [0, 0, 0]}
                if cycle[before].get(mote) is None:
                    data["cycle"][before][mote] = list()
                    for feature in features:
                        data[feature][before][mote] = list()
                    cycle[before][mote] = 0

                for transmission in read_data.get(motenumber):
                    if transmission["transmission_power_setting"] != -1000:

                        features["utility"][0] = 1 - math.pow(
                            transmission.get("transmission_power_setting") / 14.0, 2)
                        features["utility"][1] = features["utility"][1] + max(0, (
                                transmission.get("latency") * transmission.get("expiration_time") / 100 - 15)) / (
                                                         transmission.get("latency") * transmission.get(
                                                     "expiration_time") / 100 + 1)

                        features["energy_consumption"] = math.pow(10, (
                                    transmission["transmission_power_setting"] - 30) / 10)
                        features["freshness"] = features.get("freshness") + transmission["latency"] * transmission.get(
                            "expiration_time") / 100


                    else:
                        features["packet_loss"] = features.get("packet_loss") + 1
                features["freshness"] = features.get("freshness") / (10 - features["packet_loss"])
                features["utility"][1] = features["utility"][1] / (10 - features["packet_loss"])

                features["utility"][2] = min(max(1.0 - (features.get("packet_loss") / 10 - 0.1) * 4, 0), 1.0)

                if cycle[before][mote] > 0:
                    features["packet_loss"] = data["packet_loss"][before][mote][-1] * cycle[before][mote] / (
                                cycle[before][mote] + 1) + features.get("packet_loss") / (
                                                          10 * (cycle[before][mote] + 1))
                else:
                    features["packet_loss"] = features.get("packet_loss") / 10

                if before == 1:
                    features["utility"] = sum(features["utility"][1:3]) / 2
                else:
                    features["utility"] = sum(features["utility"]) / 3

                for feature in features:
                    data[feature][before][mote].append(features[feature])
                data["cycle"][before][mote].append(cycle[before][mote])
                cycle[before][mote] = cycle[before][mote] + 1

    return data


def readDataGlobal(resultsfileName):
    data = {"cycle": [dict(), dict()], "packet_loss": [dict(), dict()], "freshness": [dict(), dict()],
            "energy_consumption": [dict(), dict()], "utility": [dict(), dict()]}
    results_file = open(resultsfileName, "r")
    cycle = [dict(), dict()]
    before = 0
    counter = dict()
    for line in results_file.readlines():
        if line == "changed\n":
            before = 1
            counter = dict()
        else:
            read_data = json.loads(line)
            motenumber = list(read_data.keys())[0]
            mote = "global"
            if counter.get(motenumber) is None:
                counter[motenumber] = 0
            counter[motenumber] = counter[motenumber] + 1
            if counter.get(motenumber) > 0:

                features = {"packet_loss": 0, "energy_consumption": 0, "freshness": 0, "utility": [0, 0, 0]}
                if cycle[before].get(mote) is None:
                    data["cycle"][before][mote] = list()
                    for feature in features:
                        data[feature][before][mote] = list()
                    cycle[before][mote] = 0
                for transmission in read_data.get(list(read_data.keys())[0]):
                    if transmission["transmission_power_setting"] != -1000:
                        features["utility"][0] = 1 - math.pow(
                            transmission.get("transmission_power_setting") / 14.0, 2)
                        features["utility"][1] = features["utility"][1] + max(0, (
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

                features["freshness"] = features.get("freshness") / (10 - features["packet_loss"])
                features["utility"][1] = features["utility"][1] / (10 - features["packet_loss"])
                features["utility"][2] = min(max(1.0 - (features.get("packet_loss") / 10 - 0.1) * 4, 0), 1.0)
                if cycle[before][mote] > 0:
                    features["packet_loss"] = data["packet_loss"][before][mote][-1] * cycle[before][mote] / (
                                cycle[before][mote] + 1) + features.get("packet_loss") / (
                                                          10 * (cycle[before][mote] + 1))
                else:
                    features["packet_loss"] = features.get("packet_loss") / 10

                if before == 1:
                    features["utility"] = sum(features["utility"][1:3]) / 2
                else:
                    features["utility"] = sum(features["utility"]) / 3
                for feature in features:
                    data[feature][before][mote].append(features[feature])
                data["cycle"][before][mote].append(cycle[before][mote])
                cycle[before][mote] = cycle[before][mote] + 1

    return data


# for mote in data:
#     df = pd.DataFrame.from_dict(data[mote])
#     figures.append(px.line(df, x="cycle", y="energy_consumption", title="switching_goals"))
#
# for figure in figures:
#     figure.show()


def plot_metrics(metrics, plot_names, cycles, metric_name, shape_name, is_global=False, y_range=None, line_colors=None):
    fig = make_subplots(rows=1, cols=2)
    maxlist = list()
    for subslist in metrics[0]:
        maxlist.append(np.amax(subslist))

    max_val = np.amax(maxlist)

    for i in range(len(metrics[0])):
        if line_colors is not None:
            fig.add_trace(go.Scatter(x=cycles[0].get(plot_names[i]), y=metrics[0][i], name=plot_names[i],
                                     line=dict(color=line_colors[i]), line_shape="spline", line_width=2), row=1, col=1)
        else:
            fig.add_trace(go.Scatter(x=cycles[0].get(plot_names[i]), y=metrics[0][i], name=plot_names[i]), row=1, col=1)

    for i in range(len(metrics[1])):
        if line_colors is not None:
            fig.add_trace(go.Scatter(x=cycles[1].get(plot_names[i]), y=metrics[1][i], name=plot_names[i],
                                     line=dict(color=line_colors[i]), line_shape="spline", line_width=2), row=1, col=2)
        else:
            fig.add_trace(go.Scatter(x=cycles[1].get(plot_names[i]), y=metrics[1][i], name=plot_names[i]), row=1, col=2)

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

    fig.write_image("./figures/" + shape_name + ".pdf")
    fig.write_html("./figures/" + shape_name + ".html")


def plot_metrics_single(metrics, plot_names, cycles, metric_name, shape_name, y_range=None, line_colors=None):
    plots = []
    maxlist = list()
    for subslist in metrics[0]:
        maxlist.append(np.amax(subslist))

    max_val = np.amax(maxlist)

    for i in range(len(metrics)):
        if line_colors is not None:
            plots.append(go.Scatter(x=cycles[0].get(plot_names[i]), y=metrics[i], name=plot_names[i],
                                    line=dict(color=line_colors[i]), line_shape="spline", line_width=2))
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

    fig.write_image("./figures/" + shape_name + ".pdf")
    fig.write_html("./figures/" + shape_name + ".html")


def plot_box_new(metrics, plot_names, cycles, metric_name, shape_name, is_global=True, line_colors=None):
    units = {"cycle": "", "global packet_loss": "(%)", "global energy_consumption": "(W/byte)",
             "global freshness": "(s)", "global utility": ""}

    if not is_global:
        fig = make_subplots(rows=1, cols=2, shared_yaxes=True)
        for i in range(len(metrics[0])):

            if line_colors is not None:
                fig.add_trace(go.Box(x=cycles[0].get(plot_names[i]), y=metrics[0][i], name=(plot_names[i]),
                                     line=dict(color=line_colors[i])), row=1, col=1)
            else:
                fig.add_trace(go.Box(x=cycles[1].get(plot_names[i]), y=metrics[0][i], name=plot_names[i]), row=1, col=1)
        for i in range(len(metrics[1])):
            if line_colors is not None:
                fig.add_trace(go.Box(x=cycles[0].get(plot_names[i]), y=metrics[1][i], name=(plot_names[i]),
                                     line=dict(color=line_colors[i])), row=1, col=2)
            else:
                fig.add_trace(go.Box(x=cycles[1].get(plot_names[i]), y=metrics[1][i], name=plot_names[i]), row=1, col=2)

        fig.update_layout(yaxis_title=metric_name)

    else:
        fig = make_subplots(rows=1, cols=3, column_widths=[0.25, 0.25, 0.25])
        for feature in range(len(metrics[0]) - 1):
            if line_colors is not None:
                fig.add_trace(go.Box(x=np.full((len(metrics[0][feature])), " before"), y=metrics[0][feature],
                                     name=(plot_names[feature + 1]),
                                     line=dict(color=line_colors[feature])), row=1, col=feature + 1)
                fig.add_trace(go.Box(x=np.full((len(metrics[1][feature])), " after"), y=metrics[1][feature],
                                     name=(plot_names[feature + 1]),
                                     line=dict(color=line_colors[feature])), row=1, col=feature + 1)
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

    fig.write_image("./figures/" + shape_name + ".pdf")
    fig.write_html("./figures/" + shape_name + ".html")


def plotAll(data, name):
    feature = list(data.keys())[0]
    colors = [
        '#1f77b4',  # muted blue
        '#ff7f0e',  # safety orange
        '#2ca02c',  # cooked asparagus green
        '#d62728',  # brick red
        '#9467bd',  # muted purple
        '#8c564b',  # chestnut brown
        '#e377c2',  # raspberry yogurt pink
        '#7f7f7f',  # middle gray
        '#bcbd22',  # curry yellow-green
        '#17becf'  # blue-teal
    ]
    names = list()
    for motenumber in range(len(list(data.get(feature)[0].keys()))):
        names.append("mote " + str(motenumber + 1))

    for feature in data:
        if feature != "cycle":
            plot_metrics([list(data.get(feature)[0].values()), list(data.get(feature)[1].values())], names,
                         data.get("cycle"), feature,
                         name + "_goals_" + feature, line_colors=colors)
            if feature == "energy_consumption":

                before = 0
                for data_list in data.get(feature):
                    name_index = 0
                    before += 1
                    for mote in data_list:
                        data_mote = [data_list.get(mote)]
                        if before > 1:
                            data_mote = [data.get(feature)[0].get(mote)]
                            data_mote[0].extend(data_list.get(mote))
                        plot_metrics_single(data_mote, [names[name_index]], data.get("cycle"), feature,
                                            name + "_goals_" + feature + " mote " + str(name_index + 1))
                        name_index += 1


def plotAllBox(data, name, is_global=False):
    units = {"cycle": "", "packet_loss": "(%)", "energy_consumption": "(W/byte)", "freshness": "(s)", "utility": ""}
    feature = list(data.keys())[0]
    colors = [
        '#1f77b4',  # muted blue
        '#ff7f0e',  # safety orange
        '#2ca02c',  # cooked asparagus green
        '#d62728',  # brick red
        '#9467bd',  # muted purple
        '#8c564b',  # chestnut brown
        '#e377c2',  # raspberry yogurt pink
        '#7f7f7f',  # middle gray
        '#bcbd22',  # curry yellow-green
        '#17becf'  # blue-teal
    ]
    names = list()
    global_data = [[], []]
    if is_global:
        for feature in data:
            if feature != "cycle":
                global_data[0].append(list(data.get(feature)[0].values())[0])
                global_data[1].append(list(data.get(feature)[1].values())[0])
    for motenumber in range(len(list(data.get(feature)[0].keys()))):
        if is_global:
            for feature in data:
                names.append("global " + feature)
        else:
            names.append("mote " + str(motenumber + 1))

    if (is_global):
        plot_box_new(global_data, names,
                     data.get("cycle"), "all goals",
                     name + "_goals_box", is_global, line_colors=colors)
    else:
        for feature in data:
            if feature != "cycle":
                plot_box_new([list(data.get(feature)[0].values()), list(data.get(feature)[1].values())], names,
                             data.get("cycle"), feature + units.get(feature),
                             name + "_goals_box_" + feature, is_global, line_colors=colors)


#data_two = readData("results.txt")
data_switch = readData("results_last.txt")
data_switch_global = readDataGlobal("results_last.txt")
# [data_three, cycle_three] = readData("results_three_two.txt")

data_global_diff = {"cycle": [dict(), dict()], "packet_loss": [dict(), dict()], "energy_consumption": [dict(), dict()],
                    "freshness": [dict(), dict()], "utility": [dict(), dict()]}

for feature in data_switch_global:
    print(feature + " before: " + str(sum(data_switch_global.get(feature)[0]["global"]) / len(
        data_switch_global.get(feature)[0]["global"])) + " stdev: " + str(
        statistics.stdev(data_switch_global.get(feature)[0]["global"])))
    print(feature + " after: " + str(sum(data_switch_global.get(feature)[1]["global"]) / len(
        data_switch_global.get(feature)[1]["global"])) + " stdev: " + str(
        statistics.stdev(data_switch_global.get(feature)[0]["global"])))

#plotAll(data_two, "two")
#plotAll(data_switch, "switch")
plotAll(data_switch_global, "switch_global")
# plotAll(data_three,cycle_three,"three")
#plotAllBox(data_two, "two")
#plotAllBox(data_switch, "switch")
#plotAllBox(data_switch_global, "switch_global", is_global=True)
plotAllBox(data_switch_global, "switch_global")
# plotAllBox(data_three,cycle_three,"three")
