import json
import math

import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
import plotly.express as px
import statistics


def readData(resultsfileName):
    data = {"cycle": [dict()], "packet_loss": [dict()], "energy_consumption": [dict()],
            "freshness": [dict()], "utility": [dict()], "cluster": [dict()]}
    results_file = open(resultsfileName, "r")
    clustering = dict()
    cycle = [dict()]
    cluster = 0
    counter = dict()
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

            if counter[mote] % 300 == 0:
                cluster += 1
                cycle.append(dict())
                for feature in data:
                    data[feature].append(dict())

            if counter.get(mote) > 0:
                features = {"packet_loss": 0, "energy_consumption": 0, "freshness": 0, "utility": [0, 0, 0], "cluster": clustering.get(int(mote),0)}
                if cycle[cluster].get(mote) is None:
                    data["cycle"][cluster][mote] = list()
                    for feature in features:
                        data[feature][cluster][mote] = list()
                    cycle[cluster][mote] = 0

                for transmission in read_data.get(motenumber):
                    if transmission["transmission_power_setting"] != -1000:

                        features["utility"][0] = 1 - math.pow(
                            transmission.get("transmission_power_setting") / 14.0, 2)
                        features["utility"][1] = features["utility"][1] + max(0, (
                                transmission.get("latency") * transmission.get("expiration_time") / 100 - 35)) / (
                                                         transmission.get("latency") * transmission.get(
                                                     "expiration_time") / 100 + 1)

                        features["energy_consumption"] = math.pow(10, (
                                transmission["transmission_power_setting"] - 30) / 10)
                        features["freshness"] = features.get("freshness") + transmission["latency"] * transmission.get(
                            "expiration_time") / 100


                    else:
                        features["packet_loss"] = features.get("packet_loss") + 1
                features["freshness"] = features.get("freshness") / (10 - features["packet_loss"])
                features["utility"][1] = 1 - features["utility"][1] / (10 - features["packet_loss"])
                features["utility"][2] = min(max(1.0 - (features.get("packet_loss") / 10 - 0.1) * 4, 0), 1.0)

                if False:  # cycle[cluster][mote] > 0:
                    features["packet_loss"] = data["packet_loss"][cluster][mote][-1] * cycle[cluster][mote] / (
                            cycle[cluster][mote] + 1) + features.get("packet_loss") / (
                                                      10 * (cycle[cluster][mote] + 1))
                else:
                    features["packet_loss"] = features.get("packet_loss") / 10

                if True:  # cluster == 1:
                    features["utility"] = features["utility"][1] * 0.2 + features["utility"][2] * 0.8
                else:
                    features["utility"] = sum(features["utility"]) / 3
                for feature in features:
                    data[feature][cluster][mote].append(features[feature])
                data["cycle"][cluster][mote].append(cycle[cluster][mote])
                cycle[cluster][mote] = cycle[cluster][mote] + 1

    return data


def readDataGlobal(resultsfileName):
    data = {"cycle": [dict()], "packet_loss": [dict()], "freshness": [dict()],
            "energy_consumption": [dict()], "utility": [dict()]}
    results_file = open(resultsfileName, "r")
    cycle = [dict(), dict()]
    cluster = 0
    counter = dict()
    for line in results_file.readlines():
        if line == "changed\n":
            before = 1
            counter = dict()
        elif line[:len("recluster: ")] == "recluster: ":
            cluster += 1
            cycle.append(dict())
            for feature in data:
                data[feature].append(dict())
        else:
            read_data = json.loads(line)
            motenumber = list(read_data.keys())[0]
            mote = "global"

            if counter.get(motenumber) is None:
                counter[motenumber] = 0
            counter[motenumber] = counter[motenumber] + 1

            if counter.get(motenumber) % 300 == 0:
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
                for transmission in read_data.get(list(read_data.keys())[0]):
                    if transmission["transmission_power_setting"] != -1000:
                        features["utility"][0] = 1 - math.pow(
                            transmission.get("transmission_power_setting") / 14.0, 2)
                        features["utility"][1] = features["utility"][1] + max(0, (
                                transmission.get("latency") * transmission.get("expiration_time") / 100 - 35) /
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
                features["utility"][1] = 1 - features["utility"][1] / (10 - features["packet_loss"])

                features["utility"][2] = min(max(1.0 - (features.get("packet_loss") / 10 - 0.1) * 4, 0), 1.0)
                if False:  # cycle[cluster][mote] > 0:
                    features["packet_loss"] = data["packet_loss"][cluster][mote][-1] * cycle[cluster][mote] / (
                            cycle[cluster][mote] + 1) + features.get("packet_loss") / (
                                                      10 * (cycle[cluster][mote] + 1))
                else:
                    features["packet_loss"] = features.get("packet_loss") / 10

                if True:  # cluster == 1:
                    features["utility"] = features["utility"][1] * 0.2 + features["utility"][2] * 0.8
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


def heatmap_metric(metrics, metrics_2, plot_names, cycles, metric_name, shape_name, is_global=False, y_range=None,
                   line_colors=None):
    maxlist = list()
    for cluster in range(len(metrics)):
        if len(metrics[cluster]) > 0:
            for sublist in metrics[cluster].values():
                maxlist.append(np.amax(sublist))

    max_val = np.amax(maxlist)

    data_matrix_mean = []
    for cluster in range(len(metrics[0])):
        data_matrix_mean.append(list())
    for cluster in range(1, len(metrics)):
        i = 0
        for mote in metrics[0]:
            if metrics[cluster].get(mote) is not None:
                data_matrix_mean[i].append(np.mean(metrics[cluster][mote]))
            else:
                data_matrix_mean[i].append(None)
            i += 1

    if metrics_2 is not None:
        fig = make_subplots(2, 1)

        data_2_matrix_mean = []
        for cluster in range(len(metrics_2[0])):
            data_2_matrix_mean.append(list())
        for cluster in range(1, len(metrics_2)):
            i = 0
            for mote in metrics_2[0]:
                if metrics_2[cluster].get(mote) is not None:
                    data_2_matrix_mean[i].append(np.mean(metrics_2[cluster][mote]))
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
            z=data_matrix_mean, colorscale=[[0, 'green'], [1, 'red']], hoverongaps=False))
    if y_range is None:
        y_range = [0, max_val + 5.0 * max_val / 100.0]
    fig.update_layout(
        yaxis=dict(
            title='motes',
        ),
        xaxis=dict(
            title='cluster_cycles'
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.7
        ),
        coloraxis={'colorscale': [[0, 'green'], [1, 'red']]}
        # title={
        # 'text': text_title,
        # 'y':.99,
        # 'x':0.5,
        # 'xanchor': 'center',
        # 'yanchor': 'top'}
    )

    #fig.write_image("./figures/" + shape_name + ".pdf")
    fig.write_html("./figures/" + shape_name + "_mean.html")

    data_matrix_max = []
    for cluster in range(len(metrics[0])):
        data_matrix_max.append(list())
    for cluster in range(1, len(metrics)):
        i = 0
        for mote in metrics[0]:
            if metrics[cluster].get(mote) is not None:
                data_matrix_max[i].append(np.max(metrics[cluster][mote]))
            else:
                data_matrix_max[i].append(None)
            i += 1
    fig = go.Figure(data=go.Heatmap(
        z=data_matrix_max, colorscale=[[0, 'green'], [1, 'red']], hoverongaps = False))
    if y_range is None:
        y_range = [0, max_val + 5.0 * max_val / 100.0]
    fig.update_layout(
        yaxis=dict(
            title='motes',
            range=y_range
        ),
        xaxis=dict(
            title='cluster_cycles',
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
    fig.write_html("./figures/" + shape_name + "_max.html")

    data_matrix_min = []
    for cluster in range(len(metrics[0])):
        data_matrix_min.append(list())
    for cluster in range(1, len(metrics)):
        i = 0
        for mote in metrics[0]:
            if (metrics[cluster].get(mote) is not None):
                data_matrix_min[i].append(np.min(metrics[cluster][mote]))
            else:
                data_matrix_min[i].append(None)
            i += 1
    fig = go.Figure(data=go.Heatmap(
        z=data_matrix_min, colorscale=[[0, 'green'], [1, 'red']], hoverongaps = False))

    if y_range is None:
        y_range = [0, max_val + 5.0 * max_val / 100.0]
    fig.update_layout(
        yaxis=dict(
            title='motes',
            range=y_range
        ),
        xaxis=dict(
            title='cluster_cycles',
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
    fig.write_html("./figures/" + shape_name + "_min.html")

    data_matrix_median = []
    for cluster in range(len(metrics[0])):
        data_matrix_median.append(list())
    for cluster in range(1, len(metrics)):
        i = 0
        for mote in metrics[0]:
            if metrics[cluster].get(mote) is not None:
                data_matrix_median[i].append(np.median(metrics[cluster][mote]))
            else:
                data_matrix_median[i].append(None)
            i += 1
    fig = go.Figure(data=go.Heatmap(
        z=data_matrix_median, colorscale=[[0, 'green'], [1, 'red']], hoverongaps=False))
    if y_range is None:
        y_range = [0, max_val + 5.0 * max_val / 100.0]
    fig.update_layout(
        yaxis=dict(
            title='motes',
            range=y_range
        ),
        xaxis=dict(
            title='cluster_cycles'
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.7
        ),

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
        # for i in range(len(metrics[1])):
        #    if line_colors is not None:
        #        fig.add_trace(go.Box(x=cycles[0].get(plot_names[i]), y=metrics[1][i], name=(plot_names[i]),
        # .                             line=dict(color=line_colors[i])), row=1, col=2)
        #   else:
        #        fig.add_trace(go.Box(x=cycles[1].get(plot_names[i]), y=metrics[1][i], name=plot_names[i]), row=1, col=2)

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


def plotAll(data, data_2 = None, name = "trial"):
    feature = list(data.keys())[0]
    colors = px.colors.qualitative.Dark24
    names = list()
    for motenumber in range(len(list(data.get(feature)[0].keys()))):
        names.append("mote " + str(motenumber + 1))

    for feature in data:
        if feature != "cycle":
            plot_metrics(data.get(feature),  names,
                         data.get("cycle"), feature,
                         name + "_goals_" + feature)
            if data_2 is not None:
               heatmap_metric(data.get(feature),data_2.get(feature), names,
                            data.get("cycle"), feature,
                            name + "_goals_" + feature + "_heatmap")
            else:
                heatmap_metric(data.get(feature) ,None, names,
                               data.get("cycle"), feature,
                               name + "_goals_" + feature + "_heatmap")
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
data_switch = readData("results.txt")
data_no_switch = readData("results_no_reclustering.txt")
data_switch_global = readDataGlobal("results.txt")
data_no_switch_global = readDataGlobal("results_no_reclustering.txt")
# [data_three, cycle_three] = readData("results_three_two.txt")

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

# plotAll(data_two, "two")
plotAll(data_switch, name="switch")
plotAll(data_no_switch,data_switch, name = "comparision")
plotAll(data_switch_global, name ="switch_global")
# plotAll(data_three,cycle_three,"three")
# plotAllBox(data_two, "two")
# plotAllBox(data_switch, "switch")
# plotAllBox(data_switch_global, "switch_global", is_global=True)
# plotAllBox(data_switch_global, "switch_global")
# plotAllBox(data_three,cycle_three,"three")
