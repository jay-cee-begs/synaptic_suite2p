import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statannotations.Annotator import Annotator


def plot_with_stats(data, metric, plot_type="violin", groups = None):
    """
    Plots the specified metric for all groups in the 'Group' column and adds statistical annotations.

    Parameters:
    - data: pandas DataFrame containing the data.
    - metric: str, the column name of the metric to plot.
    - plot_type: str, the type of plot (e.g., "violin", "box", "swarm", "bar", "point").
    """
    # Dynamically fetch all groups from the 'Group' column
    if groups is None:
        groups = data["Experimental_Group"].unique().tolist()
    else:
        groups = groups
    # Filter the data for the specified groups (in case you want to look at only a few of the groups)
    filtered_data = data[data["Experimental_Group"].isin(groups)]

    # Create the plot
    plt.figure(figsize=(10, 6))
    ax = None

    if plot_type == "violin":
        ax = sns.violinplot(data=filtered_data, x="Experimental_Group", y=metric, inner="quartile", palette="muted")
    elif plot_type == "box":
        ax = sns.boxplot(data=filtered_data, x="Experimental_Group", y=metric, palette="muted")
    elif plot_type == "swarm":
        ax = sns.swarmplot(data=filtered_data, x="Experimental_Group", y=metric, palette="muted", dodge=True)
    elif plot_type == "bar":
        ax = sns.barplot(data=filtered_data, x="Experimental_Group", y=metric, ci="sd", palette="muted")
    elif plot_type == "point":
        ax = sns.pointplot(data=filtered_data, x="Experimental_Group", y=metric, ci="sd", palette="muted", dodge=True)

    # Define pairwise comparisons dynamically from the groups
    pairs = [(groups[i], groups[j]) for i in range(len(groups)) for j in range(i + 1, len(groups))]

    # Add annotations
    annotator = Annotator(ax, pairs, data=filtered_data, x="Experimental_Group", y=metric)
    annotator.configure(test="Kruskal", text_format="star", loc="inside", verbose=2)
    annotator.apply_and_annotate()

    # Customize and show the plot
    plt.title(f"{metric.replace('_', ' ')} ({plot_type.capitalize()} Plot)", fontsize=14)
    plt.ylabel(metric.replace('_', ' '), fontsize=12)
    plt.xlabel("Experimental_Group", fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def load_experiment_csv(file_path):

    data = pd.read_csv(file_path)

    # Metrics and grouping
    metrics = [
        "SpikesCount",
        "SpikesFreq",
        "AvgAmplitude",
        "AvgDecayTime",
        "total_ROIs"
    ]
    groups = data["Experimental_Group"].unique().tolist()

    return groups, metrics, data

def main():
    file_path = input("Please enter .csv file path: \n")
    graph_type = input("Please choose the type of graph to plot\n(e.g. violin, box, swarm,bar, point)\n")
    accepted_types = ["violin","box","swarm","bar","point"]
    if graph_type not in accepted_types:
        print("Please choose a valid graph type")
        graph_type = input("Please choose the type of graph to plot\n(e.g. violin, box, swarm,bar, point)\n")

    groups, metrics, data = load_experiment_csv(file_path)
    for metric in metrics:
        plot_with_stats(data, metric = metric, plot_type=graph_type) 

if __name__ == "__main__":
    main()