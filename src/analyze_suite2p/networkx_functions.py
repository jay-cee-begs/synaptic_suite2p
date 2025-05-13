import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from analyze_suite2p import suite2p_utility as transform
from analyze_suite2p.config_loader import load_json_config_file

configurations = load_json_config_file()



import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
import multiprocessing

def load_for_networkx(data_folder):  ## creates a dictionary for the suite2p paths in the given data folder (e.g.: folder for well_x)
    """
    Creates a dictionary for networkx analysis from the SUITE2P_STRUCTURE in the data folder.
    """    
    stat = transform.load_npy_array(os.path.join(data_folder, *transform.SUITE2P_STRUCTURE["stat"]))
    F = transform.load_npy_array(os.path.join(data_folder, *transform.SUITE2P_STRUCTURE["F"]))
    Fneu = transform.load_npy_array(os.path.join(data_folder, *transform.SUITE2P_STRUCTURE["Fneu"]))
    iscell = transform.load_npy_array(os.path.join(data_folder, *transform.SUITE2P_STRUCTURE['iscell']))[:,0].astype(bool)
    neuron_data = {}

    for idx, neuron_stat in enumerate(stat):
        x_median = np.median(neuron_stat['xpix'])
        y_median = np.median(neuron_stat['ypix'])
        # deltaF = cascade_predictions[idx,:]

        neuron_data[f"neuron_{idx}"] = {
            "x": x_median,
            "y": y_median,
            "IsUsed": iscell[idx]
        }
    filtered_neuron_data = {}

    for key, value in neuron_data.items():
        if value["IsUsed"]:
            filtered_neuron_data[key] = value
    
    return  filtered_neuron_data

def create_template_matrix(neuron_data):
    num_neurons = len(neuron_data)
    temp_matrix = np.random.rand(num_neurons, num_neurons)
    temp_matrix[temp_matrix<0.9] = 0
    np.fill_diagonal(temp_matrix, 0)
    G = nx.from_numpy_array(temp_matrix)
    mapping = {i: neuron_id for i, neuron_id in enumerate(neuron_data.keys())} #rename index from neuron_0... for networkx
    G = nx.relabel_nodes(G, mapping)
    return G

def extract_and_plot_neuron_connections(node_graph, neuron_data, data_folder, sample_name):
    
    for neuron_id, data in neuron_data.items():
        node_graph.add_node(neuron_id, pos=(data['x'], data['y'])) 
    pos = nx.get_node_attributes(node_graph, 'pos')
    
    #Community Detection
    neuron_clubs = list(greedy_modularity_communities(node_graph))
    community_map = {
        node:community_idx
        for community_idx, community in enumerate(neuron_clubs)
        for node in community
    }
    #Node statistics
    node_degree_dict = dict(node_graph.degree)
    clustering_coeff_dict = nx.clustering(node_graph)
    betweenness_centrality_dict = nx.betweenness_centrality(node_graph)
    try:
        eigenvector_centrality_dict = nx.eigenvector_centrality(node_graph)
    except nx.PowerIterationFailedConvergence:
        eigenvector_centrality_dict = {node: None for node in node_graph.nodes}

    #Edge Statistics
    edge_data =[]
    for (u,v,data) in node_graph.edges(data=True):
        edge_data.append({
            'source': u,
            "target": v,
            'weight': data.get("weight", 1),
        })

    community_sizes = {community_idx: len(community) for community_idx, community in enumerate(neuron_clubs)}
    raw_data = []
    for node, neuron in zip(node_graph.nodes, neuron_data):
        raw_data.append({
            "neuron_id":node,
            "x": neuron_data[node]["x"],
            "y": neuron_data[node]["y"],
            "community": community_map[node],
            "community_size":community_sizes[community_map[node]],
            "degree": node_degree_dict[node],
            "clustering_coefficient": clustering_coeff_dict[node],
            "betweenness_centrality": betweenness_centrality_dict[node],
            "eigenvector_centrality": eigenvector_centrality_dict[node],
            "total_predicted_spikes": np.nansum(neuron_data[neuron]['predicted_spikes']),
            "avg_predicted_spikes": np.nanmean(neuron_data[neuron]['predicted_spikes'])
        })
    df_nodes = pd.DataFrame(raw_data)
    df_nodes.to_csv(os.path.join(data_folder, f"{sample_name}_graph_node_data.csv"), index=False)

    df_edges = pd.DataFrame(edge_data)
    df_edges.to_csv(os.path.join(data_folder, f"{sample_name}_graph_edge_data.csv"), index = False)
    
    community_colors = [community_map[node] for node in node_graph.nodes]
    unique_clubs = len(set(community_colors))
    plt.figure(figsize=(20,20))
    ax = plt.gca()
    nx.draw(
        node_graph,
        pos=pos,
        with_labels=False,
        node_size=250,
        node_color=community_colors,
        cmap=plt.cm.tab10,  # Use a colormap with distinct colors
    )
    ax.set_title(f"Community Detection with {unique_clubs} Communities (Corrected Positions)", fontsize = 24)
    ax.set_xlabel(f"Sample: {sample_name}", fontsize = 18)
    plt.savefig(os.path.join(data_folder, f"{sample_name}_networkx_connections.png"))
    plt.close()

    
def plot_neuron_connections(data_folder):
    print('extracting neuron data for network x')
    neuron_data = load_for_networkx(data_folder)
    print("creating networkx node graph")
    node_graph = create_template_matrix(neuron_data)
    sample_name = os.path.basename(data_folder)
    print(sample_name)
    extract_and_plot_neuron_connections(node_graph, neuron_data, data_folder, sample_name)

def main():
    for sample in transform.get_file_name_list(configurations.main_folder, file_ending = 'samples', supress_printing=False):
        print(f"Processing {sample}")
        plot_neuron_connections(sample)
        print('Finished processing')

if __name__ == '__main__':
    main()
