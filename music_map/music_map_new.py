import config

import numpy as np
import requests
import json
import statistics
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt
import networkx as nx


def get_artists(user, limit):
    url = f"https://ws.audioscrobbler.com/2.0/?method=library.getartists&api_key={config.API_KEY}&user={user}&limit={limit}&format=json"
    response = requests.get(url)
    response_json = response.json()
    # add status check

    artists = response_json["artists"]["artist"]
    artists_names = map(lambda artist: artist["name"], artists)
    return(artists_names)

def get_tags(artist):
    # def tag_filter(tag):
    #     tags = ["seen live"]
    #     if tag in tags:
    #         return True
    #     else:
    #         return False
    
    url = f"https://ws.audioscrobbler.com/2.0/?method=artist.gettoptags&artist={artist}&api_key={config.API_KEY}&format=json"
    response = requests.get(url)
    tags = response.json()
    if tags.get('error'):
        return {}
    # tags_filtered = filter(tag_filter,tags)
    return(tags)
    
def get_all_tags(artists):
    tags = []
    for artist in artists:
        artist_tags = get_tags(artist)

        if not artist_tags:
            continue

        for artist_tag in artist_tags["toptags"]["tag"]:
            is_tag_in_list = False
            if artist_tag["count"] > 10: #isn't poorly voted
                for tag in tags:
                    if tag["tag"] == artist_tag["name"]:
                        is_tag_in_list = True
                        tag["artists"].add(artist)
                if not is_tag_in_list:
                    tags.append({"tag": artist_tag["name"], "artists":{artist}})
    return tags

def sort_tags(tags, limit):
    def num_artists(tag):
        return len(tag["artists"])

    tags.sort(key=num_artists, reverse=True)
    return tags[:limit]


#   a b c
# a 0 1 4
# b 1 0 5
# c 4 5 0
def calculate_weighted_adjacency_matrix(tags):
    num_vertices = len(tags)
    adj_matrix = [[0] * num_vertices for i in range(num_vertices)]

    for i in range(num_vertices):
        for j in range(i+1,num_vertices):
            artists_intersection= len(tags[i]["artists"] & tags[j]["artists"])
            adj_matrix[i][j] = artists_intersection
            adj_matrix[j][i] = artists_intersection

    return adj_matrix

def cluster_tags(tags, adj_matrix):
    # sim_list = []
    # for i in range(len(adj_matrix)):
    #     for j in range(i+1, len(adj_matrix)):
    #         sim_list.append(adj_matrix[i][j])
    # sim_list = filter(lambda e: e != 0,sim_list)
    # threshold = statistics.median(sim_list)
    threshold = 7
    clusters = []
    num_vertices = len(tags)
    # print(threshold)
    for i in range(num_vertices):
        tag = tags[i]
        group = []
        for j in range(num_vertices):
            if adj_matrix[i][j] > threshold:
                group.append({"tag":tags[j],"weight":adj_matrix[i][j]})
        group.sort(key=lambda e : e["weight"])
        if group:
            clusters.append({"tag":tag,"group":group})
    # clusters = filter(lambda e: e.group, clusters)
    return clusters

def spectral_cluster(adj_matrix, tags, clusters):
    np_adj_matrix = np.array(adj_matrix)
    G = nx.from_numpy_array(np_adj_matrix)

    tags_list = list(map(lambda e: e["tag"],tags))
    node_label_dict = {i: label for i, label in enumerate(tags_list)}
    node_sizes = list(map(lambda e: len(e["artists"]) * 40,tags))

    spectral_clustering = SpectralClustering(n_clusters=clusters, affinity='precomputed', random_state=42)
    labels = spectral_clustering.fit_predict(np_adj_matrix)

    # Draw the graph with clusters
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G)

    # Draw nodes with colors based on the cluster labels
    nx.draw(G, pos, with_labels=False, node_color=labels, cmap=plt.get_cmap('Pastel1'), node_size=node_sizes, edge_color='lightgray')
    nx.draw_networkx_labels(G, pos, labels=node_label_dict, font_size=6)
    plt.title("Spectral Clustering of Graph")
    plt.show()



def main():
    artists = get_artists(config.USERNAME,300)
    tags = get_all_tags(artists)
    tags = sort_tags(tags,100)
    adj_matrix = calculate_weighted_adjacency_matrix(tags)
    spectral_cluster(adj_matrix,tags,5)
main()