from utils.dataset_exporter import export_dataset
from tqdm import trange, tqdm
import sys
import os
import numpy as np
import networkx as nx

def make_specific_dataset():
    ''' Creates a dataset of 5200 graphs, with 25 different class labels and no node attributes or node labels.
    The class labels are the number of triangles the graph has, and the graphs are randomly generated. The number of 
    triangles is calculated via a NetworkX function that determines the number of 3-cliques the graph has.
    Each graph has between 5 and 70 nodes, and a fixed edge probability of 20%. This is to prevent extremly long
    calculation times, as we do need a NP-calculation. So we try to keep the amount of edges lower than completly random.
    One can however pick the edge probability p at random aswell.
    '''
    arbitrary_graphs = []
    for _ in trange(1,file=sys.stdout, desc='Outer Loop'): # Calculate 200 graphs per class
        for desired_triangle_number in trange(26, file=sys.stdout, desc='TriangleNumber'): 
            #we have 26 classes: Each graph has between 0 and 25 triangles in it. The loop will
            #desired_triangle_number is the number of triangles we want to produce in our current graph
            
            #Roll a random graph
            n = np.random.choice(np.arange(71)[5:],1)[0] #number of nodes between 5 and 70
            G = nx.fast_gnp_random_graph(n,0.2) #Random graph with n nodes and edge probability 0.2
            
            #Calculate the number of triangles
            list_of_triangles = [x for x in (list(nx.find_cliques(G))) if len(x)==3]
            number_of_triangles = len(list_of_triangles)
            #Set the graph label accordingly to the number of triangles it has
            G.graph['label'] = number_of_triangles

            while(number_of_triangles != desired_triangle_number): 
                #Reroll until we have the desired amount of triangles in our graph
                
                #Attempt to free memory
                del G
                del list_of_triangles
                
                
                n = np.random.choice(np.arange(101)[5:],1)[0] #number of nodes between 5 and 100
                #p = np.random.random_sample() #probability of edges in the random graph can be rolled randomly instead
                
                G = nx.fast_gnp_random_graph(n,0.2)
                list_of_triangles = [x for x in (list(nx.find_cliques(G))) if len(x)==3]
                number_of_triangles = len(list_of_triangles)
                G.graph['label'] = number_of_triangles

            arbitrary_graphs.append(G) #memorize the graph if it has the right amount of triangles
        
    out_dir = r"C:\Users\Dips\Documents\Praktikum\group1\exercise4\datasets"

    #Save dataset
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    outfile = os.path.join(out_dir, 'test')
    export_dataset(outfile, arbitrary_graphs, has_g_labels=True, has_n_labels=False, has_n_attributes=False)
make_specific_dataset()
def make_clique_number_dataset():

    min_clique = 1
    max_clique = 40
    number_cliques = max_clique - min_clique + 1
    graphs = [[] for _ in range(number_cliques)]

    max_nodes_per_graph = 60
    number_graphs_per_clique = 50
    total_number_graphs = number_cliques * number_graphs_per_clique

    found_all_graphs = False
    no_graphs_found = 0

    t = tqdm(desc="Number of graphs found", total=total_number_graphs)
    while not found_all_graphs:
        n = np.random.randint(5, max_nodes_per_graph)
        m = np.random.randint(0, int(n * (n - 1) * 0.5))
        G = nx.gnm_random_graph(n, m)

        clique_number = nx.graph_clique_number(G)

        if clique_number <= max_clique and len(graphs[clique_number - min_clique]) <= number_graphs_per_clique:
            graphs[clique_number - min_clique].append(G)
            no_graphs_found += 1
            t.update(1)

            if no_graphs_found == total_number_graphs:
                found_all_graphs = True

    out_dir = "datasets/"

    # Make output directory if neccessary and export dataset
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    outfile = os.path.join(out_dir, 'clique_number_dataset')
    out_graphs = []
    for i, graphlist in enumerate(graphs):
        for graph in graphlist:
            # assign label to each graph
            graph.graph['label'] = i + min_clique
            out_graphs.append(graph)
    export_dataset(outfile, out_graphs, has_g_labels=True, has_n_labels=False, has_n_attributes=False)

