#!/usr/bin/env python3
import os
import networkx as nx
import numpy as np
import re


class Parser:
    """ A parser to load Graph datasets """

    def __init__(self, path):
        """
        :param path: Path to the dataset (e.g. './datasets/NCI1')
        """
        name = os.path.basename(path)
        self.ds_a = os.path.join(path, name + '_A' + '.txt')
        self.ds_gi = os.path.join(path, name + '_graph_indicator' + '.txt')
        self.ds_nl = os.path.join(path, name + '_node_labels' + '.txt')

        self.ds_na = os.path.join(path, name + '_node_attributes' + '.txt')
        self.has_node_atrributes = os.path.exists(self.ds_na)
        self.ds_gl = os.path.join(path, name + '_graph_labels' + '.txt')
        self.has_graph_labels = os.path.exists(self.ds_gl)

    def _read_vertices(self):
        """ Iterator for graphs with initialized vertices and node labels """

        graph_index = 1
        vertex_index = 1
        G = nx.Graph()

        if not self.has_node_atrributes:
            with open(self.ds_gi, "r") as fp_gi, open(self.ds_nl, "r") as fp_nl:
                for gi, nl in zip(fp_gi, fp_nl):
                    if int(gi) > graph_index:
                        graph_index = int(gi)
                        yield G
                        G = nx.Graph()

                    label = int(nl)
                    G.add_node(vertex_index, node_label=label)
                    vertex_index = vertex_index + 1
        else:
            with open(self.ds_gi, "r") as fp_gi, open(self.ds_nl, "r") as fp_nl, open(self.ds_na, "r") as fp_na:
                for gi, nl, na in zip(fp_gi, fp_nl, fp_na):
                    if int(gi) > graph_index:
                        graph_index = int(gi)
                        yield G
                        G = nx.Graph()

                    label = int(nl)
                    attributes = [float(x) for x in re.split(r'[, ]', na) if not x == '']
                    G.add_node(vertex_index, node_label=label, node_attributes=attributes)
                    vertex_index = vertex_index + 1

        yield G

    def _read_labels(self):
        """ Iterator for graph labels """

        with open(self.ds_gl, "r") as fp_gl:
            for gl in fp_gl:
                yield int(gl)

    def _read_graph(self):
        """ Iterator for fully loaded Graphs """

        graphs = self._read_vertices()
        G = next(graphs)

        if self.has_graph_labels:
            labels = self._read_labels()
            label = next(labels)
            G.graph['label'] = label

        with open(self.ds_a, "r") as fp_a:
            for edges in fp_a:
                start, end = [int(e.strip()) for e in edges.split(',')]

                while (start not in G.nodes()) and (end not in G.nodes()):
                    yield G
                    try:
                        G = next(graphs)

                        if self.has_graph_labels:
                            label = next(labels)
                            G.graph['label'] = label
                    except StopIteration:
                        return

                G.add_edge(start, end)
        yield G

    def parse_all_graphs(self, max_size=np.inf):
        """
        Returns all graphs of the dataset as a list of networkx graphs
        :param max_size: Optional upper bound for the graph size. Only graphs with |V| <= max_size are returned.
        :return: The graphs of the dataset
        """

        graphs_generator = self._read_graph()
        graphs = []
        for G in graphs_generator:
            if G.order() <= max_size:
                graphs.append(G)
        return graphs
