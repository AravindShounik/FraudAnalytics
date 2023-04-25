"""
To run: python3 trustRank.py
"""

import numpy as np
import pandas as pd
from pregel import Vertex,Pregel
from numpy import mat


num_workers = 4                                 # No of threads to assign vertices

""" Vertex class for TrustRank algorithm
    Can add instance varibles and methods according to algorithm
"""
class TrustRankVertex(Vertex):
    def __init__(self,id, value, out_vertices, mode=0, dampingFactor=0.85,iterations=50):
        Vertex.__init__(self,id,value,out_vertices)
        self.dampingFactor = dampingFactor
        self.num_supersteps = iterations
        self.mode = mode

    def update(self):
        # This routine has a bug when there are pages with no outgoing
        # links (never the case for our tests).  This problem can be
        # solved by introducing Aggregators into the Pregel framework,
        # but as an initial demonstration this works fine.

        #print(f"{self.id} Vertex Superstep : {self.superstep}" )

        # mode 0 - for calculating inverse page rank
        if self.superstep < self.num_supersteps:
            messages_sum = 0
            outmessage_sum  = 0
            for (vertex,message) in self.incoming_messages:
                messages_sum = messages_sum+message
            for vertex in self.out_vertices:
                if not isinstance(vertex,TrustRankVertex):
                    print(f"\n===== Not Trust Rank - {str(vertex)} +++++ Type = {type(vertex)} ==========\n")
                #print(type(vertex))
                outmessage_sum = outmessage_sum+adj_mtx[self.id][vertex.id]
            self.value = (1-self.dampingFactor) * d[self.id] + self.dampingFactor*messages_sum
            outgoing_message = self.value / len(self.out_vertices)
            self.outgoing_messages = [(vertex,self.value*(adj_mtx[self.id][vertex.id]/outmessage_sum)) for vertex in self.out_vertices]
        else:
            self.active = False

def pregelTrustRank(vertices):
    pregel = Pregel(vertices, num_workers)
    pregel.run()
    return mat([vertex.value for vertex in pregel.vertices]).transpose()

def intialPageRank(nodes):
    return [0.0 for n in nodes]

if __name__ == "__main__":
    # Defining variables
    # read csv data file
    dataDF = pd.read_csv("PregelClass/Iron_dealers_data.csv")
    # find unique number of buyer and seller ids
    num_vertices = pd.unique(dataDF[['Buyer ID', 'Seller ID']].values.ravel('K')).size

    # read bad nodes ID
    badDF = pd.read_csv("PregelClass/bad.csv")
    bad_nodes = []
    for i in range(len(badDF)):
        bad_nodes.append(badDF.loc[i,'Bad Id'])

    # dictionary of counter -> pregel nodes
    pregelNodes = {}
    # counter variable to map buyer and seller id starting from 0
    counter = 0
    # dictionary to store buyer/seller id -> counter
    nodeMap = {}
    # list to store pregel nodes
    nodes = []

    zero_deg = []

    adj_mtx = [ [0]*num_vertices for i in range(num_vertices) ]

    damping_factor = 0.85

    for i in range(len(dataDF)):
        
        seller_id = dataDF.loc[i,'Seller ID']
        if seller_id not in nodeMap:
            p = TrustRankVertex(counter, 1, [], damping_factor)
            pregelNodes[counter] = p
            nodes.append(p)
            nodeMap[seller_id] = counter
            counter +=1
        else:
            p = pregelNodes[nodeMap[seller_id]]
        
        buyer_id = dataDF.loc[i,'Buyer ID']
        if buyer_id not in nodeMap:
            q = TrustRankVertex(counter, 1, [], 0, damping_factor)
            pregelNodes[counter] = q
            nodes.append(q)
            nodeMap[buyer_id] = counter
            counter +=1 
        else:
            q = pregelNodes[nodeMap[buyer_id]]

        edge_weight = dataDF.loc[i,'Value']
        # vertex_weight_list = [q, edge_weight]
        # append q along with the edge weight to p's out vertices
        # p.out_vertices.append(vertex_weight_list)
        p.out_vertices.append(q)
        adj_mtx[p.id][q.id] = edge_weight

    # store all the zero degree vertices
    for node in nodes:
        if len(node.out_vertices) == 0:
            zero_deg.append(node.id)

    # map bad nodes id to counter id
    for i in range(len(bad_nodes)):
        bad_nodes[i] = nodeMap[bad_nodes[i]]
    
    for zi in zero_deg:
        for bn in bad_nodes:
            bad_node = pregelNodes[bn]
            if bad_node not in pregelNodes[zi].out_vertices:
                pregelNodes[zi].out_vertices.append(bad_node)
                adj_mtx[zi][bn] = 1
    
    count = 0
    d = intialPageRank(nodes)
    for s in bad_nodes:
        count = count + 1
        d[s] = 1
    for i in range(len(d)):
        d[i] = d[i]/count
    
    node_ranks = pregelTrustRank(nodes)

    # tr = TrustRank(damping_factor, max_iters, maxer, L, nodes_new, outlinks, inlinks)
    # trust_scores = tr.trustRank()

    print("\nTrust scores of all the nodes are as follows: ")
    for i in node_ranks:
        print(i, end=" ")
    print("\n\n")