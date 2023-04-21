"""
To run: python3 trustRank.py
"""

import numpy as np
import pandas as pd
from pregel import Vertex,Pregel


num_workers = 4                                 # No of threads to assign vertices

""" Vertex class for TrustRank algorithm
    Can add instance varibles and methods according to algorithm
"""
class TrustRankVertex(Vertex):
    def __init__(self,id,value,out_vertices, in_vertices, dampingFactor=0.85,iterations=50):
        Vertex.__init__(self,id,value,out_vertices)
        self.dampingFactor = dampingFactor
        self.num_supersteps = iterations
        self.in_vertices = in_vertices

    def update(self):
        # This routine has a bug when there are pages with no outgoing
        # links (never the case for our tests).  This problem can be
        # solved by introducing Aggregators into the Pregel framework,
        # but as an initial demonstration this works fine.

        #print(f"{self.id} Vertex Superstep : {self.superstep}" )
        if self.superstep < self.num_supersteps:
            messages_sum = 0
            for (vertex,message) in self.incoming_messages:
                messages_sum = messages_sum+message
            self.value = (1-self.dampingFactor) / maxer + self.dampingFactor*messages_sum
            outgoing_message = self.value / len(self.out_vertices)
            self.outgoing_messages = [(vertex,outgoing_message) for vertex in self.out_vertices]
        else:
            self.active = False

class TrustRank:
    def __init__(self, alpha, max_iters, maxer, L, nodes, outlinks, inlinks):
        self.alpha = alpha
        self.max_iters = max_iters
        self.maxer = maxer
        self.L = L
        self.nodes = nodes
        self.outlinks = outlinks
        self.inlinks = inlinks

    def getTrustedPages(self, sorted_pr_scores):
        trusted_pages = []
        num_trusted_pages = 0

        for i, j in sorted_pr_scores.items():
            if num_trusted_pages < self.L:
                trusted_pages.append(i)
                num_trusted_pages += 1
            else:
                break

        return trusted_pages

    def inversePageRank(self, U, dMatrix):
        # INVERSE PAGERANK ALGORITHM

        a = (1 - self.alpha) * dMatrix
        b = self.alpha * U
        count = 0
        s = dMatrix
        while count < self.max_iters:
            s = np.dot(b, s) + a
            count += 1

        # Generating corresponding ordering
        pr_score = {}
        for i in range(len(s)):
            pr_score[i] = s[i]

        sorted_pr_scores = {
            k: v
            for k, v in sorted(pr_score.items(), key=lambda item: item[1], reverse=True)
        }
        return sorted_pr_scores

    def trustRank(self):
        # Computing Transition Matrix
        T = np.zeros((self.maxer, self.maxer))
        for fromNode, lst_toNodes in self.outlinks.items():
            num_outlinks = len(lst_toNodes)
            frac = 1 / num_outlinks
            for toNode in lst_toNodes:
                T[toNode][fromNode] = frac

        # Computing Inverse Transition Matrix
        U = np.zeros((self.maxer, self.maxer))
        for toNode, lst_fromNodes in self.inlinks.items():
            num_inlinks = len(lst_fromNodes)
            frac = 1 / num_inlinks
            for fromNode in lst_fromNodes:
                U[fromNode][toNode] = frac

        # Computing static score distribution vector
        dMatrix = np.zeros(self.maxer)
        num_nodes = len(self.nodes)
        frac = 1 / num_nodes
        for i in range(1, self.maxer):
            if i in self.nodes:
                dMatrix[i] = frac

        self.pregelTrustRank(self.nodes)

        # finding inverse pageRank scores -> pregel
        sorted_pr_scores = self.inversePageRank(U, dMatrix)

        # Fetching trusted pages from sorted page rank scores
        trusted_pages = self.getTrustedPages(sorted_pr_scores)

        # Computing static score distribution vector
        dMatrix = np.zeros(self.maxer)
        c = 0
        for i in range(1, self.maxer):
            if i in trusted_pages and i not in bad_nodes:
                dMatrix[i] = 1
                c += 1

        # Normalising the d matrix
        if c != 0:
            frac = 1 / c
            for i in range(1, self.maxer):
                if dMatrix[i] == 1:
                    dMatrix[i] = frac

        # Computing TrustRank Scores -> pregel
        a = (1 - self.alpha) * dMatrix
        b = self.alpha * T
        count = 0
        res = dMatrix
        while count < self.max_iters:
            res = np.dot(b, res) + a
            count += 1

        return res

    def pregelTrustRank(vertices):
        pregel = Pregel(vertices, num_workers)
        pregel.run()

if __name__ == "__main__":
    # Defining variables
    nodes = []  # list to store all the nodes
    nodes_dict = {}  # a dictionary where key is a node, and value is a list of outlinks
    nodes_dict1 = {}  # a dictionary where key is a node, and value is a list of inlinks
    maxer = 0
    max_iters = 20  # Number of biased PageRank iterations
    L = 3  # Size of Seed set/Limit of oracle invocations
    
    # data = np.load("iron_dealers_data.csv", delimiter=",")
    datadf = pd.read_csv("PregelClass/Iron_dealers_data.csv")
    datadf.drop("Value", axis=1)
    maxer = pd.unique(datadf[['Buyer ID', 'Seller ID']].values.ravel('K')).size
    
    # num_good_nodes = int(input("Enter number of good nodes: "))
    # good_nodes = []
    # for i in range(0, num_good_nodes):
    #     ele = int(input())
    #     good_nodes.append(ele)

    #bad_nodes = np.load("bad.csv", delimiter=",")
    badDf = pd.read_csv("PregelClass/bad.csv")
    bad_nodes = []
    for i in range(len(badDf)):
        bad_nodes.append(badDf.loc[i,'Bad Id'])
    
    # dictionary of counter -> pregel vertices
    pregelNodes = {}
    # counter variable to map buyer and seller id starting from 0
    counter = 0
    # dictionary to store buyer/seller id -> counter
    nodeMap = {}

    initial_value = maxer-len(bad_nodes)

    damping_factor = 0.85
    alpha = damping_factor

    for i in range(len(datadf)):
        id = datadf.loc[i,'Buyer ID']
        if id not in nodeMap:
            if id not in bad_nodes:
                p = TrustRankVertex(counter, 1/initial_value, [], [], damping_factor)
            else:
                p = TrustRankVertex(counter, 0, [], [], damping_factor)
            pregelNodes[counter] = p
            nodes.append(p)
            nodeMap[id] = counter
            counter +=1 

        id = datadf.loc[i,'Seller ID']
        if id not in nodeMap:
            if id not in bad_nodes:
                q = TrustRankVertex(counter, 1/initial_value, [], [], damping_factor)
            else:
                q = TrustRankVertex(counter, 0, [], [], damping_factor)
            pregelNodes[counter] = q
            nodes.append(q)
            nodeMap[id] = counter
            counter +=1

        p.out_vertices.append(q) 
        q.in_vertices.append(p)
    

    # create the dictionary of outlinks and inlinks
    outlinks = {}
    inlinks = {}
    for node in nodes:
        outlinks[node.id] = node.out_vertices
        inlinks[node.id] = node.in_vertices
    
    # map bad nodes id to counter id
    for i in range(len(bad_nodes)):
        bad_nodes[i] = nodeMap[bad_nodes[i]]

    tr = TrustRank(damping_factor, max_iters, maxer, L, nodes, outlinks, inlinks)
    trust_scores = tr.trustRank()

    print("\nTrust scores of all the nodes are as follows: ")
    for i in trust_scores:
        print(i, end=" ")
    print("\n\n")