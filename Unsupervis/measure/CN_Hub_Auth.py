import pandas as pd
import numpy
from operator import itemgetter, attrgetter
import networkx as nx
import matplotlib.pyplot as plt
import xlsxwriter
import operator
from itertools import islice
def common_out_neighbors(g, i, j):
    return set(g.successors(i)).intersection(g.successors(j))

def common_in_neighbors(g, i, j):
    return set(g.predecessors(i)).intersection(g.predecessors(j))

def CN_HA_AH(G1,missLink,nonexiteEdges):
    print('CN_HA_AH')
    Hub_WholeGraph,Auth_WholeGraph=nx.hits(G1,max_iter=100,normalized=False)
    #################################################CN_MissLink#############################################
    CN_MissLinkout = {(e[0],e[1]):(list(common_out_neighbors(G1,e[0],e[1]))) for e in missLink}
    CN_MissLinkin = {(e[0],e[1]):(list(common_in_neighbors(G1,e[0],e[1]))) for e in missLink}
    dic_CN_MissLink=common_neighbors_PageRank_LP_HA_AH(CN_MissLinkout,CN_MissLinkin,Hub_WholeGraph,Auth_WholeGraph)
    ################################################CN_MissLink###############################################
    
    ####################################CN_nonexiteEdges####################################
    CN_nonexiteEdgesout={(e[0],e[1]):(list(common_out_neighbors(G1,e[0],e[1]))) for e in nonexiteEdges}
    CN_nonexiteEdgesin={(e[0],e[1]):(list(common_in_neighbors(G1,e[0],e[1]))) for e in nonexiteEdges}
    dic_CN_nonexiteEdges=common_neighbors_PageRank_LP_HA_AH(CN_nonexiteEdgesout,CN_nonexiteEdgesin,Hub_WholeGraph,Auth_WholeGraph)
    
    return(dic_CN_MissLink,dic_CN_nonexiteEdges)

def CN_A_H(G1,missLink,nonexiteEdges):
    print('CN_A_H')
    Hub_WholeGraph,Auth_WholeGraph=nx.hits(G1,max_iter=100,normalized=False)
    CN_MissLink = {(e[0],e[1]):(list(common_in_neighbors(G1,e[0],e[1]))) for e in missLink}
    dic_CN_MissLink=common_neighbors_PageRank_LP_AH(CN_MissLink,Hub_WholeGraph,Auth_WholeGraph)
    CN_nonexiteEdges={(e[0],e[1]):(list(common_in_neighbors(G1,e[0],e[1]))) for e in nonexiteEdges}
    dic_CN_nonexiteEdges=common_neighbors_PageRank_LP_AH(CN_nonexiteEdges,Hub_WholeGraph,Auth_WholeGraph)
    return(dic_CN_MissLink,dic_CN_nonexiteEdges)

def CN_H_A(G1,missLink,nonexiteEdges):
    print('CN_H_A')
    Hub_WholeGraph,Auth_WholeGraph=nx.hits(G1,max_iter=100,normalized=False)
    #################################################CN_MissLink#############################################
    CN_MissLink = {(e[0],e[1]):(list(common_out_neighbors(G1,e[0],e[1]))) for e in missLink}
    dic_CN_MissLink=common_neighbors_PageRank_LP_HA(CN_MissLink,Hub_WholeGraph,Auth_WholeGraph)
    CN_nonexiteEdges={(e[0],e[1]):(list(common_out_neighbors(G1,e[0],e[1]))) for e in nonexiteEdges}
    dic_CN_nonexiteEdges=common_neighbors_PageRank_LP_HA(CN_nonexiteEdges,Hub_WholeGraph,Auth_WholeGraph)
    
    return(dic_CN_MissLink,dic_CN_nonexiteEdges)



def common_neighbors_PageRank_LP_HA_AH(CNout,CNin,Hub,Auth):
    dic={}
    for u in CNout:
        k=0
        commonNighboor=(len(CNout[u]))
        for v in CNout[u]:
            k=k+Auth.get(v)
        k=k+(Hub.get(u[0])+Hub.get(u[1]))
        dic.update({u:k})
    ############################################################
    for u in CNin:
        k1=0
        commonNighboor=(len(CNin[u]))
        for v in CNin[u]:
            k1=k1+Hub.get(v)
        k1=k1+(Auth.get(u[0])+Auth.get(u[1]))
        dic.update({u:dic.get(u)+k1})
    return dic

def common_neighbors_PageRank_LP_HA(CN,Hub,Auth):
    dic={}
    for u in CN:
        k=0
        commonNighboor=(len(CN[u]))
        for v in CN[u]:
            k=k+Auth.get(v)
        k=k+(Hub.get(u[0])+Hub.get(u[1]))
        dic.update({u:k})
    return dic

def common_neighbors_PageRank_LP_AH(CN,Hub,Auth):
    dic={}
    for u in CN:
        k=0
        commonNighboor=(len(CN[u]))
        for v in CN[u]:
            k=k+Hub.get(v)
        k=k+(Auth.get(u[0])+Auth.get(u[1]))
        dic.update({u:k})
    return dic


def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z
    
    
def take(n, iterable):
    "Return first n items of the iterable as a list"
    return dict(islice(iterable, n))

           

def find(key, dictionary):
    for k, v in dictionary.items():
        if k == key:
            yield v
        elif isinstance(v, dict):
            for result in find(key, v):
                yield result
        elif isinstance(v, list):
            for d in v:
                for result in find(key, d):
                    yield result



