import pandas as pd
import pandas as pd
import networkx as nx
import itertools
dic_isomorph={0: [],1: [(3, 2)], 
2: [(3, 1), (3, 2)], 
3: [(2, 3), (3, 2)],
4: [(2, 3), (3, 1)], 
5: [(2, 3), (3, 1), (3, 2)], 
6: [(2, 1), (3, 1)], 
7: [(2, 1), (3, 1), (3, 2)], 
8: [(2, 1), (2, 3), (3, 1), (3, 2)], 
9: [(1, 3), (2, 3), (3, 2)], 
10: [(1, 3), (2, 3), (3, 1), (3, 2)], 
11: [(1, 3), (2, 1), (3, 2)], 
12: [(1, 3), (2, 1), (3, 1), (3, 2)],
13: [(1, 3), (2, 1), (2, 3), (3, 1)], 
14: [(1, 3), (2, 1), (2, 3), (3, 1), (3, 2)], 
15: [(1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)]}
####################################################################################################################################################3

def common_neighbors(g, i, j):
    s1=set(g.predecessors(i)).intersection(g.predecessors(j))
    s2=set(g.successors(i)).intersection(g.successors(j))
    s1.update(s2)
    return s1
def isomorph_Calss_four_triads(G,N1,N2,N3):
    global dic_isomorph
    H = G.subgraph([N1,N2,N3])
    #print('N1,N2,N3',N1,N2,N3)
    scoure=0
    for motif_num, v in (dic_isomorph.items()):
        G_isomorph=nx.DiGraph()
        G_isomorph.add_edges_from(v)
        boolean=nx.is_isomorphic(G_isomorph,H)
        if(boolean==True):
            scoure=round(motif_num*(1/16),4)
            #print('motif_num:',motif_num,boolean)
    return(scoure)
    
    

def Measure_MotifsTriad(G,Link):
    CN_MissLinkout = {(e[0],e[1]):(list(common_neighbors(G,e[0],e[1]))) for e in Link}
    Mmofit=[]
    for u in CN_MissLinkout:
        k=0
        lenCN=len(CN_MissLinkout[u])
        if(lenCN!=0):
            for v in CN_MissLinkout[u]:
                k=k+isomorph_Calss_four_triads(G,u[0],u[1],v)
            mearsure=k/lenCN
            Mmofit.append((u[0],u[1],mearsure))
        elif(lenCN==0):
            Mmofit.append((u[0],u[1],0))
    return Mmofit

#g=[(1,2),(2,1),(3,2),(2,3),(4,1),(1,4),(2,4),(1,3),(2,5)]
'''g= [(1, 2), (1, 3), (1, 4), (2, 1), (2, 3), (2, 4), (3, 1),
    (3, 2), (3, 4), (4, 1), (4, 2), (4, 3),(1,5),(2,5)]
G=nx.DiGraph()
G.add_edges_from(g)
Link=[(1,2),(1,3),(4,3)]
print(Measure_Motifs(G,Link))'''
