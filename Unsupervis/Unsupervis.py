# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 17:03:56 2021

@author: hossein
"""
import networkx as nx
import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import itertools as IT
import random
from measure.AADirect import *
from measure.CN_Hub_Auth import *
from measure.RADir import *
from measure.triads_motif_measure import *
from measure.n2vLP import *
from measure.AucMeasure import Auc
from measure.AucMeasure_InputDic import *
from measure.Precision import Measure_Precision
from measure.dic_Precision import dic_Measure_Precision


#=======================
#======================

def RandomNumber(Low,Hight,Num):
    L=[]
    i=0
    while i<Num:
        r=random.randint(Low,Hight)
        if(r not in L):
            L.append(r)
        i+=1
    return L

def common_out_neighbors(g, i, j):
    return set(g.successors(i)).intersection(g.successors(j))

def common_in_neighbors(g, i, j):
    return set(g.predecessors(i)).intersection(g.predecessors(j))
#=============================Retuen_Miss_Link========================================================
def Retuen_Miss_Link(WholeGraph,alpha):
    TraningGraph=WholeGraph.copy()
    edges=WholeGraph.edges()
    Number=int(len(edges)*alpha)
    sample=iterSample(edges,Number)
    TraningGraph.remove_edges_from(sample)
    set1=set(WholeGraph.nodes())
    set2=set(TraningGraph.nodes())
    set3=set1.difference(set2)
    list1=list(set3)
    if list1!=[]:
        for i in range(len(list1)):
            TraningGraph.add_node(list1[i])
    G2=nx.difference(WholeGraph,TraningGraph)
    missLink=list(G2.edges())
    return missLink,WholeGraph,TraningGraph
#========================================================================================================
def Weight_sample(WholeGraph,Nnmber_Negative):
    Nonedges=iterSample(nx.non_edges(WholeGraph),Nnmber_Negative)
    return Nonedges

#===========================================================================================

    
def iterSample(iterable, samplesize):
    results = []

    for i, v in enumerate(iterable):
        r = random.randint(0, i)
        if r < samplesize:
            if i < samplesize:
                results.insert(r, v) # add first samplesize items in random order
            else:
                results[r] = v # at a decreasing rate, replace random items

    if len(results) < samplesize:
        raise ValueError("Sample larger than population.")

    return results
#Input non_edges That is iterable Output list

#===========================================================================================

def Final_Result(WholeGraph_DF,nameDataSet):
    MissEdges,WholeGraph,TraningGraph = Retuen_Miss_Link(WholeGraph_DF,0.3)
    L=100
    Miss_Scoure=[]
    Non_Scoure=[]
    resultAUC_Dic={}
    ResultPER_Dic={}
    CE=len(WholeGraph.edges())-1
    if(nameDataSet=='Wiki-Vote'):
        NE=50000000-1
        print('Wiki-Vote')
    else:
        NE=len(list(nx.non_edges(WholeGraph)))-1
    Nnmber_Negative=(CE/(CE+NE))*NE
    #Number_Positive=(NE/(CE+NE))*CE
    NonexistentEdges=iterSample(nx.non_edges(WholeGraph),Nnmber_Negative)
    print('ok')
    Miss_Scoure=n2vlp(TraningGraph,MissEdges)
    Non_Scoure=n2vlp(TraningGraph,NonexistentEdges)
    auc=Auc(Miss_Scoure,Non_Scoure)
    resultAUC_Dic.update({'n2v':auc})
    Precision=Measure_Precision(Miss_Scoure,Non_Scoure,L)
    ResultPER_Dic.update({'n2v':Precision})
    #########################CNHA########################################
    dic_miss,dic_NonExistent=CN_HA_AH(TraningGraph,MissEdges,NonexistentEdges)
    print('cnha')
    auc=Dic_AUC(dic_miss,dic_NonExistent)
    Precision=dic_Measure_Precision(dic_miss,dic_NonExistent,MissEdges,L)
    print('auc',auc)
    resultAUC_Dic.update({'SCNHA':auc})
    ResultPER_Dic.update({'SCNHA':Precision})

    dic_miss,dic_NonExistent=CN_H_A(TraningGraph,MissEdges,NonexistentEdges)
    Precision=dic_Measure_Precision(dic_miss,dic_NonExistent,MissEdges,L)
    auc=Dic_AUC(dic_miss,dic_NonExistent)
    resultAUC_Dic.update({'CN_H_A':auc})
    ResultPER_Dic.update({'CN_H_A':Precision})

    dic_miss,dic_NonExistent=CN_A_H(TraningGraph,MissEdges,NonexistentEdges)
    Precision=dic_Measure_Precision(dic_miss,dic_NonExistent,MissEdges,L)
    auc=Dic_AUC(dic_miss,dic_NonExistent)
    resultAUC_Dic.update({'CN_A_H':auc})
    ResultPER_Dic.update({'CN_A_H':Precision})
    
    #########################CNHA########################################
    #########################AA##########################################
    SmissLinkAADir=list(adamic_adar_index(TraningGraph,MissEdges,TyppeGraph='DirGhraphIn-Out'))
    NonExistentAAdir=list(adamic_adar_index(TraningGraph,NonexistentEdges,TyppeGraph='DirGhraphIn-Out'))
    auc=Auc(SmissLinkAADir,NonExistentAAdir)
    print('auc',auc)
    resultAUC_Dic.update({'AA-IN-Out':auc})
    Precision=Measure_Precision(SmissLinkAADir,NonExistentAAdir,L)
    ResultPER_Dic.update({'AA-IN-Out':Precision})

    SmissLinkAADir=list(adamic_adar_index(TraningGraph,MissEdges,TyppeGraph='DirGhraphIn'))
    NonExistentAAdir=list(adamic_adar_index(TraningGraph,NonexistentEdges,TyppeGraph='DirGhraphIn'))
    auc=Auc(SmissLinkAADir,NonExistentAAdir)
    resultAUC_Dic.update({'AA-IN':auc})
    Precision=Measure_Precision(SmissLinkAADir,NonExistentAAdir,L)
    ResultPER_Dic.update({'AA-IN':Precision})


    SmissLinkAADir=list(adamic_adar_index(TraningGraph,MissEdges,TyppeGraph='DirGhraphOut'))
    NonExistentAAdir=list(adamic_adar_index(TraningGraph,NonexistentEdges,TyppeGraph='DirGhraphOut'))
    auc=Auc(SmissLinkAADir,NonExistentAAdir)
    resultAUC_Dic.update({'AA-OUT':auc})
    Precision=Measure_Precision(SmissLinkAADir,NonExistentAAdir,L)
    ResultPER_Dic.update({'AA-OUT':Precision})
    #########################AA##########################################
    #########################Common nightboar###############################################
    print('lin pridiction Using Common nightboar')
    CN_MissLink = [(e[0],e[1],len(list(common_out_neighbors(TraningGraph,e[0],e[1])))+len(list(common_in_neighbors(TraningGraph,e[0],e[1])))) for e in MissEdges]
    CN_nonexiteEdges=[(e[0],e[1],len(list(common_out_neighbors(TraningGraph,e[0],e[1])))+len(list(common_in_neighbors(TraningGraph,e[0],e[1])))) for e in NonexistentEdges]
    auc=Auc(CN_MissLink,CN_nonexiteEdges)
    resultAUC_Dic.update({'CN-i-o':auc})
    Precision=Measure_Precision(CN_MissLink,CN_nonexiteEdges,L)
    ResultPER_Dic.update({'CN-i-o':Precision})


    CN_MissLink = [(e[0],e[1],len(list(common_out_neighbors(TraningGraph,e[0],e[1])))) for e in MissEdges]
    CN_nonexiteEdges=[(e[0],e[1],len(list(common_out_neighbors(TraningGraph,e[0],e[1])))) for e in NonexistentEdges]
    Precision=Measure_Precision(CN_MissLink,CN_nonexiteEdges,L)
    auc=Auc(CN_MissLink,CN_nonexiteEdges)
    resultAUC_Dic.update({'CN-o':auc})
    ResultPER_Dic.update({'CN-o':Precision})


    CN_MissLink = [(e[0],e[1],len(list(common_in_neighbors(TraningGraph,e[0],e[1])))) for e in MissEdges]
    CN_nonexiteEdges=[(e[0],e[1],len(list(common_in_neighbors(TraningGraph,e[0],e[1])))) for e in NonexistentEdges]
    Precision=Measure_Precision(CN_MissLink,CN_nonexiteEdges,L)
    auc=Auc(CN_MissLink,CN_nonexiteEdges)
    resultAUC_Dic.update({'CN-i':auc})
    ResultPER_Dic.update({'CN-i':Precision})

    #########################Common nightboar###############################################
    ############################## Resource Allocation##############################
    print('Resource Allocation')
    SmissLinkAADir=list(resource_allocation_index(TraningGraph,MissEdges,TyppeGraph='DirGhraphIn-Out'))
    NonExistentAAdir=list(resource_allocation_index(TraningGraph,NonexistentEdges,TyppeGraph='DirGhraphIn-Out'))
    Precision=Measure_Precision(SmissLinkAADir,NonExistentAAdir,L)
    auc=Auc(SmissLinkAADir,NonExistentAAdir)
    resultAUC_Dic.update({'RA-In-Out':auc})
    ResultPER_Dic.update({'RA-In-Out':Precision})
 

    print('Resource Allocation in')
    SmissLinkAADir=list(resource_allocation_index(TraningGraph,MissEdges,TyppeGraph='DirGhraphIn'))
    NonExistentAAdir=list(resource_allocation_index(TraningGraph,NonexistentEdges,TyppeGraph='DirGhraphIn'))
    Precision=Measure_Precision(SmissLinkAADir,NonExistentAAdir,L)
    auc=Auc(SmissLinkAADir,NonExistentAAdir)
    resultAUC_Dic.update({'RA-In':auc})
    ResultPER_Dic.update({'RA-In':Precision})
    


    print('Resource Allocation out')
    SmissLinkAADir=list(resource_allocation_index(TraningGraph,MissEdges,TyppeGraph='DirGhraphOut'))
    NonExistentAAdir=list(resource_allocation_index(TraningGraph,NonexistentEdges,TyppeGraph='DirGhraphOut'))
    Precision=Measure_Precision(SmissLinkAADir,NonExistentAAdir,L)
    auc=Auc(SmissLinkAADir,NonExistentAAdir)
    resultAUC_Dic.update({'RA-Out':auc})
    ResultPER_Dic.update({'RA-Out':Precision})
    ############################## Resource Allocation##############################
    ############################################MotifsTriad################################
    Mofit_miss=Measure_MotifsTriad(TraningGraph,MissEdges)
    Mofit_NonExistent=Measure_MotifsTriad(TraningGraph,NonexistentEdges)
    auc=Auc(Mofit_miss,Mofit_NonExistent)
    resultAUC_Dic.update({'MotifsTriad':auc})

    Precision=Measure_Precision(Mofit_miss,Mofit_NonExistent,L)
    ResultPER_Dic.update({'MotifsTriad':Precision})
    
    return resultAUC_Dic,ResultPER_Dic








