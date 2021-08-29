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
def Final_Result_Multi_Dataset1(WholeGraph,TraningGraph,L,Non,miss):
    ResultAUC_Dic={}
    ResultPER_Dic={}
    '''if(bigGraph=='no'):
        miss,Non,G1,G2=MissLink_and_NonExistentLink(WholeGraph_DF.values,TraningGraph_DF.values)
    elif(bigGraph=='yes'):
        miss,Non,G1,G2=MissLink_and_NonExistentLink_for_big_Graph(WholeGraph_DF.values,TraningGraph_DF.values,num_nonEdges)
     '''   
    G1=TraningGraph
    print('miss',len(miss))
    ######################################Number Negative#############################
    CE=len(G1.edges())-1
    NE=len(Non)-1
    Nnmber_Negative=(CE/(CE+NE))*NE
    Number_Positive=(NE/(CE+NE))*CE
    list1=RandomNumber(1,NE,Nnmber_Negative)
    NonExistent=[]
    for w in list1:
        NonExistent.append(Non[w])
    print('CE',CE)    
    print('NonExistent',len(NonExistent))
    print('miss',len(miss))
    ##############################################################
    Class=[]
    for i in range(len(miss)):
        Class.append(1)
    for j in range(len(NonExistent)):
        Class.append(0)

    ################################adamic_adar_index#######################################
    print('#####################AADricted Graph####################################')
    SmissLinkAADir=list(adamic_adar_index(G1,miss,TyppeGraph='DirGhraphIn-Out'))
    NonExistentAAdir=list(adamic_adar_index(G1,NonExistent,TyppeGraph='DirGhraphIn-Out'))
    data1=pd.DataFrame(SmissLinkAADir+NonExistentAAdir,columns=['Node1','Node2','AA-in-out'])

    SmissLinkAADir=list(adamic_adar_index(G1,miss,TyppeGraph='DirGhraphIn'))
    NonExistentAAdir=list(adamic_adar_index(G1,NonExistent,TyppeGraph='DirGhraphIn'))
    mix=SmissLinkAADir+NonExistentAAdir
    CN=[k[2] for k in mix]
    data1['AA-in']=CN

    SmissLinkAADir=list(adamic_adar_index(G1,miss,TyppeGraph='DirGhraphOut'))
    NonExistentAAdir=list(adamic_adar_index(G1,NonExistent,TyppeGraph='DirGhraphOut'))
    mix=SmissLinkAADir+NonExistentAAdir
    CN=[k[2] for k in mix]
    data1['AA-Out']=CN
    
    print('#######################################')
    ################################adamic_adar_index#######################################
    #########################Common nightboar###############################################
    print('lin pridiction Using Common nightboar')
    CN_MissLink = [(e[0],e[1],len(list(common_out_neighbors(G1,e[0],e[1])))+len(list(common_in_neighbors(G1,e[0],e[1])))) for e in miss]
    print('common_out_neighbors')
    CN_nonexiteEdges=[(e[0],e[1],len(list(common_out_neighbors(G1,e[0],e[1])))+len(list(common_in_neighbors(G1,e[0],e[1])))) for e in NonExistent]
    mix=CN_MissLink+CN_nonexiteEdges
    CN=[k[2] for k in mix]
    data1['CN-inout']=CN

    CN_MissLink = [(e[0],e[1],len(list(common_out_neighbors(G1,e[0],e[1])))) for e in miss]
    CN_nonexiteEdges=[(e[0],e[1],len(list(common_out_neighbors(G1,e[0],e[1])))) for e in NonExistent]
    mix=CN_MissLink+CN_nonexiteEdges
    CN=[k[2] for k in mix]
    data1['CND-out']=CN

    CN_MissLink = [(e[0],e[1],len(list(common_in_neighbors(G1,e[0],e[1])))) for e in miss]
    CN_nonexiteEdges=[(e[0],e[1],len(list(common_in_neighbors(G1,e[0],e[1])))) for e in NonExistent]
    mix=CN_MissLink+CN_nonexiteEdges
    CN=[k[2] for k in mix]
    data1['CND-in']=CN
     #########################Common nightboar###############################################
    ############################## Resource Allocation##############################
    print('Resource Allocation')
    SmissLinkAADir=list(resource_allocation_index(G1,miss,TyppeGraph='DirGhraphIn-Out'))
    print('Len SmissLinkAADir',len(SmissLinkAADir))
    NonExistentAAdir=list(resource_allocation_index(G1,NonExistent,TyppeGraph='DirGhraphIn-Out'))
    mix=SmissLinkAADir+NonExistentAAdir
    #print('RA',len(mix))
    commRA=[k[2] for k in mix]
    data1['RA-in-out']=commRA

    print('Resource Allocation in')
    SmissLinkAADir=list(resource_allocation_index(G1,miss,TyppeGraph='DirGhraphIn'))
    NonExistentAAdir=list(resource_allocation_index(G1,NonExistent,TyppeGraph='DirGhraphIn'))
    mix=SmissLinkAADir+NonExistentAAdir
    commRA=[k[2] for k in mix]
    data1['RA-in']=commRA

    print('Resource Allocation out')
    SmissLinkAADir=list(resource_allocation_index(G1,miss,TyppeGraph='DirGhraphOut'))
    NonExistentAAdir=list(resource_allocation_index(G1,NonExistent,TyppeGraph='DirGhraphOut'))
    mix=SmissLinkAADir+NonExistentAAdir
    commRA=[k[2] for k in mix]
    data1['RA-Out']=commRA

    ##############################AucAndPre_co_ni_With_Pageranking#########################
    dic_miss,dic_NonExistent=CN_HA_AH(G1,miss,NonExistent)
    mix=merge_two_dicts(dic_miss,dic_NonExistent)
    #print('CN_HA_AH',len(mix))
    CN_Hun_Auth1=[mix.get(p) for p in mix]
    data1['CN_HA_AH']=CN_Hun_Auth1

    dic_miss,dic_NonExistent=CN_H_A(G1,miss,NonExistent)
    mix=merge_two_dicts(dic_miss,dic_NonExistent)
    CN_Hun_Auth1=[mix.get(p) for p in mix]
    data1['CN_H_A']=CN_Hun_Auth1

    dic_miss,dic_NonExistent=CN_A_H(G1,miss,NonExistent)
    mix=merge_two_dicts(dic_miss,dic_NonExistent)
    CN_Hun_Auth1=[mix.get(p) for p in mix]
    data1['CN_A_H']=CN_Hun_Auth1

    ##############################AucAndPre_co_ni_With_Pageranking#########################
    ###########################quadr_motif_measure#########################################
    print('################################mofit#########################')
    Mofit_miss=Measure_MotifsTriad(G1,miss)
    print('#####################')
    #print('Len Miss',len(miss))
    #print('Len Mofit_miss',len(Mofit_miss))
    Mofit_NonExistent=Measure_MotifsTriad(G1,NonExistent)
    mix=Mofit_miss+Mofit_NonExistent
    print(len(mix))
    CN_MOfit=[k[2] for k in mix]
    data1['Motifs_triads']=CN_MOfit
    print('################################mofit#########################')
    ##############################AucAndPre_co_ni_With_Pageranking#########################
    ###########################n2v#########################################
    print('#############################N2V##############################')
    Miss_Scoure=n2vlp(G1,miss)
    Non_Scoure=n2vlp(G1,NonExistent)

    mix=Miss_Scoure+Non_Scoure
    print(len(mix))
    CN_n2v=[k[2] for k in mix]
    data1['N2V']=CN_n2v
    
    
    ###########################n2v#########################################
    data1['Class']=Class
    strName="MLExcel"+" "+L+".xlsx"
    print(strName)
    print(data1)
    '''data1.to_excel(strName,
           sheet_name='MLExcel')'''
    return data1





