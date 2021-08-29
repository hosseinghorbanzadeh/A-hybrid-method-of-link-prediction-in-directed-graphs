def Dic_AUC(dic_CN_MissLink,dic_CN_nonexiteEdges):
    n=0
    m=0
    
    for i in dic_CN_MissLink:
        for j in dic_CN_nonexiteEdges:
            if dic_CN_MissLink[i]>dic_CN_nonexiteEdges[j]:
                n=n+1
                #print('n',n)
            if dic_CN_MissLink[i]==dic_CN_nonexiteEdges[j]:
                m=m+1
                #print('m',m)
    print('n,m:',n,m)
    auc1=(n+0.5*m)/(len(dic_CN_MissLink)*len(dic_CN_nonexiteEdges))
    auc=round(auc1,4)
    return auc

