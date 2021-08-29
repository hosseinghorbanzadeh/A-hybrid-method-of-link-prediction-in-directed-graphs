def Auc(SmissLink,Snonexits):
    n=0
    m=0
    for i in range(len(SmissLink)):
        for j in range(len(Snonexits)):
            if SmissLink[i][2]>Snonexits[j][2]:
                n=n+1
            if SmissLink[i][2]==Snonexits[j][2]:
                m=m+1
    auc1=(n+0.5*m)/(len(SmissLink)*len(Snonexits))
    auc=round(auc1,4)
    return auc
