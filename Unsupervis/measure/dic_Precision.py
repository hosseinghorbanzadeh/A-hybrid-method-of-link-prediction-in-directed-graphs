import random
from itertools import islice
import operator
def Merge_two_dict(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z

def Take(n, iterable):
    "Return first n items of the iterable as a list"
    return dict(islice(iterable, n))

def Find(key, dictionary):
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

def dic_Measure_Precision(dic_CN_MissLink,dic_CN_nonexiteEdges,MissLink,L):
    Marge_Dic=Merge_two_dict(dic_CN_MissLink,dic_CN_nonexiteEdges)
    sorted_dic = sorted(Marge_Dic.items(),key=operator.itemgetter(1),reverse=True)
    tempDic=Take(L,sorted_dic)
    k=0
    for i in range(len(MissLink)):
        L1=(list(Find(MissLink[i], tempDic)))
        if L1!=[]:
            k=k+1
    return(round((k/L),4))

'''d1={(1,2):3,(2,3):5}
d2={(5,4):3,(6,7):8}
L=1
list1=[(6,7),(2,3)]
print(dic_Measure_Precision(d1,d2,list1,2))'''
