import networkx as nx
import itertools as IT
import pandas as pd
import random

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


'''count1=3375749028''' 
