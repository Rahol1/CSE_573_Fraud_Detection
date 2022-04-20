# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 14:12:41 2022

@author: pkandala
"""

from py2neo import Graph
import pandas as pd
graphConnection = Graph(password="pkandala", bolt_port=11003, http_port=11004)

graphQuery = """
MATCH (l:LINK)
RETURN l.id AS id,
l.degree AS degree,
l.closeness AS closeness,
l.pagerank AS pagerank, 
l.betweenness AS betweeness,
l.louvain AS louvain,
l.community AS community,
l.coefficientCluster AS clusterCommunity,
l.connectedCommunity AS connectedCommunity,
l.similarity AS similarity
"""
graphData = graphConnection.run(graphQuery)
graphDict = {}
for data in graphData:
    graphDict[data['id']] = {'degree': data['degree'],'closeness': data['closeness'], 'pagerank': data['pagerank'],'betweeness': data['betweeness'], 'louvain': data['louvain'],'community': data['community'],'clusterCommunity': data['clusterCommunity'],'connectedCommunity': data['connectedCommunity'],'similarity': data['similarity']}
    
def add_degree(x):
    return graphDict[x.split("'")[1]]['degree']
def add_closeness(x):
    return graphDict[x.split("'")[1]]['closeness']
def add_pagerank(x):
    return graphDict[x.split("'")[1]]['pagerank']
def add_betweeness(x):
    return graphDict[x.split("'")[1]]['betweeness']
def add_louvain(x):
    return str(graphDict[x.split("'")[1]]['louvain']) 
def add_community(x):
    return str(graphDict[x.split("'")[1]]['community']) 
def add_clusterCommunity(x):
    return str(graphDict[x.split("'")[1]]['clusterCommunity']) 
def add_connectedCommunity(x):
    return graphDict[x.split("'")[1]]['connectedCommunity']
def add_similarity(x):
    return graphDict[x.split("'")[1]]['similarity']

df = pd.read_csv("Dataset.csv")

df = df.sample(frac=1).reset_index(drop=True)
graphDF = df.drop('Unnamed: 0', axis = 1)

graphDF['merchDegree'] = graphDF.merchant.apply(add_degree)
graphDF['custDegree'] = graphDF.customer.apply(add_degree)
graphDF['merchCloseness'] = graphDF.merchant.apply(add_closeness)
graphDF['custCloseness'] = graphDF.customer.apply(add_closeness)
graphDF['custPageRank'] = graphDF.customer.apply(add_pagerank)
graphDF['merchPageRank'] = graphDF.merchant.apply(add_pagerank)
graphDF['custBetweeness'] = graphDF.customer.apply(add_betweeness)
graphDF['merchBetweeness'] = graphDF.merchant.apply(add_betweeness)
graphDF['merchlouvain'] = graphDF.merchant.apply(add_louvain)
graphDF['custlouvain'] = graphDF.customer.apply(add_louvain)
graphDF['merchCommunity'] = graphDF.merchant.apply(add_community)
graphDF['custCommunity'] = graphDF.customer.apply(add_community)

graphDF.to_csv("graphDataSet.csv")