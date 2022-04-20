Fraud detection using graph database
# Data
Consist of Dataset <br>
BankSim Dataset : bs140513_032310.csv <br>
Under Sampled Data : dataset.csv <br>
Graph based Data : newDataset2.csv <br>

# Code
under-sampling.py - Generates under sampled file dataset.csv  <br>
<br>
GraphDatasetCreation.ipynb - Generates graph features from neo4j to newDataset2.csv <br>

cypherCode.cyp - Commands for pushing the data into neo4j tool.

Features Extracted : merchDegree ,custDegree , merchCloseness , custCloseness , custPageRank , merchPageRank , custBetweeness ,merchBetweeness ,merchlouvain , custlouvain ,merchCommunity ,custCommunit <br>

baseline_model.ipynb - Trains  raw features(dataset.csv ) on RandomForestClassifier, classify_with_kmeans  and computes feature importance. <br>

graph_model.ipynb - rains  graph features(newDataset2.csv) on RandomForestClassifier, classify_with_kmeans  and computes feature importance.


# Evaluations 

This folder contains the code for UI for the project. 

Link for UI : https://swm-visualization-g26.netlify.app



