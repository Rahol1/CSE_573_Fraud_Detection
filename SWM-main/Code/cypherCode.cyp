 CREATE CONSTRAINT ON (c:Customer) ASSERT c.id IS UNIQUE;
CREATE CONSTRAINT ON (m:Merchant) ASSERT m.id IS UNIQUE;

LOAD CSV WITH HEADERS FROM
"file:///"C:\dataset.csv" AS line
WITH line,
SPLIT(line.customer, "'") AS customerID,
SPLIT(line.merchant, "'") AS merchantID,
SPLIT(line.age, "'") AS customerAge,
SPLIT(line.gender, "'") AS customerGender,
SPLIT(line.category, "'") AS transCategory

MERGE (customer:Customer {id: customerID[1], age: customerAge[1], gender: customerGender[1]})

MERGE (merchant:Merchant {id: merchantID[1]})

CREATE (transaction:Transaction {amount: line.amount, fraud: line.fraud, category: transCategory[1], step: line.step})-[:WITH]->(merchant)
CREATE (customer)-[:PERFORMS]->(transaction);



MATCH (c1:Customer)-[:PERFORMS]->(t1:Transaction)-[:WITH]->(m1:Merchant)
WITH c1, m1
MERGE (p1:LINK {id: m1.id})

MATCH (c1:Customer)-[:PERFORMS]->(t1:Transaction)-[:WITH]->(m1:Merchant)
WITH c1, m1, count(*) as cnt
MERGE (p2:LINK {id:c1.id})

MATCH (c1:Customer)-[:PERFORMS]->(t1:Transaction)-[:WITH]->(m1:Merchant)
WITH c1, m1, count(*) as cnt
MATCH (p1:LINK {id:m1.id})
WITH c1, m1, p1, cnt
MATCH (p2:LINK {id: c1.id})
WITH c1, m1, p1, p2, cnt
CREATE (p2)-[:PAYSTO {cnt: cnt}]->(p1)

MATCH (c1:Customer)-[:PERFORMS]->(t1:Transaction)-[:WITH]->(m1:Merchant)
WITH c1, m1, count(*) as cnt
MATCH (p1:LINK {id:c1.id})
WITH c1, m1, p1, cnt
MATCH (p2:LINK {id: m1.id})
WITH c1, m1, p1, p2, cnt
CREATE (p1)-[:PAYSTO {cnt: cnt}]->(p2)

// degree graph feature computation
CALL gds.alpha.degree.write({
  nodeProjection: 'LINK',
  relationshipProjection: {
    relType: {
      type: 'PAYSTO',
      orientation: 'UNDIRECTED',
      properties: {
        cnt: {
          property: 'cnt',
          defaultValue: 1
        }
      }
    }
  },
  relationshipWeightProperty: 'cnt',
  writeProperty: 'degree'
});

// For viewing database

MATCH (node:`LINK`)
WHERE exists(node.`degree`)
RETURN node, node.`degree` AS score
ORDER BY score DESC;


// PageRank graph feature computation

CALL gds.pageRank.write({
  nodeProjection: 'LINK',
  relationshipProjection: {
    relType: {
      type: 'PAYSTO',
      orientation: 'UNDIRECTED',
      properties: {}
    }
  },
  relationshipWeightProperty: null,
  dampingFactor: 0.85,
  maxIterations: 20,
  writeProperty: 'pagerank'
});

// For viewing PageRank in database

MATCH (node:`LINK`)
WHERE exists(node.`pagerank`)
RETURN node, node.`pagerank` AS score
ORDER BY score DESC;

// Betweenness graph feature computation

CALL gds.betweenness.write({
  nodeProjection: 'LINK',
  relationshipProjection: {
    relType: {
      type: 'PAYSTO',
      orientation: 'UNDIRECTED',
      properties: {}
    }
  },
  writeProperty: 'betweenness'
});

// For viewing betweenness in database
MATCH (node:`LINK`)
WHERE exists(node.`betweenness`)
RETURN node, node.`betweenness` AS score
ORDER BY score DESC;

// Closeness graph features computation

CALL gds.alpha.closeness.write({
  nodeProjection: 'LINK',
  relationshipProjection: {
    relType: {
      type: 'PAYSTO',
      orientation: 'UNDIRECTED',
      properties: {}
    }
  },
  writeProperty: 'closeness'
});

// For viewing in closeness database
MATCH (node:`LINK`)
WHERE exists(node.`closeness`)
RETURN node, node.`closeness` AS score
ORDER BY score DESC;

// louvain graph features computation
CALL gds.louvain.write({
  nodeProjection: 'LINK',
  relationshipProjection: {
    relType: {
      type: 'PAYSTO',
      orientation: 'UNDIRECTED',
      properties: {}
    }
  },
  relationshipWeightProperty: null,
  includeIntermediateCommunities: false,
  seedProperty: '',
  writeProperty: 'louvain'
});

// For viewing louvain in database

MATCH (node:`LINK`)
WHERE exists(node.`louvain`)
WITH node, node.`louvain` AS community
WITH collect(node) AS allNodes,
CASE WHEN apoc.meta.type(community) = "long[]" THEN community[-1] ELSE community END AS community,
CASE WHEN apoc.meta.type(community) = "long[]" THEN community ELSE null END as communities
RETURN community, communities, allNodes AS nodes, size(allNodes) AS size
ORDER BY size DESC;

// community graph features computation

CALL gds.labelPropagation.write({
  nodeProjection: 'LINK',
  relationshipProjection: {
    relType: {
      type: 'PAYSTO',
      orientation: 'UNDIRECTED',
      properties: {
        cnt: {
          property: 'cnt',
          defaultValue: 1
        }
      }
    }
  },
  relationshipWeightProperty: 'cnt',
  writeProperty: 'community'
});

// For viewing louvain in database
MATCH (node:`LINK`)
WHERE exists(node.`community`)
WITH node.`community` AS community, collect(node) AS allNodes
RETURN community, allNodes AS nodes, size(allNodes) AS size
ORDER BY size DESC;

// community graph features computation
CALL gds.wcc.write({
  nodeProjection: 'LINK',
  relationshipProjection: {
    relType: {
      type: 'PAYSTO',
      orientation: 'UNDIRECTED',
      properties: {}
    }
  },
  writeProperty: 'connectedCommunity'
});

//For viewing connectedCommunity in database

MATCH (node:`LINK`)
WHERE exists(node.`connectedCommunity`)
WITH node.`connectedCommunity` AS community, collect(node) AS allNodes
RETURN community, allNodes AS nodes, size(allNodes) AS size
ORDER BY size DESC;

// Community graph features computation

CALL gds.triangleCount.write({
  nodeProjection: 'LINK',
  relationshipProjection: {
    relType: {
      type: 'PAYSTO',
      orientation: 'UNDIRECTED',
      properties: {}
    }
  },
  writeProperty: 'trianglesCount'
});

// For viewing connectedCommunity in database

MATCH (node:`LINK`)
WHERE exists(node.`trianglesCount`)
RETURN node, node.`trianglesCount` AS triangles
ORDER BY triangles DESC;

// node similarity graph features computation
CALL gds.nodeSimilarity.write({
  similarityCutoff: 0.1,
  degreeCutoff: 1,
  nodeProjection: 'LINK',
  relationshipProjection: {
    relType: {
      type: 'PAYSTO',
      orientation: 'NATURAL',
      properties: {}
    }
  },
  writeProperty: 'similarity',
  writeRelationshipType: 'SIMILAR_JACCARD'
});

// For viewing similarity in database

MATCH (from)-[rel:`SIMILAR_JACCARD`]-(to)
WHERE exists(rel.`similarity`)
RETURN from, to, rel.`similarity` AS similarity
ORDER BY similarity DESC;

// node cluster coefficient graph features computation

CALL gds.localClusteringCoefficient.write({
  nodeProjection: 'LINK',
  relationshipProjection: {
    relType: {
      type: 'PAYSTO',
      orientation: 'UNDIRECTED',
      properties: {}
    }
  },
  writeProperty: 'coefficientCluster'
);

//For viewing cluster coefficient in database

MATCH (node:`LINK`)
WHERE exists(node.`coefficientCluster`)
RETURN node, node.`coefficientCluster` AS coefficient
ORDER BY coefficient DESC;


