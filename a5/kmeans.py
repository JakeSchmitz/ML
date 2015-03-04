import arff
import random
import math

class kmeans:
  def __init__(self, datafile, distance):
    self.rawdata = arff.load(datafile)
    self.data = dict()
    self.dmetric = distance
    i = 0
    for row in self.rawdata:
      self.data[i] = list(row)
      i += 1
    self.datasize = i

  def kclusters(self, k=2):
    oldcenters = self.initClusters(k)
    centerdiff = 1000
    while centerdiff > 1:
      centerdiff = 0
      newcenters = self.runIteration(oldcenters)
      for i, c in newcenters.iteritems():
        centerdiff += self.dmetric(c, oldcenters[i])
      oldcenters = newcenters
      print 'Just finished another iteration'
    print 'Finished clustering'

  def initClusters(self, k):
    centers = dict()
    for i in range(k):
      centers[i] = self.data[random.randint(0,self.datasize)]
    return centers

  def runIteration(self, centers):
    clusters = dict()
    for k,v in self.data.iteritems():
      closest = self.closestmean(v, centers)
      if closest not in clusters:
        clusters[closest] = []
      clusters[closest].append(v) 
    newcenters = self.clusterCenters(clusters)
    return self.clusterCenters(clusters)

  def clusterCenters(self, clusters):
    centers = dict()
    for i, cluster in clusters.iteritems():
      v = dict()
      S = len(cluster)
      for row in cluster:
        for j, c in enumerate(row[:-1]):
          if j not in v:
            v[j] = 0
          v[j] += float(c)/S
      centers[i] = list(v.values())
    return centers      
      
  def closestmean(self, record, centers):
    distance = dict()
    for i, center in centers.iteritems():
      distance[i] = self.dmetric(record, center)
    minD = 0
    for i, dist in distance.iteritems():
      if distance[minD] > distance[i]: 
        minD = i
    return minD
    
# Assumes x and y are same dimensional arrays
def dist(x, y):
  d = 0
  for i, v in enumerate(list(x)[:-1]):
    d += (float(y[i]) - float(v)) * (float(y[i]) - float(v))
  return math.sqrt(d)

km = kmeans('hw4train.arff', dist)
km.kclusters(4)
