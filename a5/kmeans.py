import arff
import random
import math

# Assumes x and y are same dimensional arrays
def dist(x, y):
  d = float(0)
  for i, v in enumerate(list(x)):
    d += (float(y[i]) - float(v)) ** 2
  return math.sqrt(d)

def wcss(cluster, center, distance=dist):
  d = 0
  for r in cluster:
    thisdist = distance(r, center) ** 2
    d += thisdist
  return d

class KMeans:
  def __init__(self, datafile, distance=dist):
    self.rawdata = arff.load(datafile)
    self.data = dict()
    self.dmetric = distance
    i = 0
    for row in self.rawdata:
      self.data[i] = list(row)
      i += 1
    self.datasize = i

  # If we want to run kmeans until centers don't change, use autocluster
  def autoCluster(self, k=2):
    centers = self.initClusters(k)
    centerdiff = 1000
    clusters = dict()
    while centerdiff > 1:
      centerdiff = 0
      clusters, newcenters = self.runIteration(centers, k)
      for i, c in newcenters.iteritems():
        centerdiff += self.dmetric(c, centers[i])
      centers = newcenters
    summed_squares = 0
    for j, c in clusters.iteritems():
      summed_squares += wcss(c, centers[j])
    # return 2 dictionaries, clusters maps from cluster num to list of pts in cluster
    # oldcenters is the last computed set of centers from the current clusters dict
    return clusters, centers, summed_squares

  def initClusters(self, k):
    centers = dict()
    for i in range(k):
      centers[i] = self.data[random.randint(0,self.datasize- 1)]
    return centers

  def runIteration(self, centers, k):
    clusters = dict()
    # for each record in the input data
    for k,v in self.data.iteritems():
      # find the closest center
      closest = self.closestCenter(v, centers) 
      if closest not in clusters:
        clusters[closest] = []
      clusters[closest].append(v) 
    # After adding each piece of data to a cluster, make sure none are empty
    if len(clusters.keys()) < len(centers.keys()):
      for i in centers.keys():
        if i not in clusters:
          still_empty = True
          for j in clusters.keys():
            if still_empty and len(clusters[j]) > 1:
              clusters[i] = [clusters[j].pop()]
              still_empty = False
    newcenters = self.clusterCenters(clusters)
    return clusters, newcenters

  # Creates a dict of cluster_id -> cluster_center
  def clusterCenters(self, clusters):
    centers = dict()
    # for each cluster find the center
    for i, cluster in clusters.iteritems():
      v = dict()
      S = float(len(cluster))
      # for each entry in this cluster, iterate over columns
      for row in cluster:
        for j, c in enumerate(row):
          if j not in v:
            v[j] = 0
          v[j] += float(c)
      for j, val in v.iteritems():
        v[j] = float(val / S)
      centers[i] = v.values()
    return centers      
      
  def closestCenter(self, record, centers):
    distance = dict()
    minD = None
    for i, center in centers.iteritems():
      distance[i] = self.dmetric(record, center)
      if minD is None:
        minD = i
      if distance[i] <= distance[minD]:
        minD = i
    return minD

