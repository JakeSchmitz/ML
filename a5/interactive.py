from kmeans import KMeans
import sys
import math

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

def run_interactive(k, cs):
  fname = raw_input('Path to data: ')
  k = raw_input('Enter a value for K: ')
  KM = KMeans(fname)
  cs = KM.initClusters(int(k))
  runprompt = 'Run another iteration? (1 for yes, 0 for no): '
  running = input(runprompt)
  i = 0
  while str(running) is not '0':
    clusters, cs = KM.runIteration(cs, k)
    summed_squares = 0
    for j, c in clusters.iteritems():
      print str(j) + ': ' + str(wcss(c, cs[j], KM.dmetric)) + ' size: ' + str(len(c))
      summed_squares += wcss(c, cs[j])
    print 'Total squared distances from centers: ' + str(summed_squares)
    running = input(runprompt)
  again = input('Run again with a different K? (put 0 to quit): ')
  if str(again) is not '0':
    run(int(again), KM.initClusters(int(again)))
  else:
    exit()

def run_automated():
  fname = sys.argv[1]
  startK, endK = int(sys.argv[2]), int(sys.argv[3])
  trials = int(sys.argv[4])
  KM = KMeans(fname)
  wcss_dict = dict()
  for i in range(startK, endK):
    print 'Running clustering for k = ' + str(i)
    for x in range(trials):
      clus, cen, wcs = KM.autoCluster(i)
      if i not in wcss_dict:
        wcss_dict[i] = wcs
      if wcs < wcss_dict[i]:
        wcss_dict[i] = wcs 
  print wcss_dict

#run_interactive()
run_automated()

