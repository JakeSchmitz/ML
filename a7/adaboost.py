# Implementation of AdaBoost for Comp 135 
# Jacob Schmitz

# 
# Overview
# Use the command line to provide the file names of training and 
# test files, as well as L, the number of iterations to train
#
# Training:
# Initialize Probability table
# For i from 1 through L:
#   Draw sample Xi from Probability Distribution
#   Train Learner 1 on sample Xi
#   Calculate Error ei
#   if ei < 1/2:
#     Bi = ei / (1 - ei)
#     Update Probability Distribution (normalize)
#
# Testing:
# For each data point in the test set:
# Calculate .....
#

import arff
import sys
import random
import math

class AdaBoost:
  def __init__(self, trainingfile, testfile, ls):
    self.trdata = arff.load(trainingfile)
    self.tsdata = arff.load(testfile)
    self.steps = int(ls)
    self.betas = dict()
    self.probs = dict()
    self.probs[0] = dict()
    self.samples = dict()
    self.learners = dict()
    self.errors = dict()
    self.traindata = dict()
    self.testdata = dict()
    datasize = 0
    for r in self.trdata:
      self.probs[0][datasize] = 1.0
      self.traindata[datasize] = list(r)
      datasize += 1
    self.trainsize = float(datasize)
    datasize = 0
    for r in self.tsdata:
      self.testdata[datasize] = list(r)
      datasize += 1
    self.testsize = float(datasize)
    i = 0
    for x in self.probs[0]:
      self.probs[0][i] = 1.0 / self.trainsize
      i += 1

  # Given i, a key already set in the probability table
  # Draw a random sample of training data
  def drawSample(self, i=0):
    # Note that a sample is just a list of indexes in the training data 
    # This is to avoid redundant storage
    s = dict()
    for x in range(int(self.trainsize)):
      nr = random.random()
      cut = float(0.0)
      indx = 0
      for p in self.probs[i].values():
        cut += p
        if nr < cut:
          s[x] = indx
          break
        indx += 1
    return s
    
  def trainlearner(self, i):
    s = self.samples[i]
    bestpluserror = 1.0
    bestminuserror = 1.0
    bestpluscutoff = self.traindata[s[0]][i]
    bestminuscutoff = self.traindata[s[0]][i]
    for d in s:
      label = self.traindata[d][-1]
      attr = self.traindata[d][i]
      pluserrors = 0.0
      minuserrors = 0.0
      for x in s:
        xlabel = self.traindata[x][-1] 
        xattr = self.traindata[x][i]
        if xattr < attr:
          if xlabel > 0:
            minuserrors += 1.0
        else:
          if xlabel < 0:
            pluserrors += 1.0
        perror = pluserrors / self.trainsize
        merror = minuserrors / self.trainsize
        print merror
        if merror < bestminuserror:
          bestminuserror = merror
          bestminuscutoff = attr
        if perror < bestpluserror:
          bestpluserror = perror
          bestpluscutoff = attr 
    answer = dict()
    answer["attribute"] = i
    if bestpluserror < bestminuserror:
      answer["error"] = bestpluserror
      answer["cutoff"] = bestpluscutoff
    else:
      answer["error"] = bestminuserror
      answer["cutoff"] = bestminuscutoff
    return answer

  def run(self):
    for i in range(self.steps):
      self.samples[i] = self.drawSample(i)
      print self.trainlearner(i)
      #self.testlearners(lrnr) 

AB = AdaBoost(sys.argv[1], sys.argv[2], sys.argv[3])
print AB.run()

