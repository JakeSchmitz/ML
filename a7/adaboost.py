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

  def testCutoff(self, attr, cutoff, sample):
    pluserror = 0.0
    minuserror = 0.0
    pluserrors = 0.0
    minuserrors = 0.0
    for x in sample:
      xlabel = self.traindata[x][-1] 
      xattr = self.traindata[x][attr]
      if xattr < cutoff:
        # If x < cutoff and was labeled positive, thats an error in the + direction
        if int(xlabel) > 0:
          pluserrors += 1.0
        # If x < cutoff and was labeled negative, thats an error in the - direction
        else:
          minuserrors += 1.0
      # If x >= cutoff
      else:
        # label should be positive in + direction, negative in - direction
        if int(xlabel) > 0:
          minuserrors += 1.0
        # If it's negative that means the direction would be - (or it's an error)
        else:
          pluserrors += 1.0
    pluserror = pluserrors / self.trainsize
    minuserror = minuserrors / self.trainsize
    return pluserror, minuserror
      
  def trainLearner(self, step, i):
    s = self.samples[step]
    bestpluserror = 1.0
    bestminuserror = 1.0
    bestpluscutoff = self.traindata[s[0]][i]
    bestminuscutoff = self.traindata[s[0]][i]
    for d in s:
      label = self.traindata[d][-1]
      cut = self.traindata[d][i]
      pluserror, minuserror = self.testCutoff(i, cut, s)
      if minuserror < bestminuserror:
        bestminuserror = minuserror
        bestminuscutoff = cut
      if pluserror < bestpluserror:
        bestpluserror = pluserror
        bestpluscutoff = cut 
    answer = dict()
    answer["attribute"] = i
    if bestpluserror < bestminuserror:
      answer["error"] = bestpluserror
      answer["cutoff"] = bestpluscutoff
      answer["direction"] = 1
    else:
      answer["error"] = bestminuserror
      answer["cutoff"] = bestminuscutoff
      answer["direction"] = -1
    return answer

  # Given a current step (learner being trained)
  # Create the next probability distribution based on the success
  # of the current learner
  def updateProbs(self, step):
    newstep = step + 1
    learner = self.learners[step]
    self.probs[newstep] = dict()
    probsum = 0.0
    cut = learner["cutoff"]
    attr = learner["attribute"]
    direction = float(learner["direction"])
    # create new probability distribution for learner
    for i in range(int(self.trainsize)):
      oldprob = self.probs[step][i]
      newprob = oldprob
      v = self.traindata[i][attr]
      if float(direction * (v - cut)) <  0.0:
        newprob -= self.betas[step] * oldprob
      self.probs[newstep][i] = newprob
      probsum += newprob
    # normalize the new distribution
    for i in range(int(self.trainsize)):
      self.probs[newstep][i] /= probsum
      sumprob += self.probs[newstep][i]

  def run(self):
    self.trainLearners()
    print self.learners

  # Trains as many learners as asked for via command line
  # Resulting learners are in self.learners
  # which is a hash from iteration -> learner 
  # where only iterations with e < 1/2 are used
  def trainLearners(self):
    for i in range(self.steps):
      self.samples[i] = self.drawSample(i)
      #print self.trainLearner(i)
      besterror = 1
      bestlearner = None
      for j in range(len(self.traindata[0]) - 1):
        lrn = self.trainLearner(i, j)
        if lrn["error"] < besterror:
          bestlearner = lrn
          besterror = lrn["error"]
      print bestlearner
      # This learner is better than guessing, so save it
      if besterror < 0.5:
        b = besterror / (1.0 - besterror)
        self.betas[i] = b
        self.learners[i] = bestlearner
        self.updateProbs(i)
      #self.testlearners(lrnr) 

AB = AdaBoost(sys.argv[1], sys.argv[2], sys.argv[3])
print AB.run()

