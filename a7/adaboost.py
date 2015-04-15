# Implementation of AdaBoost for Comp 135 
# Jacob Schmitz

#
# To run: python adaboost.py trainfile.arff testfile.arff L
# Where L is the number of iterations to run
#

# I'll be honest, I can't get good results with this and I have
# no idea why. I can get accuracies in the range of 65-77%
# but there's no strong correlation between the number of 
# classifiers and the accuracy. 

# For L = 1 I get a classifier on attribute 2 (or 1 if 0 indx)
# with a cutoff of 144.0 (or similar depending on distribution)
# and that yields an accuracy of 75.78%, which I've only beaten 
# that with L = 5 by random luck and got accuracy = 77.8% 

#
# Read Input
# Train:
# Initialize Probability table
# For i from 1 through L:
#   Draw sample Xi from Probability Distribution
#   Train Learner 1 on sample Xi by checking each attribute
#     For an optimal cutoff value
#   Calculate Error ei
#   if ei < 1/2:
#     Bi = ei / (1 - ei)
#     Update Probability Distribution (normalize)
# Test:
# For each data point in the test set:
# Calculate weighted average over all learners 
# Take the sign to classify the point
# Compute error rate/ accuracy of predictions
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
    self.attrdir = dict()
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
    for i, x in self.probs[0].iteritems():
      self.probs[0][i] = 1.0 / self.trainsize

  # Given i, a key already set in the probability table
  # Draw a random sample of training data
  def drawSample(self, i):
    # Note that a sample is just a list of indexes in the training data 
    # This is to avoid redundant storage
    s = dict()
    for x in range(int(self.trainsize)):
      nr = random.random()
      cut = 0.0
      unset = True
      sett = False
      for indx, p in self.probs[i].iteritems():
        cut += p
        if nr <= cut and unset:
          s[x] = indx
          break
    self.samples[i] = s

  def testCutoff(self, step, attr, cutoff, sample):
    pluserrors = 0.0
    minuserrors = 0.0
    for ind, x in sample.iteritems():
      xlabel = self.traindata[x][-1] 
      xattr = self.traindata[x][attr]
      prob = self.probs[step][x]
      if xattr < cutoff:
        # If x < cutoff and was labeled positive, thats an error in the + direction
        if int(xlabel) > 0:
          pluserrors += prob
        # If x < cutoff and was labeled negative, thats an error in the - direction
        else:
          minuserrors += prob
      # If x >= cutoff
      elif xattr > cutoff:
        # label should be positive in + direction, negative in - direction
        if int(xlabel) > 0:
          minuserrors += prob
        # If it's negative that means the direction would be - (or it's an error)
        else:
          pluserrors += prob
      # Don't count it as +/- if it is the cutoff, distribute error evenly
      else:
        pluserrors += prob / 2
        minuserrors += prob / 2
    return pluserrors, minuserrors
      
  # Given a sample and an attribute index find the best cutoff value and direction
  # for that attribute and calculate the error over the sample of the learner
  def trainLearner(self, step, i, s):
    # Sample should've been generated in trainLearners
    bestpluserror = 1.0
    bestminuserror = 1.0
    bestpluscutoff = self.traindata[s[0]][i]
    bestminuscutoff = self.traindata[s[0]][i]
    # For every data point in the sample
    # Use the values of the attribute as potential cutoff value
    for indx, d in s.iteritems():
      # Get potential cut
      cut = self.traindata[d][i]
      # Calculate the error of using this cut on this sample 
      # pluserror is the error if the direction is +, minuserror is for the reverse
      pluserror, minuserror = self.testCutoff(step, i, cut, s)
      if minuserror < bestminuserror:
        bestminuserror = minuserror
        bestminuscutoff = cut
      if pluserror < bestpluserror:
        bestpluserror = pluserror
        bestpluscutoff = cut 
    answer = dict()
    answer["attribute"] = i
    # Use the best error we saw with every potential cutoff
    if bestpluserror < bestminuserror:
      answer["error"] = bestpluserror
      answer["cutoff"] = bestpluscutoff
      answer["direction"] = 1.0
    else:
      answer["error"] = bestminuserror
      answer["cutoff"] = bestminuscutoff
      answer["direction"] = -1.0
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
    b = self.betas[step]
    # create new probability distribution for learner
    for i, x in self.traindata.iteritems():
      oldprob = self.probs[step][i]
      newprob = oldprob
      v = x[attr]
      if direction * (v - cut) >  0.0:
        newprob = b * oldprob
      self.probs[newstep][i] = newprob
      probsum += newprob
    # normalize the new distribution
    for i in range(int(self.trainsize)):
      self.probs[newstep][i] = self.probs[newstep][i] / probsum
    #print self.probs[newstep]

  # Here is the "main" of adaboost
  # Pretty simple, train the learners, print them, then test
  def run(self):
    self.trainLearners()
    self.printLearners()
    self.testLearners()

  # Testing the learners involves taking all of the test 
  # cases and doing a weighted average of the weak learners
  # applied to the test point
  # take the sign of the sum to get the predicted classification
  def testLearners(self):
    errors = 0
    for i, x in self.testdata.iteritems():
      expected = 0.0
      for step, lrn in self.learners.iteritems():
        weight = math.log(1.0/self.betas[step], 2)
        djx = -1.0
        a = lrn["attribute"]
        diff = float(x[a]) - float(lrn["cutoff"]) 
        if diff * float(lrn["direction"]) > 0:
          djx = 1.0
        expected += weight * djx
      exp = math.copysign(1, expected)
      if int(exp) != int(x[-1]):
        errors += 1
    print str(errors) + ' out of ' + str(self.testsize) + ' test cases were misclassified'
    print 'Accuracy = ' + str(1.0 - float(float(errors)/float(self.testsize)))

  # Trains as many learners as asked for via command line
  # Resulting learners are in self.learners
  # which is a hash from iteration -> learner 
  # where only iterations with e < 1/2 are used
  def trainLearners(self):
    for i in range(self.steps):
      self.drawSample(i)
      s = self.samples[i]
      #print self.trainLearner(i)
      besterror = 1.0
      bestlearner = None
      for j in range(len(self.traindata[0]) - 1):
        lrn = self.trainLearner(i, j, s)
        #print lrn
        #key = str(lrn["attribute"]) + '/' + str(lrn["cutoff"])
        if lrn["error"] <= besterror:
          #self.attrdir[key] = True
          bestlearner = lrn
          besterror = lrn["error"]
      # This learner is better than guessing, so save it
      if besterror < 0.5:
        b = besterror/(1.0 - besterror)
        self.betas[i] = b
        self.learners[i] = bestlearner
        self.updateProbs(i)
      else:
        self.probs[i + 1] =  self.probs[i]

  def printLearners(self):
    for i, l in self.learners.iteritems():
      lstr = 'Learner ' + str(i)
      lstr += ' Attribute: ' + str(l["attribute"]) + ' Cutoff: ' + str(l["cutoff"])
      lstr += ' Direction: ' + str(l["direction"]) + ' Beta: ' + str(self.betas[i])
      print lstr

AB = AdaBoost(sys.argv[1], sys.argv[2], sys.argv[3])
AB.run()

