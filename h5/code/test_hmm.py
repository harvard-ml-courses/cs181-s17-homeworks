#!/usr/bin/env python

"""
test_hmm.py -- unit tests for hmms implemented in hmm.py
"""

from hmm import *
from viterbi import *
from dataset import DataSet
import functools
import math
import unittest

class HMMsTest(unittest.TestCase):
    # test for learn_from_labeled_data()
    def test_simple_hmm_learning(self):
	state_seq = [[0,1,1,0,1,0,1,1], [0,0,1,0]]
	obs_seq =   [[0,0,1,1,0,0,0,1], [0,1,0,0]]
	hmm = HMM(range(2), range(2))
	hmm.learn_from_labeled_data(state_seq, obs_seq)
	print hmm
	eps = 0.00001
	self.assertTrue(max_delta(hmm.initial, [0.750000,0.250000]) < eps)
	self.assertTrue(max_delta(hmm.transition, 
				 [[0.285714, 0.714286],
	                          [0.571429, 0.428571]]) < eps)
	self.assertTrue(max_delta(hmm.observation,
	                          [[0.625000, 0.375000],
	                           [0.625000, 0.375000]]) < eps)
	
	
def simple_weather_model():
    hmm = HMM(['s1','s2'], ['R','NR'])
    init = [0.7, 0.3]
    trans = [[0.8,0.2],
             [0.1,0.9]]
    observ = [[0.75,0.25], 
              [0.4,0.6]]
    hmm.set_hidden_model(init, trans, observ)
    return hmm

class ViterbiTest(unittest.TestCase):
    def toy_model(self):
	hmm = HMM(['s1','s2'], ['R','NR'])
	init = [0.5, 0.5]
	trans = [[0.2,0.8],
		 [0.8,0.2]]
	observ = [[0.8,0.2], 
		  [0.2,0.8]]
	hmm.set_hidden_model(init, trans, observ)
	return hmm

		
    def test_viterbi_simple_sequence(self):
	hmm = simple_weather_model()
	seq = [1, 1, 0]  # NR, NR, R
	hidden_seq = hmm.most_likely_states(seq)
	print "most likely states for [NR, NR, R] = %s" % hidden_seq
	self.assertEqual(hidden_seq, [1,1,1])

    def test_viterbi_long_sequence(self):
	hmm = self.toy_model()
	N = 10
	seq = [1,0,1,0,1,0,1,1,0] * 400
	hidden_seq = hmm.most_likely_states(seq, False)
	# Check if we got right answer from the version with logs.
	self.assertEqual(hidden_seq[2000:2010], [1, 0, 1, 0, 1, 1, 0, 1, 0, 1])


class RobotTest(unittest.TestCase):
    def test_small_robot_dataset(self):
	data_filename = "robot_small.data"
	data_filename = normalize_filename(data_filename)
	hmm, d = train_hmm_from_data(data_filename)
	err_full = run_viterbi(hmm, d, True)
	self.assertAlmostEqual(err_full, 2.0/9)		
		
if __name__ == '__main__':
    unittest.main()
    
