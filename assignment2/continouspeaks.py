import sys
import os
import time

sys.path.append('/home/song/Documents/CS7641/assignment2/ABAGAIL/ABAGAIL.jar')
import java.io.FileReader as FileReader
import java.io.File as File
import java.lang.String as String
import java.lang.StringBuffer as StringBuffer
import java.lang.Boolean as Boolean
import java.util.Random as Random

import dist.DiscreteDependencyTree as DiscreteDependencyTree
import dist.DiscreteUniformDistribution as DiscreteUniformDistribution
import dist.Distribution as Distribution
import opt.DiscreteChangeOneNeighbor as DiscreteChangeOneNeighbor
import opt.EvaluationFunction as EvaluationFunction
import opt.GenericHillClimbingProblem as GenericHillClimbingProblem
import opt.HillClimbingProblem as HillClimbingProblem
import opt.NeighborFunction as NeighborFunction
import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.example.FourPeaksEvaluationFunction as FourPeaksEvaluationFunction
import opt.ga.CrossoverFunction as CrossoverFunction
import opt.ga.SingleCrossOver as SingleCrossOver
import opt.ga.DiscreteChangeOneMutation as DiscreteChangeOneMutation
import opt.ga.GenericGeneticAlgorithmProblem as GenericGeneticAlgorithmProblem
import opt.ga.GeneticAlgorithmProblem as GeneticAlgorithmProblem
import opt.ga.MutationFunction as MutationFunction
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm
import opt.ga.UniformCrossOver as UniformCrossOver
import opt.prob.GenericProbabilisticOptimizationProblem as GenericProbabilisticOptimizationProblem
import opt.prob.MIMIC as MIMIC
import opt.prob.ProbabilisticOptimizationProblem as ProbabilisticOptimizationProblem
import shared.FixedIterationTrainer as FixedIterationTrainer
import opt.example.ContinuousPeaksEvaluationFunction as ContinuousPeaksEvaluationFunction
from array import array
from time import clock
from itertools import product

"""
Commandline parameter(s):
   none
"""

N = 100
T = 49
maxIters = 50000
fill = [2] * N
ranges = array('i', fill)
outfile = './CONTINOUSPEAKS_50000_XXX_LOG.csv'
ef = ContinuousPeaksEvaluationFunction(T)
odd = DiscreteUniformDistribution(ranges)
nf = DiscreteChangeOneNeighbor(ranges)
mf = DiscreteChangeOneMutation(ranges)
cf = SingleCrossOver()
df = DiscreteDependencyTree(.1, ranges)
hcp = GenericHillClimbingProblem(ef, odd, nf)
gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
pop = GenericProbabilisticOptimizationProblem(ef, odd, df)

# RHC
def rhc():
    fname = outfile.replace('XXX', 'RHC')
    with open(fname, 'w') as f:
        f.write('iterations,fitness,time,fevals\n')
    rhc = RandomizedHillClimbing(hcp)
    perform(rhc, fname)

# SA
def sa():
    for CE in [0.1, 0.3, 0.5, 0.7, 0.9]:
        fname = outfile.replace('XXX', 'SA{}'.format(CE))
        with open(fname, 'w') as f:
            f.write('iterations,fitness,time,fevals\n')
        sa = SimulatedAnnealing(1E10, CE, hcp)
        perform(sa, fname)

# GA
def ga():
    for popu, mate, mutate in product([50], [30, 20, 10], [20, 10]):
        fname = outfile.replace('XXX', 'GA{}_{}_{}'.format(popu, mate, mutate))
        with open(fname, 'w') as f:
            f.write('iterations,fitness,time,fevals\n')
        ga = StandardGeneticAlgorithm(popu, mate, mutate, gap)
        perform(ga, fname)

# MIMIC
def mimic():
    for samples, keep, m in product([80], [40], [0.15, 0.35, 0.55, 0.75, 0.95]):
        fname = outfile.replace('XXX', 'MIMIC{}_{}_{}'.format(samples, keep, m))
        with open(fname, 'w') as f:
            f.write('iterations,fitness,time,fevals\n')
        mimic = MIMIC(samples, keep, pop)
        perform(mimic, fname)


def perform(alg, fname):
    fit = FixedIterationTrainer(alg, 10)
    times = [0]
    for i in range(0, maxIters, 10):
        start = clock()
        fit.train()
        elapsed = time.clock() - start
        times.append(times[-1] + elapsed)
        score = ef.value(alg.getOptimal())
        st = '{},{},{}\n'.format(i, score, times[-1])
        # print st
        with open(fname, 'a') as f:
            f.write(st)

if __name__=="__main__":
    mimic()
    rhc()
    sa()
    ga()