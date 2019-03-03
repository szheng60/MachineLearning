import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

sa_1e_15= "SA_1e+12_0.15_LOG.csv"
sa_1e_95 = "SA_1e+12_0.95_LOG.csv"

sa_10000000000_15 = "SA_100000000.0_0.15_LOG.csv"
sa_10000000000_35 = "SA_100000000.0_0.35_LOG.csv"
sa_10000000000_55 = "SA_100000000.0_0.55_LOG.csv"
sa_10000000000_7 = "SA_100000000.0_0.7_LOG.csv"
sa_10000000000_85 = "SA_100000000.0_0.85_LOG.csv"
sa_10000000000_95 = "SA_100000000.0_0.95_LOG.csv"

# sa_10000000000_15 = "SA_10000000000.0_0.15_LOG.csv"
# sa_10000000000_95 = "SA_10000000000.0_0.95_LOG.csv"

ga_100_10_5_3000 =  "GA_100_10_5_3000_LOG.csv"
ga_100_10_10_3000 =  "GA_100_10_10_3000_LOG.csv"
ga_100_10_20_3000 =  "GA_100_10_20_3000_LOG.csv"

ga_100_20_5_3000 =  "GA_100_20_5_3000_LOG.csv"
ga_100_20_10_3000 =  "GA_100_20_10_3000_LOG.csv"
ga_100_20_20_3000 =  "GA_100_20_20_3000_LOG.csv"


ga_20_10_5_3000 = "GA_20_10_5_3000_LOG.csv"
ga_50_10_5_3000 = "GA_50_10_5_3000_LOG.csv"

# csv_list = [sa_10000000000_15, sa_10000000000_35, sa_10000000000_55, sa_10000000000_7, sa_10000000000_85, sa_10000000000_95]
# csv_list = [sa_100000000_95, sa_100000000_15, sa_10000000000_95, sa_10000000000_15, sa_1e_95, sa_1e_15]
# csv_list = [ga_100_10_5_3000, ga_100_10_10_3000, ga_100_10_20_3000]
# csv_list = [ga_100_10_20_3000, ga_100_20_20_3000]
csv_list = [ga_20_10_5_3000, ga_50_10_5_3000, ga_100_10_5_3000]

# c1_ga = "CONTINOUSPEAKS_50000_GA50_30_10_LOG"
# c1_mimic = "CONTINOUSPEAKS_50000_MIMIC80_40_0.55_LOG"
# c1_rhc = "CONTINOUSPEAKS_50000_RHC_LOG"
# c1_sa = "CONTINOUSPEAKS_50000_SA0.7_LOG"
# c1_ga = "CONTINOUSPEAKS_GA50_30_10_LOG"
# c1_mimic = "CONTINOUSPEAKS_MIMIC80_40_0.55_LOG"
# c1_rhc = "CONTINOUSPEAKS_RHC_LOG"
# c1_sa = "CONTINOUSPEAKS_SA0.7_LOG"
# continuous_list = [c1_ga, c1_mimic, c1_rhc, c1_sa]
c_sa_1 = "CONTINOUSPEAKS_50000_SA0.1_LOG"
c_sa_2 = "CONTINOUSPEAKS_50000_SA0.3_LOG"
c_sa_3 = "CONTINOUSPEAKS_50000_SA0.5_LOG"
c_sa_4 = "CONTINOUSPEAKS_50000_SA0.7_LOG"
c_sa_5 = "CONTINOUSPEAKS_50000_SA0.9_LOG"
continuous_list = [c_sa_1, c_sa_2, c_sa_3, c_sa_4, c_sa_5]


# k_ga = "KNAPSACK_500_5000_GA50_30_10_LOG"
# k_mimic = "KNAPSACK_500_5000_MIMIC80_40_0.95_LOG"
# k_rhc = "KNAPSACK_500_5000_RHC_LOG"
# k_sa = "KNAPSACK_500_5000_SA0.7_LOG"
k_ga = "KNAPSACK_GA50_30_10_LOG"
k_mimic = "KNAPSACK_MIMIC80_40_0.95_LOG"
k_rhc = "KNAPSACK_RHC_LOG"
k_sa = "KNAPSACK_SA0.7_LOG"
# knapsack_list = [k_ga, k_mimic, k_rhc, k_sa]

k_mimic_5000_15 = "KNAPSACK_500_5000_MIMIC80_40_0.15_LOG"
k_mimic_5000_35 = "KNAPSACK_500_5000_MIMIC80_40_0.35_LOG"
k_mimic_5000_55 = "KNAPSACK_500_5000_MIMIC80_40_0.55_LOG"
k_mimic_5000_75 = "KNAPSACK_500_5000_MIMIC80_40_0.75_LOG"
k_mimic_5000_95 = "KNAPSACK_500_5000_MIMIC80_40_0.95_LOG"


k_mimic_80_40_m_15 = "KNAPSACK_5000_MIMIC80_40_0.15_LOG"
k_mimic_80_40_m_35 = "KNAPSACK_5000_MIMIC80_40_0.35_LOG"
k_mimic_80_40_m_55 = "KNAPSACK_5000_MIMIC80_40_0.55_LOG"
k_mimic_80_40_m_75 = "KNAPSACK_5000_MIMIC80_40_0.75_LOG"
k_mimic_80_40_m_95 = "KNAPSACK_5000_MIMIC80_40_0.95_LOG"

#
# k_mimic_80_40_m_15 = "KNAPSACK_MIMIC80_40_0.15_LOG"
# k_mimic_80_40_m_35 = "KNAPSACK_MIMIC80_40_0.35_LOG"
# k_mimic_80_40_m_55 = "KNAPSACK_MIMIC80_40_0.55_LOG"
# k_mimic_80_40_m_75 = "KNAPSACK_MIMIC80_40_0.75_LOG"
# k_mimic_80_40_m_95 = "KNAPSACK_MIMIC80_40_0.95_LOG"


k_mimic_80_10_55 = "KNAPSACK_5000_MIMIC80_10_0.55_LOG"
k_mimic_80_20_55 = "KNAPSACK_5000_MIMIC80_20_0.55_LOG"
k_mimic_80_30_55 = "KNAPSACK_5000_MIMIC80_30_0.55_LOG"
k_mimic_80_40_55 = "KNAPSACK_5000_MIMIC80_40_0.55_LOG"
k_mimic_80_50_55 = "KNAPSACK_5000_MIMIC80_50_0.55_LOG"
k_mimic_80_60_55 = "KNAPSACK_5000_MIMIC80_60_0.55_LOG"
k_mimic_80_70_55 = "KNAPSACK_5000_MIMIC80_70_0.55_LOG"

k_mimic_20_10_55 = "KNAPSACK_5000_MIMIC20_10_0.55_LOG"
k_mimic_30_10_55 = "KNAPSACK_5000_MIMIC30_10_0.55_LOG"
k_mimic_40_10_55 = "KNAPSACK_5000_MIMIC40_10_0.55_LOG"
k_mimic_50_10_55 = "KNAPSACK_5000_MIMIC50_10_0.55_LOG"
k_mimic_60_10_55 = "KNAPSACK_5000_MIMIC60_10_0.55_LOG"
k_mimic_70_10_55 = "KNAPSACK_5000_MIMIC70_10_0.55_LOG"
# k_mimic_80_40_55 = "KNAPSACK_5000_MIMIC70_40_0.55_LOG"
# k_mimic_80_10_55 = "KNAPSACK_5000_MIMIC80_10_0.55_LOG"

# knapsack_list = [k_ga, k_mimic, k_rhc, k_sa]
# knapsack_list = [k_mimic_5000_15, k_mimic_5000_35, k_mimic_5000_55, k_mimic_5000_75, k_mimic_5000_95]
# knapsack_list = [k_mimic_80_40_m_15, k_mimic_80_40_m_35, k_mimic_80_40_m_55,
#                  k_mimic_80_40_m_75, k_mimic_80_40_m_95]
# knapsack_list = [k_mimic_80_10_55, k_mimic_80_20_55, k_mimic_80_30_55, k_mimic_80_40_55,
#                  k_mimic_80_50_55, k_mimic_80_60_55, k_mimic_80_70_55]
knapsack_list = [k_mimic_20_10_55, k_mimic_30_10_55, k_mimic_40_10_55, k_mimic_50_10_55, k_mimic_60_10_55,
                 k_mimic_70_10_55, k_mimic_80_10_55]

# k_mimic_80_10_75 = "KNAPSACK_MIMIC80_10_0.75_LOG"
# k_mimic_80_20_75 = "KNAPSACK_MIMIC80_20_0.75_LOG"
# k_mimic_80_30_75 = "KNAPSACK_MIMIC80_30_0.75_LOG"
# k_mimic_80_40_75 = "KNAPSACK_MIMIC80_40_0.75_LOG"
#
# knapsack_list = [k_mimic_80_10_75, k_mimic_80_20_75, k_mimic_80_30_75, k_mimic_80_40_75]


# tsp_5000_ga = "TSP_5000_GA50_30_10_LOG"
# tsp_5000_mimic = "TSP_5000_MIMIC80_40_0.55_LOG"
# tsp_5000_rhc = "TSP_5000_RHC_LOG"
# tsp_5000_sa = "TSP_5000_SA0.7_LOG"

# tsp_list = [tsp_5000_ga, tsp_5000_mimic, tsp_5000_rhc, tsp_5000_sa]

# tsp_5000_ga = "TSP_100000_GA50_30_10_LOG"
# tsp_5000_mimic = "TSP_100000_MIMIC80_40_0.55_LOG"
# tsp_5000_rhc = "TSP_100000_RHC_LOG"
# tsp_5000_sa = "TSP_100000_SA0.7_LOG"
#
# tsp_list = [tsp_5000_ga, tsp_5000_mimic, tsp_5000_rhc, tsp_5000_sa]

# tsp_sample_50 = "TSP_GA50_30_20_LOG"
# tsp_sample_60 = "TSP_GA60_30_20_LOG"
# tsp_sample_70 = "TSP_GA70_30_20_LOG"
# tsp_sample_80 = "TSP_GA80_30_20_LOG"
# tsp_sample_90 = "TSP_GA90_30_20_LOG"
#
# tsp_list = [tsp_sample_50, tsp_sample_60, tsp_sample_70, tsp_sample_80, tsp_sample_90]

# tsp_90_mate_10 = "TSP_GA90_10_20_LOG"
# tsp_90_mate_20 = "TSP_GA90_20_20_LOG"
# tsp_90_mate_30 = "TSP_GA90_30_20_LOG"
# tsp_90_mate_40 = "TSP_GA90_40_20_LOG"
#
# tsp_list = [tsp_90_mate_10, tsp_90_mate_20, tsp_90_mate_30, tsp_90_mate_40]

tsp_90_40_mutation_5 = "TSP_GA90_40_5_LOG"
tsp_90_40_mutation_10 = "TSP_GA90_40_10_LOG"
tsp_90_40_mutation_20 = "TSP_GA90_40_20_LOG"
tsp_90_40_mutation_30 = "TSP_GA90_40_30_LOG"

tsp_list = [tsp_90_40_mutation_5, tsp_90_40_mutation_10, tsp_90_40_mutation_20, tsp_90_40_mutation_30]

def plotTSP():
    result_x = pd.DataFrame()
    result_y = pd.DataFrame()
    for l in tsp_list:
        df = pd.read_csv(l + ".csv")
        result_x = df['iterations']
        result_y[len(result_y.columns)] = df['fitness']
        # result_y[len(result_y.columns)] = df['time']

    plt.xlabel("Iteration")
    # plt.ylabel("Time")
    plt.ylabel("Fitness")
    # plt.ylabel("Accuracy")
    plt.title("Traveling Sales Man Evaluation")
    # plt.title("Test Accuracy on Iteration")
    plt.plot(result_x,result_y)
    plt.gca().legend(tsp_list)
    axes = plt.gca()
    # axes.set_xlim([0, 10000])
    # axes.set_ylim([0, 0.04])
    # axes.set_ylim([0.0795, 0.084])
    plt.show()


def plotKnapSack():
    result_x = pd.DataFrame()
    result_y = pd.DataFrame()
    for l in knapsack_list:
        df = pd.read_csv(l + ".csv")
        result_x = df['iterations']
        result_y[len(result_y.columns)] = df['fitness']
        # result_y[len(result_y.columns)] = df['time']

    plt.xlabel("Iteration")
    # plt.ylabel("Time")
    plt.ylabel("Fitness")
    # plt.ylabel("Accuracy")
    plt.title("Knapsack Evaluation")
    # plt.title("Test Accuracy on Iteration")
    plt.plot(result_x,result_y)
    plt.gca().legend(knapsack_list)
    axes = plt.gca()
    # axes.set_xlim([0, 10000])
    # axes.set_ylim([0, 0.04])
    # axes.set_ylim([0.0795, 0.084])
    plt.show()

def plotContinousPeaks():
    result_x = pd.DataFrame()
    result_y = pd.DataFrame()
    for l in continuous_list:
        df = pd.read_csv(l + ".csv")
        result_x = df['iterations']
        # result_y[len(result_y.columns)] = df['fitness']
        result_y[len(result_y.columns)] = df['time']

    plt.xlabel("Iteration")
    plt.ylabel("Time")
    # plt.ylabel("Fitness")
    # plt.ylabel("Accuracy")
    plt.title("Continuous Peaks Evaluation")
    # plt.title("Test Accuracy on Iteration")
    plt.plot(result_x,result_y)
    plt.gca().legend(continuous_list)
    axes = plt.gca()
    axes.set_xlim([0, 10000])
    axes.set_ylim([0, 0.04])
    # axes.set_ylim([0.0795, 0.084])
    plt.show()

def getData(list=csv_list):

    result_x = pd.DataFrame()
    result_y = pd.DataFrame()
    for l in list:
        df = pd.read_csv(l)
        # elp = df['acc_trg']
        # print(l, np.max(elp))
        result_x = df['iteration']
        result_y[len(result_y.columns)] = df['MSE_tst']
        # result_y[len(result_y.columns)] = df['acc_tst']

    return result_x, result_y

def plot(x, y):
    plt.xlabel("Iteration")
    plt.ylabel("MSE")
    # plt.ylabel("Accuracy")
    plt.title("Train VS Test MSE on Iteration")
    # plt.title("Test Accuracy on Iteration")
    plt.plot(x,y)
    plt.gca().legend(csv_list)
    axes = plt.gca()
    # axes.set_ylim([0.79, 0.82])
    axes.set_ylim([0.0795, 0.084])
    plt.show()

def plotFig(file):
    df = pd.read_csv(file)
    print(df.columns)
    x = df['iteration']
    # y = df[['MSE_trg', 'MSE_tst']]
    y = df[['acc_trg', 'acc_tst']]

    # print(np.max(df['acc_tst']))
    # print(np.min(df['MSE_tst']))

    plt.xlabel("Iteration")
    # plt.ylabel("MSE")
    plt.ylabel("Accuracy")
    # plt.title("Train VS Test MSE on Iteration")
    plt.title("Train VS Test Accuracy on Iteration")
    plt.plot(x,y)
    plt.gca().legend(('train', 'test'))
    plt.show()

if __name__=="__main__":
    # plotFig("GA_20_10_5_3000_LOG.csv")
    # plotFig("SA_1e+12_0.15_LOG.csv")
    # x, y =getData()
    # plot(x, y)
    # plotContinousPeaks()
    # plotKnapSack()
    plotTSP()