from readData import load
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

if __name__ == "__main__":
    # fileName1 = "hard_ql_qinit_0.0_lr_0.1_ep_0.1.txt"
    # fileName2 = "hard_ql_qinit_0.0_lr_0.1_ep_0.3.txt"
    # fileName3 = "hard_ql_qinit_0.0_lr_0.1_ep_0.5.txt"
    # fileName4 = "hard_ql_qinit_0.0_lr_0.1_ep_0.7.txt"
    # fileName5 = "hard_ql_qinit_0.0_lr_0.1_ep_0.9.txt"
    # fileName1 = "hard_ql_qinit_0.0_lr_0.1_ep_0.9.txt"
    # fileName2 = "hard_ql_qinit_0.0_lr_0.3_ep_0.9.txt"
    # fileName3 = "hard_ql_qinit_0.0_lr_0.5_ep_0.9.txt"
    # fileName4 = "hard_ql_qinit_0.0_lr_0.7_ep_0.9.txt"
    # fileName5 = "hard_ql_qinit_0.0_lr_0.9_ep_0.9.txt"

    # fileName1 = "hard_ql_gamma_0.7_lr_0.1_ep_0.9.txt"
    # fileName2 = "hard_ql_gamma_0.8_lr_0.1_ep_0.9.txt"
    # fileName3 = "hard_ql_gamma_0.9_lr_0.1_ep_0.9.txt"

    fileName1 = "easy_ql_gamma_0.7_lr_0.1_ep_0.9.txt"
    fileName2 = "easy_ql_gamma_0.8_lr_0.1_ep_0.9.txt"
    fileName3 = "easy_ql_gamma_0.9_lr_0.1_ep_0.9.txt"
    dat1 = load(fileName1)
    dat2 = load(fileName2)
    dat3 = load(fileName3)
    # dat4 = load(fileName4)
    # dat5 = load(fileName5)

    tmp1 = pd.Series(data=dat1['steps'])
    tmp2 = pd.Series(data=dat2['steps'])
    tmp3 = pd.Series(data=dat3['steps'])
    # tmp4 = pd.Series(data=dat4['steps'])
    # tmp5 = pd.Series(data=dat5['steps'])

    # df = pd.concat([tmp1, tmp2, tmp3, tmp4, tmp5], axis=1)
    df = pd.concat([tmp1, tmp2, tmp3], axis=1)
    df.columns = ['0.7', '0.8', '0.9']
    df.plot()
    plt.ylabel("Steps")
    plt.xlabel("Iteration")
    plt.title("Q-learning")
    plt.show()

    print("------------------------------\n")
    print(dat1[dat1['steps']==53].head(5))
    print("------------------------------\n")
    print(dat2[dat2['steps']==53].head(5))
    print("------------------------------\n")
    print(dat3[dat3['steps']==53].head(5))
    print("------------------------------\n")
    # print(dat4[dat4['steps']==53].head(5))
    # print("------------------------------\n")
    # print(dat5[dat5['steps']==53].head(5))