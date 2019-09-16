import matplotlib
import matplotlib.pyplot as plt
import numpy as np 

def show_means(ax, boxplot):
    mean_nr = boxplot['means'][0]
    mean_tsr = boxplot['means'][1]

    # x position of the mean lines for NR and TSR respectively
    xpos_mean_nr = mean_nr.get_xdata()
    xpos_mean_tsr = mean_tsr.get_xdata()

    # Lets make the text have a horizontal offset which is some 
    # fraction of the width of the box
    xoff_mean_nr = 0.10 * (xpos_mean_nr[1] - xpos_mean_nr[0])
    xoff_mean_tsr = 0.10 * (xpos_mean_tsr[1] - xpos_mean_tsr[0])

    # x position of the labels
    xlabel_mean_nr = xpos_mean_nr[1] + xoff_mean_nr
    xlabel_mean_tsr = xpos_mean_tsr[1] + xoff_mean_tsr

    mean_nr = mean_nr.get_ydata()[1]
    mean_tsr = mean_tsr.get_ydata()[1]

    ax.text(xlabel_mean_nr, mean_nr,
                r'$\bar x$ = {:2.3g}'.format(mean_nr), va='center')
    ax.text(xlabel_mean_tsr, mean_tsr,
                r'$\bar x$ = {:2.3g}'.format(mean_tsr), va='center')
    
    
def plot_fix(mean_fixations_t2, mean_fixations_t3):
    fig, ax = plt.subplots(1,2,figsize=(12,6))
    ax[0].scatter(np.arange(0, 12), mean_fixations_t2, alpha=0.4, color='blue', label='First Half')
    ax[0].scatter(np.arange(0, 12), mean_fixations_t3, alpha=0.4, color='red', label='Second Half')
    ax[0].set_xticks(ticks=np.arange(0, 12))
    ax[0].set_xticklabels(labels=['Subject'+' '+str(i) for i in range(1, 13)], rotation=90)
    ax[0].set_xlabel("Test subjects")
    ax[0].set_ylabel("Mean Fixations (frequency)")
    ax[0].legend(fancybox=True, framealpha=1, loc='lower right')
    ax[0].set_title("Mean number of fixations per word per subject", fontsize=11)

    boxplot = ax[1].boxplot([mean_fixations_t2, mean_fixations_t3], showmeans=True,
               labels=['NR', 'TSR'], meanline=True)
    ax[1].set_xticks(ticks=np.arange(1, 3))
    ax[1].set_xticklabels(labels=['NR', 'TSR'])
    ax[1].set_ylabel("Mean Fixations (frequency)")
    ax[1].set_title("Mean number of fixations per word per subject", fontsize=11)
    show_means(ax[1], boxplot)

    plt.show()
    
    
def plot_omissions(mean_omissions_t2, mean_omissions_t3):
    fig, ax = plt.subplots(1,2,figsize=(12,6))

    ax[0].scatter(np.arange(0, 12), mean_omissions_t2, alpha=0.4, color='blue', label='First Half')
    ax[0].scatter(np.arange(0, 12), mean_omissions_t3, alpha=0.4, color='red', label='Second Half')
    ax[0].set_xticks(ticks=np.arange(0, 12))
    ax[0].set_xticklabels(labels=['Subject'+' '+str(i) for i in range(1, 13)], rotation=90)
    ax[0].set_xlabel("Test subjects")
    ax[0].set_ylabel("Proportion")
    ax[0].legend(fancybox=True, framealpha=1, loc='lower right')
    ax[0].set_title("Mean omission rate on sentence level", fontsize=11)

    boxplot = ax[1].boxplot([mean_omissions_t2, mean_omissions_t3], showmeans=True,
               labels=['NR', 'TSR'], meanline=True)
    ax[1].set_xticks(ticks=np.arange(1, 3))
    ax[1].set_xticklabels(labels=['NR', 'TSR'])
    ax[1].set_ylabel("Proportion")
    ax[1].set_title("Mean omission rate on sentence level", fontsize=11)
    show_means(ax[1], boxplot)

    plt.show()
    
def plot_gd(mean_gd_t2, mean_gd_t3):
    fig, ax = plt.subplots(1,2,figsize=(12,6))

    ax[0].scatter(np.arange(0, 12), mean_gd_t2, alpha=0.4, color='blue', label='First Half')
    ax[0].scatter(np.arange(0, 12), mean_gd_t3, alpha=0.4, color='red', label='Second Half')
    ax[0].set_xticks(ticks=np.arange(0, 12))
    ax[0].set_xticklabels(labels=['Subject'+' '+str(i) for i in range(1, 13)], rotation=90)
    ax[0].set_xlabel("Test subjects")
    ax[0].set_ylabel("Mean GD (ms)")
    ax[0].legend(fancybox=True, framealpha=1, loc='lower right')
    ax[0].set_title("Mean gaze duration per word", fontsize=11)

    boxplot = ax[1].boxplot([mean_gd_t2, mean_gd_t3], showmeans=True,
               labels=['NR', 'TSR'], meanline=True)
    ax[1].set_xticks(ticks=np.arange(1, 3))
    ax[1].set_xticklabels(labels=['NR', 'TSR'])
    ax[1].set_ylabel("Mean GD (ms)")
    ax[1].set_title("Mean gaze duration per word", fontsize=11)
    show_means(ax[1], boxplot)

    plt.show()
    
def plot_trt(mean_trt_t2, mean_trt_t3):
    fig, ax = plt.subplots(1,2,figsize=(12,6))

    ax[0].scatter(np.arange(0, 12), mean_trt_t2, alpha=0.4, color='blue', label='First Half')
    ax[0].scatter(np.arange(0, 12), mean_trt_t3, alpha=0.4, color='red', label='Second Half')
    ax[0].set_xticks(ticks=np.arange(0, 12))
    ax[0].set_xticklabels(labels=['Subject'+' '+str(i) for i in range(1, 13)], rotation=90)
    ax[0].set_xlabel("Test subjects")
    ax[0].set_ylabel("Mean TRT (ms)")
    ax[0].legend(fancybox=True, framealpha=1, loc='lower right')
    ax[0].set_title("Mean total reading time per word", fontsize=11)

    boxplot = ax[1].boxplot([mean_trt_t2, mean_trt_t3], showmeans=True,
               labels=['NR', 'TSR'], meanline=True)
    ax[1].set_xticks(ticks=np.arange(1, 3))
    ax[1].set_xticklabels(labels=['NR', 'TSR'])
    ax[1].set_ylabel("Mean TRT (ms)")
    ax[1].set_title("Mean total reading time per word", fontsize=11)
    show_means(ax[1], boxplot)

    plt.show()
    

def plot_ffd(mean_ffd_t2, mean_ffd_t3):
    fig, ax = plt.subplots(1,2,figsize=(12,6))

    ax[0].scatter(np.arange(0, 12), mean_ffd_t2, alpha=0.4, color='blue', label='First Half')
    ax[0].scatter(np.arange(0, 12), mean_ffd_t3, alpha=0.4, color='red', label='Second Half')
    ax[0].set_xticks(ticks=np.arange(0, 12))
    ax[0].set_xticklabels(labels=['Subject'+' '+str(i) for i in range(1, 13)], rotation=90)
    ax[0].set_xlabel("Test subjects")
    ax[0].set_ylabel("Mean FFD (ms)")
    ax[0].legend(fancybox=True, framealpha=1, loc='lower right')
    ax[0].set_title("Mean first fixation duration per word", fontsize=11)

    boxplot = ax[1].boxplot([mean_ffd_t2, mean_ffd_t3], showmeans=True,
               labels=['NR', 'TSR'], meanline=True)
    ax[1].set_xticks(ticks=np.arange(1, 3))
    ax[1].set_xticklabels(labels=['NR', 'TSR'])
    ax[1].set_ylabel("Mean FFD (ms)")
    ax[1].set_title("Mean first fixation duration per word", fontsize=11)
    show_means(ax[1], boxplot)

    plt.show()
