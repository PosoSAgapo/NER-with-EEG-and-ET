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