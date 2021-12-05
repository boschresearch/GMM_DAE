''' helpers
    To visualize the prior and aggregtaed postrerior as pairplot.
'''
import matplotlib
matplotlib.use('Agg')
import seaborn as sb
import pandas as pd


def create_pairplot_from_array(parameter_array, parameter_names, title):
    """
    Creates seaborn pairplot from numpy array.
    """
    df = pd.DataFrame(parameter_array, columns=parameter_names)

    fig = sb.pairplot(
        df,
        diag_kind="kde",
        corner=True,
        plot_kws={"alpha": 0.5},
        diag_kws={"cumulative": True, "shade": False},
    )
    fig.fig.suptitle(title)
    return fig

