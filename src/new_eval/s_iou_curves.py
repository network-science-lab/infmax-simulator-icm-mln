from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.interpolate


def read_and_sort(df_path: str) -> pd.DataFrame:
    df = pd.read_csv(df_path)
    try: 
        df["actor"] = df["Unnamed: 0"]
        df = df.drop("Unnamed: 0", axis=1)
    except:
        ...
    df = df.set_index("actor")
    df = df.sort_values(
        ["exposed", "simulation_length", "peak_infected", "peak_iteration"],
        ascending=[False, True, True, False]
    )
    return df


def acc(df_y: pd.DataFrame, df_yhat: pd.DataFrame, cutoff: int) -> float:
    y = set(df_y.iloc[:cutoff].index.values)
    y_hat = set(df_yhat.iloc[:cutoff].index.values)
    return len(y.intersection(y_hat)) / cutoff


def cummulated_acc(df_y: pd.DataFrame, df_yhat: pd.DataFrame) -> np.array:
    assert len(df_y) == len(df_yhat)
    accs = []
    cutoffs = np.linspace(1, len(df_y), len(df_y), dtype=int)
    for cutoff in cutoffs:
        cutoff_acc = acc(df_y=df_y, df_yhat=df_yhat, cutoff=cutoff)
        accs.append(cutoff_acc)
    return np.array(accs)


def random_acc(y_df_len: int) -> float:
    return np.array([cutoff for cutoff in range(1, y_df_len + 1)]) / y_df_len


def average_curves(
        matrices: list[np.ndarray], target_length: int = None, kind: str = "linear"
) -> np.array:
    if target_length is None:
        target_length = max(len(m) for m in matrices)

    common_x = np.linspace(0, 1, target_length)
    resampled_curves = []

    for m in matrices:
        x_old = np.linspace(0, 1, len(m))
        interpolator = scipy.interpolate.interp1d(x_old, m, kind=kind, fill_value="extrapolate")
        resampled_curves.append(interpolator(common_x))

    avg_curve = np.mean(resampled_curves, axis=0)
    return avg_curve


def plot_accs(df_ys: list[pd.DataFrame], df_yhats: list[pd.DataFrame], df_names: list[str]) -> Figure:
    fig, ax = plt.subplots(nrows=1, ncols=1)

    acc_yhats = []
    for df_y, df_yhat, name in zip(df_ys, df_yhats, df_names):
        acc_yhat = cummulated_acc(df_y=df_y, df_yhat=df_yhat)
        cutoffs_yhat = random_acc(len(df_y))
        ax.plot(cutoffs_yhat, acc_yhat, label=f"{name}", alpha=0.2)
        acc_yhats.append(acc_yhat)

    acc_avg = average_curves(acc_yhats)
    cutoffs_avg = random_acc(len(acc_avg))
    ax.plot(cutoffs_avg, acc_avg, label="acc avg", color="green")

    ax.plot(cutoffs_avg, cutoffs_avg, "--", label="acc rand", color="red")

    ax.set_xlabel("size of cutoff")
    ax.set_xlim(0, 1)
    ax.set_ylabel("intersection(y_hat, y) / cutoff")
    ax.set_ylim(0, 1)
    ax.legend(
        loc="lower right",
        # ncol=2,
        bbox_to_anchor=(1.3, 0),
        # fancybox=True,
        handletextpad=0.1,
        borderaxespad=0.1,
        fontsize="small",
    )
    ax.set_aspect("equal")
    return fig
