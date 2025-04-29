import pandas as pd


IM_ORDER = [
    "deg-c",
    "deg-cd",
    "deep-im",
    "mn2v-km",
    "nghb-s",
    "nghb-sd",
    "random",
    "ts-net"
]

csv_path_real = "data/iou_curves/final_real/comparison_score_avg.csv"
csv_path_art = "data/iou_curves/final_artificial/comparison_score_avg.csv"

def process_averaged_csv(csv_path):
    df = pd.read_csv(csv_path, index_col=0)
    print(df)
    df = df.loc[df["protocol"] == "AND"][
        # ["im_name", "auc_single", "auc_cutoff", "auc_full", "val_single", "val_cutoff"]
        ["im_name", "val_single", "auc_cutoff", "val_cutoff", "auc_full"]
    ]
    df = df.set_index("im_name").rename(index={"random_choice": "random"}).reindex(IM_ORDER)
    return df

# df_real = process_averaged_csv(csv_path_real)
# df_art = process_averaged_csv(csv_path_art)
# print(df_real)
# print(df_art)

# df_all = df_art.join(df_real, lsuffix="_a", rsuffix="_r")
# print(df_all)
# df_all = df_all.style.format(precision=3)
# df_all.to_latex("avg_scores.tex")


def wrap_top3(col: pd.Series) -> pd.Series:
    col = col.astype(float)
    ranking = col.rank(method="min", ascending=False)
    col = col.apply(lambda x: "{:.3f}".format(x))
    for im_name in ranking[ranking.values == 1.0].index:
        col[im_name] = r"\first{" + col[im_name] + "}"
    for im_name in ranking[ranking.values == 2.0].index:
        col[im_name] = r"\second{" + col[im_name] + "}"
    for im_name in ranking[ranking.values == 3.0].index:
        col[im_name] = r"\third{" + col[im_name] + "}"
    return col


cva_path_all = "data/iou_curves/final_real/comparison_score_partial.csv"
df = pd.read_csv(cva_path_all, index_col=0)
print(df)
df = df.loc[df["protocol"] == "AND"][
    ["im_name", "net_type", "val_single", "auc_cutoff", "val_cutoff", "auc_full"]
]
print(df)

pivot_df = df.pivot(index='im_name', columns='net_type')[['val_single', 'val_cutoff', 'auc_full']]
pivot_df = pivot_df.reorder_levels([1, 0], axis=1).sort_index(axis=1, level=0, sort_remaining=False)
print(pivot_df)

pivot_df = pivot_df.apply(wrap_top3)
print(pivot_df)
pivot_df.to_latex("partial_scores.tex")
