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

def wrap_top3(col: pd.Series, only_first: bool = False) -> pd.Series:
    col = col.astype(float).round(3)
    ranking = col.rank(method="min", ascending=False)
    col = col.apply(lambda x: "{:.3f}".format(x))
    for im_name in ranking[ranking.values == 1.0].index:
        col[im_name] = r"\first{" + col[im_name] + "}"
    if only_first:
        return col
    for im_name in ranking[ranking.values == 2.0].index:
        col[im_name] = r"\second{" + col[im_name] + "}"
    for im_name in ranking[ranking.values == 3.0].index:
        col[im_name] = r"\third{" + col[im_name] + "}"
    return col

# COMPARISON OF ALL METHODS

# cva_path_all = "data/iou_curves/final_real/comparison_score_partial.csv"
# df = pd.read_csv(cva_path_all, index_col=0)
# print(df)
# df = df.loc[df["protocol"] == "AND"][
#     ["im_name", "net_type", "val_single", "auc_cutoff", "val_cutoff", "auc_full"]
# ]
# print(df)

# pivot_df = df.pivot(index='im_name', columns='net_type')[['val_single', 'val_cutoff', 'auc_full']]
# pivot_df = pivot_df.reorder_levels([1, 0], axis=1).sort_index(axis=1, level=0, sort_remaining=False)
# print(pivot_df)

# pivot_df = pivot_df.apply(wrap_top3)
# print(pivot_df)
# pivot_df.to_latex("partial_scores.tex")

# ABLATION STUDY / AVERAGED RESULTS

FOLDERS = [
    "final_artificial",
    "final_real",
    "avg-scores",
    # "20250419220536",
    # "20250419220605",
    # "model-aggregations",
    # "20250421105452",
    # "20250421105547",
    # "model-channels",
    # "20250422174945",
    # "20250422175018",
    # "data-transformations",
    # "20250423213639",
    # "20250423221831",
    # "model-encoders"
    # "20250424065140",
    # "20250424065230",
    # "data-features"
    # "20250425072846",  # this should be manualy splitted into two tables
    # "20250425073145",
    # "task-and-mae"  # or
]

def process_averaged_csv(csv_path):
    df = pd.read_csv(csv_path, index_col=0)
    print(df)
    df = df.loc[df["protocol"] == "AND"][
        ["im_name", "val_single", "auc_cutoff", "val_cutoff", "auc_full"]
    ]
    df = df.set_index("im_name").rename(index={"random_choice": "random"}).sort_index()
    return df

csv_path_real = f"data/iou_curves/{FOLDERS[1]}/comparison_score_avg.csv"
csv_path_art = f"data/iou_curves/{FOLDERS[0]}/comparison_score_avg.csv"

df_real = process_averaged_csv(csv_path_real)
df_art = process_averaged_csv(csv_path_art)
print(df_real)
print(df_art)

df_all = pd.concat([df_art, df_real], axis=1, keys=["Artificial networks", "Real networks"])
print(df_all)
df_all = df_all.rename(columns={
    "val_single": "$T_{val}$",
    "auc_cutoff": "$S_{auc}$",
    "val_cutoff": "$S_{val}$",
    "auc_full": "$F_{auc}$",
}, level=1)
df_all = df_all.apply(wrap_top3, only_first=False)
print(FOLDERS[2])
print(df_all)
df_all.to_latex(f"{FOLDERS[2]}.tex")
