"""A script to produce rankings of IM methods w.r.t. merged csv."""

import pandas as pd


def main(csv_path: str):
    df = pd.read_csv(csv_path, index_col=0)  # this should be a merged file
    print(df)
    RANKINGS = {}
    for protocol in df["protocol"].unique():
        print(protocol)
        d2f =  df.loc[df["protocol"] == protocol]
        for dataset_split in d2f["network_type"].unique():
            print(dataset_split)
            d3f = d2f.loc[d2f["network_type"] == dataset_split]
            for net_type in d3f["net_type"].unique():
                print(net_type)
                d4f = d3f.loc[d3f["net_type"] == net_type]
                rankings = {}
                for net_name in d4f["net_name"].unique():
                    d5f = d4f.loc[d4f["net_name"] == net_name].copy()
                    for metric in ["val_single", "val_cutoff"]:
                        d5f.loc[:, "ranking"] = d5f[metric].rank(ascending=False, method="dense")
                        # d5f.sort_values(by="val_single", ascending=False)
                        rankings[f"{net_name}-{metric}"] = d5f.set_index("im_name")["ranking"].to_dict()
                print("dupa")
                RANKINGS[f"{protocol}-{dataset_split}-{net_type}"] = pd.DataFrame(
                    rankings
                ).mean(axis=1).sort_values().to_dict()
        DF = pd.DataFrame(RANKINGS)
        print(DF)
        DF.to_csv("aaa.csv")
                        

if __name__ == "__main__":
    main(csv_path="comparison_sroce_partial.csv")
