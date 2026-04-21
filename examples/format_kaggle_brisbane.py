# coding: utf-8
from pfun_cma_model.misc.pathdefs import PFunDataPaths
import pandas as pd


def main():
    dpath = PFunDataPaths().brist1d_data_fpath

    df = pd.read_parquet(dpath)
    print(df.head())

    df_subj = df[df.p_num == "p01"]
    print(df_subj.head())

    # blood glucose (get columns)
    # bg_cols = [c for c in df_subj.columns if "bg" in c]
    bg_cols = ["bg+1:00", ]
    
    df_subj_bg = df_subj[
        ["time", ] + bg_cols
    ]
    df_subj_bg.dropna(axis=1, inplace=True)
    df_subj_bg["displayTime"] = df_subj_bg.time
    df_subj_bg["systemTime"] = df_subj_bg["displayTime"].copy()

    print("\n\n", df_subj_bg)
    
    return df_subj_bg


if __name__ == '__main__':
    df_subj_bg = main()
