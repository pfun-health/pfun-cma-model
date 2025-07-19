from pfun_cma_model.engine.data_utils import format_data
from pfun_cma_model.misc.pathdefs import PFunDataPaths


__all__ = [
    "format_data",
    "PFunDataPaths",
    "load_sample_data"
]


def load_sample_data():
    df_raw = PFunDataPaths().read_sample_data()
    df = format_data(df_raw)
    return df


if __name__ == "__main__":
    df = load_sample_data()
    print(df.head())
    print(df.tail())