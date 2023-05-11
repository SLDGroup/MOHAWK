from mobility_utils import create_mobility_data, read_df
from multiprocessing import freeze_support
import os

def create_data(devices, aps_lst, mtype_lst, non_mobility=False):
    if not os.path.exists("mobility_objects"):
        os.mkdir("mobility_objects")
    df = read_df()
    for aps in aps_lst:
        for mtype in mtype_lst:
            for seed in [42, 182, 392]:
                create_mobility_data(number_aps=aps, distance_based=False,
                                     ignore_out_of_range=False, distance=100, df=df,
                                     top_devices=devices, mtype=mtype, seed=seed, non_mobility=non_mobility)
def main():
    create_data(devices=1000, aps_lst=[100], mtype_lst=["hierfavg","macfl","mohawk"], non_mobility=True)
    create_data(devices=1000, aps_lst=[100], mtype_lst=["hierfavg","macfl","mohawk"], non_mobility=False)

if __name__ == "__main__":
    freeze_support()
    main()
