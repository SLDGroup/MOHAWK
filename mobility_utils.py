import os.path
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle
from multiprocessing import Pool
import multiprocessing

BBox = [-97.8395, -97.6819, 30.1961, 30.3511]


def read_df():
    if os.path.exists("mobility_objects/mobility_data_filt.pkl"):
        with open("mobility_objects/mobility_data_filt.pkl", "rb") as f:
            df = pickle.load(f)
            return df
    with open("RVF_ATX_PID_HZ_Places_Lookup.tsv",'r') as file:
        lookup = pd.read_csv(file,delimiter='\t')
    with open("RVF_ATX_PID_HZ-2020-05.tsv",'r') as file:
        data = pd.read_csv(file,delimiter='\t')
    df = pd.merge(data,lookup,on='venueid')

    mylist =[]
    for i in tqdm(range(len(df))):
        year = int(df["utc_date"][i].split('-')[0])
        month = int(df["utc_date"][i].split('-')[1])
        day = int(df["utc_date"][i].split('-')[2])
        hour = int(df["utc_hour"][i])


        date_time_obj = datetime(year, month=month, day=day, hour=hour)
        mylist.append(date_time_obj)
    df["timestamp"] = mylist
    df = df.sort_values(by="timestamp")
    df = df.reset_index()  # make sure indexes pair with number of rows
    with open(f"mobility_objects/mobility_data_filt.pkl", "wb") as f:
        pickle.dump(df, f)
    return df


def check_in_bounds(df):
    leftbound = df['geolong'] >= BBox[0]
    rightbound = df['geolong'] <= BBox[1]
    upperbound = df['geolat'] >= BBox[2]
    lowerbound = df['geolat'] <= BBox[3]
    condition = leftbound & rightbound & upperbound & lowerbound
    df = df[condition]
    return df


def filter_devices(df, top_devices=100):
    if top_devices < 0:
        top_devices = np.inf
    df = check_in_bounds(df)
    df = df.reset_index()  # make sure indexes pair with number of rows

    id_per_timestamp = {}
    my_ids = {}
    for idx ,t in tqdm(enumerate(df["timestamp"])):
        local_id = df["persistentid"][idx]
        if t not in id_per_timestamp:
            id_per_timestamp[t] = [local_id]
        else:
            if local_id not in id_per_timestamp[t]:
                id_per_timestamp[t].append(local_id)
            else:
                continue

        if local_id not in my_ids:
            my_ids[local_id] = {t: [df["geolat"][idx],df["geolong"][idx]]}
        else:
            my_ids[local_id][t] = [df["geolat"][idx],df["geolong"][idx]]
    print(f"IDs: {len(my_ids)}")
    print(f"Timestamps: {len(id_per_timestamp)}")

    id_occurrence = {}
    for local_id in my_ids.keys():
        id_occurrence[local_id] = len(my_ids[local_id])
    id_occurrence = {k: v for k, v in sorted(id_occurrence.items(), key=lambda item: item[1], reverse=True)}

    topkid = []
    cnt = 0
    for k in id_occurrence.keys():
        if cnt < top_devices:
            topkid.append(k)
        cnt += 1
    print(f"Selected IDs: {len(topkid)}")
    return topkid, id_per_timestamp, my_ids


def get_aps_coords(ap_df):
    aps_coords = {}
    for idx, row in ap_df.iterrows():
        ap_lat = row['geolat']
        ap_long = row['geolong']
        aps_coords[f"AP{idx}"] = [ap_long, ap_lat]
    return aps_coords


def check_duplicate_aps(mydict, all=False):
    # Receive AP: distance dictionary
    # Create distance: AP dictionary
    flipped = {}
    for key, value in mydict.items():
        if value not in flipped:
            flipped[value] = [key]
        else:
            flipped[value].append(key)
    # Check for any duplicate distances
    for k in flipped.keys():
        # If duplicate distances found just pick one AP randomly with equal probability for each AP to be picked
        if len(flipped[k]) > 1:
            flipped[k] = [np.random.choice(flipped[k])] # probabilities not given => samples assumes a uniform distribution over all entries

    # Ending up with a dictionary that has only one AP for every distance
    # Sort the dict based on its values; we want the shortest distance first
    sorted_keys = sorted(flipped.keys())
    try:
        if sorted_keys is None:
            with open(f"error_dict0.pkl", "wb") as f:
                pickle.dump(mydict, f)
            print("Sorted keys is NONE")
            print(sorted_keys[0])
            print(sorted_keys)
        if sorted_keys[0] is None:
            with open(f"error_dict1.pkl", "wb") as f:
                pickle.dump(mydict, f)
            print("Sorted keys is NONE")
            print(sorted_keys[0])
            print(sorted_keys)
    except BaseException as e:
        print(e)
        with open(f"error_dict0.pkl", "wb") as f:
            pickle.dump(mydict, f)
        print("Sorted keys is NONE")
        print(sorted_keys[0])
        print(sorted_keys)
    if all:
        return sorted_keys, flipped
    else:
        return sorted_keys[0], flipped[sorted_keys[0]]


def distance_to_ap(long, lat, ap_df, AP=None):
    ap_distances = {}
    aps_coords = get_aps_coords(ap_df)
    dist = None
    ap_name = None
    lst = []
    for k in aps_coords.keys():
        if AP is not None and k == AP:
            ap_long = aps_coords[k][0]
            ap_lat = aps_coords[k][1]
            dist = np.sqrt((long-ap_long)*(long-ap_long)+(lat-ap_lat)*(lat-ap_lat))
            ap_distances[k] = dist
            ap_name = [AP]
            break
        if AP is None:
            ap_long = aps_coords[k][0]
            ap_lat = aps_coords[k][1]
            dist = np.sqrt((long - ap_long) * (long - ap_long) + (lat - ap_lat) * (lat - ap_lat))
            ap_distances[k] = dist
    if AP is None:
        dist, ap_name = check_duplicate_aps(ap_distances, all=False)
    if dist is None:
        print(ap_distances)
    return ap_name, dist


def check_mapper_bounds(number, number_aps):
    if number < 0:
        return number_aps-1
    elif number > (number_aps-1):
        return 0
    else:
        return number

def go_through_all_clients(distance_based, id_per_timestamp, selectorAP, my_ids, t, aps, distance,
                           ignore_out_of_range, topkid):
    mob_data_filt = []
    for id in id_per_timestamp[t]:
        if id not in topkid:
            continue
        selectedAP = None
        if not distance_based:
            selectedAP = selectorAP[t][id]
        lat = my_ids[id][t][0]
        long = my_ids[id][t][1]
        ap_name, dist = distance_to_ap(long, lat, aps, AP=selectedAP)
        # ap_name, dist, speedup = distance_to_ap(long, lat, aps, apsmax, AP=selectedAP)
        # Distance larger than 100m
        if dist is None:
            print(dist)
            print(id)
            print(ap_name)
            print(selectedAP)
            print(t)
        if dist > 0.00089977 and distance == 100:
            if ignore_out_of_range:
                internet_speed = 0
            else:
                internet_speed = 50
        else:
            # 25 Mbps  LTE  https://www.lifewire.com/how-fast-is-4g-wireless-service-577566
            # 1000Mbps Wi-Fi
            internet_speed = 1000
        my_dict = {"id": id,
                   "lat": lat,
                   "long": long,
                   "AP_name": ap_name,
                   "dist_to_ap": dist,
                   "internet_speed": internet_speed}

        mob_data_filt.append(my_dict)
    return t, mob_data_filt

def create_mobility_data(number_aps=100, distance_based=False, ignore_out_of_range=False, distance=100, df=None,
                         top_devices=1000, mtype="hierfavg", seed=42, non_mobility=False):
    print()
    if ignore_out_of_range:
        connection="wifi"
    else:
        connection="wifi_lte"

    places = pd.read_csv("RVF_ATX_PID_HZ_Places_Lookup.tsv", sep='\t')
    places = places[['geolat', 'geolong']]
    places = check_in_bounds(places)
    places = places.reset_index()
    np.random.seed(seed)
    aps = places.sample(n=number_aps)
    aps = aps.reset_index()

    topkid, id_per_timestamp, my_ids = filter_devices(df=df, top_devices=top_devices)
    number_devices = len(topkid)
    fname = f"{mtype}_{number_devices}dev_{number_aps}ap_{connection}_{distance}m_s{seed}"
    print(fname)
    print(f"All places: {len(places)}")
    mapper = {}
    cnt = 0
    mydiv = number_devices/number_aps
    ap_cnt = 0
    selectorAP = {}
    if mtype == "hierfavg":
        distance_based = False
        for id in topkid:
            if cnt < mydiv + ap_cnt * mydiv:
                mapper[id] = f'AP{ap_cnt}'
            else:
                ap_cnt += 1
                mapper[id] = f'AP{ap_cnt}'
            cnt += 1

        selectorAP = {}
        for t in id_per_timestamp.keys():
            selectorAP[t]={}
            for id in topkid:
                selectorAP[t][id] = mapper[id]
    elif mtype == "macfl":
        distance_based = False
        for id in topkid:
            if cnt < mydiv + ap_cnt * mydiv:
                mapper[id] = ap_cnt
            else:
                ap_cnt += 1
                mapper[id] = ap_cnt
            cnt += 1
        selectorAP = {}
        first = True
        for t in id_per_timestamp.keys():
            selectorAP[t] = {}
            if first:
                for id in topkid:
                    selectorAP[t][id] = f"AP{mapper[id]}"
                first = False
            else:
                for id in topkid:
                    if np.random.random() <= 0.5: # Move
                        if np.random.random() <= 0.5: # Move right
                            mapper[id] = check_mapper_bounds(mapper[id]+1, number_aps)
                            selectorAP[t][id] = f"AP{mapper[id]}"
                        else:  # Move left
                            mapper[id] = check_mapper_bounds(mapper[id]-1, number_aps)
                            selectorAP[t][id] = f"AP{mapper[id]}"
                    else: # stay in the same space
                        selectorAP[t][id] = f"AP{mapper[id]}"
    elif mtype == "mohawk":
        distance_based = True

    mobility_data_filt = {}
    lst = []
    for t in tqdm(id_per_timestamp.keys()):
        lst.append((distance_based, id_per_timestamp, selectorAP, my_ids, t, aps, distance,
                    ignore_out_of_range, topkid))

    with Pool(processes=multiprocessing.cpu_count()) as pool:
        my_ret = pool.starmap(go_through_all_clients, lst)
        for t, mob_data_filt in my_ret:
            mobility_data_filt[t] = mob_data_filt

    for t in tqdm(id_per_timestamp.keys()):
        if not non_mobility:
            mobility_data_filt[t]=[]
            for id in id_per_timestamp[t]:
                if id not in topkid:
                    continue
                selectedAP = None
                if not distance_based:
                    selectedAP = selectorAP[t][id]
                lat = my_ids[id][t][0]
                long = my_ids[id][t][1]
                ap_name, dist = distance_to_ap(long, lat, aps, AP=selectedAP)
                # Distance larger than 100m
                if dist is None:
                    print(dist)
                    print(id)
                    print(ap_name)
                    print(selectedAP)
                    print(t)
                if dist > 0.00089977 and distance == 100:
                    if ignore_out_of_range:
                        internet_speed = 0
                    else:
                        internet_speed = 50
                else:
                    # 25 Mbps  LTE  https://www.lifewire.com/how-fast-is-4g-wireless-service-577566
                    # 1000Mbps Wi-Fi
                    internet_speed = 1000
                my_dict = {"id": id,
                           "lat": lat,
                           "long": long,
                           "AP_name": ap_name,
                           "dist_to_ap": dist,
                           "internet_speed": internet_speed}
                mobility_data_filt[t].append(my_dict)
        else:
            mobility_data_filt[t] = []
            for id in mapper.keys():
                if id not in topkid:
                    continue
                selectedAP = selectorAP[t][id]
                ap_name = [selectedAP]
                my_dict = {"id": id,
                           "lat": 0,
                           "long": 0,
                           "AP_name": ap_name,
                           "dist_to_ap": 0,
                           "internet_speed": 0}
                mobility_data_filt[t].append(my_dict)

    unique_ids = {}
    idx_unique_ids = {}
    cnt = 0
    if not non_mobility:
        for t in mobility_data_filt.keys():
            for idx,elem in enumerate(mobility_data_filt[t]):
                if elem['id'] not in unique_ids:
                    if cnt % 2 == 0:
                        unique_ids[elem['id']] = 'MC1'
                        idx_unique_ids[elem['id']] = cnt
                    else:
                        unique_ids[elem['id']] = 'RPi'
                        idx_unique_ids[elem['id']] = cnt
                    cnt += 1
                mobility_data_filt[t][idx]['device_type'] = unique_ids[elem['id']]
                mobility_data_filt[t][idx]['device_idx'] = idx_unique_ids[elem['id']]
    else:
        for t in mobility_data_filt.keys():
            for idx,elem in enumerate(mobility_data_filt[t]):
                if elem['id'] not in unique_ids:
                    if cnt % 2 == 0:
                        unique_ids[elem['id']] = 'MC1'
                        idx_unique_ids[elem['id']] = cnt
                    else:
                        unique_ids[elem['id']] = 'RPi'
                        idx_unique_ids[elem['id']] = cnt
                    cnt += 1
                mobility_data_filt[t][idx]['device_type'] = unique_ids[elem['id']]
                mobility_data_filt[t][idx]['device_idx'] = idx_unique_ids[elem['id']]


    rpis = 0
    mc1s = 0
    for k in unique_ids.keys():
        if unique_ids[k] == 'RPi':
            rpis += 1
        elif unique_ids[k] == 'MC1':
            mc1s += 1
    print(rpis, mc1s)

    if not non_mobility:
        with open(f"mobility_objects/{fname}.pkl", "wb") as f:
            pickle.dump(mobility_data_filt,f)
    else:
        with open(f"mobility_objects/{fname}_nonmobility.pkl", "wb") as f:
            pickle.dump(mobility_data_filt,f)

    return fname
