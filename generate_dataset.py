from utils import get_datasets

num_users = 1000
for seed in [42, 182, 392]:
    for dataset_name in ["mnist", "emnist", "cifar10", "cifar100"]:
        if dataset_name == "cifar100":
            images_per_client = 2500
        else:
            images_per_client = 500
        # Global data for testing
        get_datasets(data_iid=True, num_users=1, dataset_name=dataset_name, global_data=True,
                     seed=seed, images_per_client=images_per_client)

        # IID and Non-IID
        get_datasets(data_iid=True, num_users=num_users, dataset_name=dataset_name,
                     seed=seed, images_per_client=images_per_client)
        get_datasets(data_iid=False, num_users=num_users, dataset_name=dataset_name,
                     seed=seed, images_per_client=images_per_client)