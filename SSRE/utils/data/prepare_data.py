from utils.data.para import datasets_all, AVAILABLE_TRANSFORMS_train, AVAILABLE_TRANSFORMS_test


def prepare_CL(dataset_name, args):
    dataset = datasets_all[dataset_name]
    dataset_train = dataset('{dir}/{name}'.format(dir=args.data_dir, name=args.dataset_name),
                            train=True, transform=AVAILABLE_TRANSFORMS_train[args.dataset_name],
                            target_transform=None, download=True)
    dataset_test = dataset('{dir}/{name}'.format(dir=args.data_dir, name=args.dataset_name),
                           train=False, transform=AVAILABLE_TRANSFORMS_test[args.dataset_name],
                           target_transform=None, download=True)
    return [dataset_train, dataset_test]


def prepare_HSI(dataset_name, args):
    dataset = datasets_all[dataset_name]
    dataset_train = dataset(args.data_dir, train=True, transform=AVAILABLE_TRANSFORMS_train[args.dataset_name],
                            target_transform=None, args=args, download=False)
    dataset_test = dataset(args.data_dir, train=False, transform=AVAILABLE_TRANSFORMS_test[args.dataset_name],
                           target_transform=None, args=args, download=False)
    args.input_channels = int(getattr(dataset_train, "spectral_dim", getattr(args, "pca_dim", 3)))
    return [dataset_train, dataset_test]
