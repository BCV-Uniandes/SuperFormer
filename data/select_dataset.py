
def define_Dataset(dataset_opt):
    dataset_type = dataset_opt['dataset_type'].lower()
    # -----------------------------------------
    # super-resolution
    # -----------------------------------------
    if dataset_type in ['hcp']:
        from data.dataset_HCP import Dataset_HCP as D
    elif dataset_type in ['hcp_2d']:
        from data.dataset_HCP import HCP1200_2D_Dataset as D
    else:
        raise NotImplementedError('Dataset [{:s}] is not found.'.format(dataset_type))
    dataset = D(dataset_opt)
    print('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__, dataset_opt['name']))
    return dataset
