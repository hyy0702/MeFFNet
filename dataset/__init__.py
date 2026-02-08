from dataset.data_chemblbdb_Assay_reg import MetaDataset
from dataset.data_base import SystemDataLoader


def dataset_constructor(args):
    datasource = args.datasource
    if datasource in ["chembl", "bdb", "bdb_ic50", "esol", "lipo", "freesolv", "qm9",
                        "bace", "bbbp", "hiv", "muv", "clintox", "tox21", "toxcast", "sider"]:
        dataset = MetaDataset
    else:
        raise ValueError(f"model {datasource} is not supported")

    return SystemDataLoader(args, dataset)