job_config = {}


def add_config(
    name,
    train_task_name,
    model_name,
    agg_name,
    train_dataset_name,
    train_collate_fn_name=None,
    test_task_name="TestTask",
    news_dataset_name="NewsDataset",
    user_dataset_name="UserDataset",
):
    job_config[name] = {}
    job_config[name]["train_task_name"] = train_task_name
    job_config[name]["model_name"] = model_name
    job_config[name]["agg_name"] = agg_name
    job_config[name]["train_dataset_name"] = train_dataset_name
    job_config[name]["train_collate_fn_name"] = train_collate_fn_name
    job_config[name]["test_task_name"] = test_task_name
    job_config[name]["news_dataset_name"] = news_dataset_name
    job_config[name]["user_dataset_name"] = user_dataset_name


add_config(
    name="AdverItemNorm-UNIFEDREC",
    train_task_name="AdverItemNormTrainTask",
    model_name="Unifedrec",
    agg_name="UserAggregator",
    train_dataset_name="TrainNewDataset",  # empty dataset
    train_collate_fn_name="train_mp_collate_fn",
)

add_config(
    name="AdverItemNorm",
    train_task_name="AdverItemNormTrainTask",
    model_name="NRMS",
    agg_name="UserAggregator",
    train_dataset_name="TrainMPDataset",
    train_collate_fn_name="train_mp_collate_fn",
)

add_config(
    name="AdverItemNorm-MultiKrum",
    train_task_name="AdverItemNormTrainTask",
    model_name="NRMS",
    agg_name="MultiKrumAggregator",
    train_dataset_name="TrainMPDataset",
    train_collate_fn_name="train_mp_collate_fn",
)
add_config(
    name="AdverItemNorm-NormBound",
    train_task_name="AdverItemNormTrainTask",
    model_name="NRMS",
    agg_name="NormBoundAggregator",
    train_dataset_name="TrainMPDataset",
    train_collate_fn_name="train_mp_collate_fn",
)

add_config(
    name="AdverItemNorm-FLTrust",
    train_task_name="AdverItemNormTrainTask",
    model_name="NRMS",
    agg_name="FLTrust",
    train_dataset_name="TrainMPDataset",
    train_collate_fn_name="train_mp_collate_fn",
)
add_config(
    name="AdverItemNorm-FABA",
    train_task_name="AdverItemNormTrainTask",
    model_name="NRMS",
    agg_name="FABA",
    train_dataset_name="TrainMPDataset",
    train_collate_fn_name="train_mp_collate_fn",
)
add_config(
    name="AdverItemNorm-FoolsGold",
    train_task_name="AdverItemNormTrainTask",
    model_name="NRMS",
    agg_name="FoolsGold",
    train_dataset_name="TrainMPDataset",
    train_collate_fn_name="train_mp_collate_fn",
)
add_config(
    name="AdverItemNorm-FLAIR",
    train_task_name="AdverItemNormTrainTask",
    model_name="NRMS",
    agg_name="FLAIR",
    train_dataset_name="TrainMPDataset",
    train_collate_fn_name="train_mp_collate_fn",
)
add_config(
    name="AdverNewsNorm-FLTrust",
    train_task_name="AdverNewsNormTrainTask",
    model_name="NRMS",
    agg_name="FLTrust",
    train_dataset_name="TrainMPDataset",
    train_collate_fn_name="train_mp_collate_fn",
)

add_config(
    name="User-FLTrust",
    train_task_name="UserTrainTask",
    model_name="NRMS",
    agg_name="FLTrust",
    train_dataset_name="TrainMPDataset",
    train_collate_fn_name="train_mp_collate_fn",
)

add_config(
    name="User",
    train_task_name="UserTrainTask",
    model_name="NRMS",
    agg_name="UserAggregator",
    train_dataset_name="TrainMPDataset",
    train_collate_fn_name="train_mp_collate_fn",
)

add_config(
    name="news",
    train_task_name="AdverNewsNormTrainTask",
    model_name="NRMS",
    agg_name="UserAggregator",
    train_dataset_name="TrainMPDataset",
    train_collate_fn_name="train_mp_collate_fn",
)

add_config(
    name="SVM",
    train_task_name="AdverItemNormTrainTask",
    model_name="NRMS",
    agg_name="SVMAggregator",
    train_dataset_name="TrainMPDataset",
    train_collate_fn_name="train_mp_collate_fn",
)

add_config(
    name="cluster",
    train_task_name="AdverItemNormTrainTask",
    model_name="NRMS",
    agg_name="ClustAggregator",
    train_dataset_name="TrainMPDataset",
    train_collate_fn_name="train_mp_collate_fn",
)

add_config(
    name="w/o news",
    train_task_name="WONewsTask",
    model_name="NRMS",
    agg_name="UserAggregator",
    train_dataset_name="TrainMPDataset",
    train_collate_fn_name="train_mp_collate_fn",
)

add_config(
    name="w/o user",
    train_task_name="WOUserTask",
    model_name="NRMS",
    agg_name="UserAggregator",
    train_dataset_name="TrainMPDataset",
    train_collate_fn_name="train_mp_collate_fn",
)


# ========================== LSTUR ===================================
add_config(
    name="AdverItemNorm-LSTUR",
    train_task_name="AdverItemNormTrainTask",
    model_name="LSTUR",
    agg_name="UserAggregator",
    train_dataset_name="TrainMPDataset",
    train_collate_fn_name="train_mp_collate_fn",
)
add_config(
    name="AdverItemNorm-MultiKrum-LSTUR",
    train_task_name="AdverItemNormTrainTask",
    model_name="LSTUR",
    agg_name="MultiKrumAggregator",
    train_dataset_name="TrainMPDataset",
    train_collate_fn_name="train_mp_collate_fn",
)
add_config(
    name="AdverItemNorm-NormBound-LSTUR",
    train_task_name="AdverItemNormTrainTask",
    model_name="LSTUR",
    agg_name="NormBoundAggregator",
    train_dataset_name="TrainMPDataset",
    train_collate_fn_name="train_mp_collate_fn",
)

if __name__ == "__main__":
    print(job_config)
