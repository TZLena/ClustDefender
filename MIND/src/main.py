import random
import numpy as np

import torch
import os
import logging
from pathlib import Path

from task import get_task
from config import job_config
from parameters import parse_args, setuplogger
from MIND.preprocess.preprocess import *

data_root_path = '../raw/mind'

if __name__ == "__main__":

    print("start main")

    args = parse_args()
    args.seed = 100
    #args.lr = 1e-5
    #args.max_train_steps = 100
    #args.mal_user_ratio = 0.05
    #args.validation_steps = 50
    #args.max_train_steps = 100
    #args.user_num = 100
    #args.lr = 0.0001
    #args.mode = "test"

    out_path = Path(args.out_path) / f"{args.run_name}-{args.data}"
    out_path.mkdir(exist_ok=True, parents=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    setuplogger(args, out_path)
    logging.info(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device(f"cuda:0")
    torch.cuda.set_device(device)

    args.project_name = "ua-100"
    args.job_name= "AdverItemNorm-FABA"
    args.run_name = "faba1021"

    config = job_config[args.job_name]
    logging.info(f"[-] load config of {args.job_name}")
    logging.info(config)

    if args.mode == "train":
       task_cls = get_task(config["train_task_name"])
    else:
       task_cls = get_task(config["test_task_name"])

    task = task_cls(args, config, device)
    task.start()
