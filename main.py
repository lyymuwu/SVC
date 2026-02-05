import os
from utils import GPU_Search
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_Search())
import numpy as np
import torch
import time
import copy
import sys
sys.path.append('src/')
from src.task_vectors import TaskVector
from src.eval import eval_single_dataset
from src.args import parse_arguments
from utils import *
from merge_func import *

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)


exam_datasets = ['SUN397', 'Cars', 'RESISC45', 'EuroSAT', 'SVHN', 'GTSRB', 'MNIST', 'DTD']
train_datasets = exam_datasets
eval_datasets = exam_datasets
# model = 'ViT-B-32' #'ViT-B-16' #'ViT-B-32' # 'ViT-L-14'
args = parse_arguments()
args.data_location = os.path.join(args.base_dir, "data")
args.save = os.path.join(args.base_dir, "checkpoints", args.model)
args.logs_path = 'logs/' + args.model
args.pretrained_checkpoint = os.path.join(args.base_dir, "checkpoints", args.model, 'zeroshot.pt') 
args.scaling_coef = 1
args.DATASETS = exam_datasets
args.Target = range(len(train_datasets))
# args.merge = "TSV-M"  
# args.alpha = 0.1
# args.calibrate_flag = True

str_time_ = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
log = create_log_dir(args.logs_path, 'log_{}_{}_mainV2.txt'.format(str_time_, args.merge))
log.info("Merge method: {}, ".format(args.merge))
log.info("Configure: {}".format(args))
starttime = time.time()

################################################################################
task_vectors = [
    TaskVector(
        args.pretrained_checkpoint,
        os.path.join(args.base_dir, "checkpoints", args.model, dataset_name, "finetuned.pt")
    ) for dataset_name in train_datasets
]
task_vector_avg = copy.deepcopy(sum(task_vectors))  * (1/len(task_vectors))


# Merging methods
merge_methods = {
    "TA": TA, "TIES": layer_wise_TIES, "DARE": DARE, "TSV-M": TSVM, 
    "Iso-C": ISO_C, "Iso-CTS": ISO_CTS, "STAR": STAR,
} 
merge_methods[args.merge](task_vector_avg, task_vectors, args)


if args.calibrate_flag:
    args.right_only = False
    layer_wise_Align(task_vector_avg, task_vectors, args)


image_encoder = task_vector_avg.apply_to(args.pretrained_checkpoint, scaling_coef=args.scaling_coef)
log.info('*'*20 + 'Merge Method:' + str(args.merge) + '*'*20)

accs = []
for dataset in eval_datasets:
    metrics = eval_single_dataset(image_encoder, dataset, args)
    log.info(str(dataset) + ':' + str(metrics.get('top1')*100)+'%')
    accs.append(metrics.get('top1')*100)
log.info('Avg ACC:' + str(round(np.mean(accs),2)) + '%')
log.info('Time:' + str(time.time()-starttime))