import os
import json
import tqdm

import torch
import torch.nn.functional as F

import utils
from src.datasets.common import get_dataloader, get_dataloader_shuffle, maybe_dictionarize
from src.heads import get_classification_head
from src.modeling import ImageClassifier

from src.datasets.registry import get_dataset

def eval_single_dataset(image_encoder, dataset_name, args, use_train=False, 
        no_print=False, use_shuffle_test=False, use_val=False, val_test=False, 
        quick_iter=False, constrain_batch_size=None, mean_batch=False, seed=42):
    classification_head = get_classification_head(args, dataset_name)
    model = ImageClassifier(image_encoder, classification_head)

    model.eval()

    dataset = get_dataset(
        dataset_name,
        model.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
        num_workers=12,
        use_val=use_val,
        seed=seed
    )
    
    if use_shuffle_test:
        dataloader = get_dataloader_shuffle(dataset)
    else:
        dataloader = get_dataloader(
            dataset, is_train=use_train, args=args, image_encoder=None, use_val=use_val, val_test=val_test)
    
    device = args.device

    with torch.no_grad():
        top1, correct, n = 0., 0., 0.
        total_loss = 0.
        for i, data in enumerate(tqdm.tqdm(dataloader)):
            data = maybe_dictionarize(data)
            x = data['images'].to(device)
            y = data['labels'].to(device) # [batch_size]
            if mean_batch:
                x = x.mean(dim=0, keepdim=True)
                y = y[:1]
            elif constrain_batch_size is not None:
                x = x[:constrain_batch_size]
                y = y[:constrain_batch_size]

            logits = utils.get_logits(x, model) # [batch_size, num_classes]
            loss = F.cross_entropy(logits, y, reduction='sum')

            pred = logits.argmax(dim=1, keepdim=True).to(device)

            correct += pred.eq(y.view_as(pred)).sum().item()
            total_loss += loss.item()
            
            n += y.size(0)
            
            if quick_iter:
                break

        top1 = correct / n
        loss = total_loss / n

    metrics = {'top1': top1}
    metrics['loss'] = loss
    if not no_print:
        print(f'Done evaluating on {dataset_name}. Accuracy: {round(100*top1,2)}')
    
    return metrics


def eval_single_dataset_head(image_encoder, head, dataset_name, args):
    model = ImageClassifier(image_encoder, head)

    model.eval()

    dataset = get_dataset(dataset_name, model.val_preprocess, location=args.data_location,  batch_size=args.batch_size)
    dataloader = get_dataloader(dataset, is_train=False, args=args, image_encoder=None)
    device = args.device

    with torch.no_grad():
        top1, correct, n = 0., 0., 0.
        for i, data in enumerate(tqdm.tqdm(dataloader)):
            data = maybe_dictionarize(data)
            x = data['images'].to(device)
            y = data['labels'].to(device)

            logits = utils.get_logits(x, model)

            pred = logits.argmax(dim=1, keepdim=True).to(device)

            correct += pred.eq(y.view_as(pred)).sum().item()

            n += y.size(0)

        top1 = correct / n

    metrics = {'top1': top1}
    print(f'Done evaluating on {dataset_name}. Accuracy: {100 * top1:.2f}%')

    return metrics

def eval_single_dataset_preprocess_head(image_encoder, head, dataset_name, args):
    model = ImageClassifier(image_encoder, head)

    model.eval()

    dataset = get_dataset(dataset_name, model.val_preprocess, 
        location=args.data_location,  batch_size=args.batch_size, num_workers=16)
    dataloader = get_dataloader(dataset, is_train=False, args=args, image_encoder=None)
    device = args.device

    with torch.no_grad():
        top1, correct, n = 0., 0., 0.
        for i, data in enumerate(tqdm.tqdm(dataloader)):
            data = maybe_dictionarize(data)
            x = data['images'].to(device)
            y = data['labels'].to(device)

            logits = utils.get_logits(x, model)

            pred = logits.argmax(dim=1, keepdim=True).to(device)

            correct += pred.eq(y.view_as(pred)).sum().item()

            n += y.size(0)

        top1 = correct / n

    metrics = {'top1': top1}
    print(f'Done evaluating on {dataset_name}. Accuracy: {100 * top1:.2f}%')

    return metrics

def evaluate(image_encoder, args):
    if args.eval_datasets is None:
        return
    info = vars(args)
    for i, dataset_name in enumerate(args.eval_datasets):
        print('Evaluating on', dataset_name)

        results = eval_single_dataset(image_encoder, dataset_name, args)

        if 'top1' in results:
            print(f"{dataset_name} Top-1 accuracy: {results['top1']:.4f}")
        for key, val in results.items():
            if 'worst' in key or 'f1' in key.lower() or 'pm0' in key:
                print(f"{dataset_name} {key}: {val:.4f}")
            info[dataset_name + ':' + key] = val

    if args.results_db is not None:
        dirname = os.path.dirname(args.results_db)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(args.results_db, 'a+') as f:
            f.write(json.dumps(info) + '\n')
        print(f'Results saved to {args.results_db}.')
    else:
        print('Results not saved (to do so, use --results_db to specify a path).')

    return info
