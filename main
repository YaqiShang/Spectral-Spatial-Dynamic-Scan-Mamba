from __future__ import print_function
from __future__ import division
import torch.utils.data as data
import numpy as np
import seaborn as sns
import visdom
import os
import cv2
import json
from utils import (
    metrics,
    convert_to_color_,
    convert_from_color_,
    display_predictions,
    explore_spectrums,
    plot_spectrums,
    sample_gt,
    show_results,
    compute_imf_weights,
    get_device,
    seed_torch,
)
from datasets import get_dataset, MultiModalX, open_file, DATASETS_CONFIG
from model_utils import get_model, train
import argparse


import time

import torch

def convert_numpy_to_json_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_to_json_serializable(item) for item in obj]
    else:
        return obj

def truly_optimized_test(model, img1, img2, test_gt, hyperparams):
    device = hyperparams["device"]
    patch_size = hyperparams["patch_size"]
    batch_size = hyperparams["batch_size"]
    n_classes = hyperparams["n_classes"]

    test_indices = np.nonzero(test_gt)
    n_test_samples = len(test_indices[0])
    predictions = np.zeros(test_gt.shape)
    model.eval()
    half_patch = patch_size // 2
    padded_img1 = np.pad(img1, ((half_patch, half_patch), (half_patch, half_patch), (0, 0)), mode='symmetric')
    padded_img2 = np.pad(img2, ((half_patch, half_patch), (half_patch, half_patch), (0, 0)), mode='symmetric')
    for batch_start in range(0, n_test_samples, batch_size):
        batch_end = min(batch_start + batch_size, n_test_samples)
        current_batch_size = batch_end - batch_start

        batch_patches1 = np.zeros((current_batch_size, patch_size, patch_size, img1.shape[2]))
        batch_patches2 = np.zeros((current_batch_size, patch_size, patch_size, img2.shape[2]))
        batch_indices = []

        for i in range(current_batch_size):
            x, y = test_indices[0][batch_start + i], test_indices[1][batch_start + i]
            batch_indices.append((x, y))

            x_pad, y_pad = x + half_patch, y + half_patch
            batch_patches1[i] = padded_img1[x_pad - half_patch:x_pad + half_patch + 1,
                                y_pad - half_patch:y_pad + half_patch + 1]
            batch_patches2[i] = padded_img2[x_pad - half_patch:x_pad + half_patch + 1,
                                y_pad - half_patch:y_pad + half_patch + 1]

        tensor_patches1 = torch.from_numpy(batch_patches1.transpose(0, 3, 1, 2)).float().to(device)
        tensor_patches2 = torch.from_numpy(batch_patches2.transpose(0, 3, 1, 2)).float().to(device)

        with torch.no_grad():
            outputs = model(tensor_patches1, tensor_patches2)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            _, predicted = torch.max(outputs, 1)
            predicted = predicted.cpu().numpy()

            for i, (x, y) in enumerate(batch_indices):
                predictions[x, y] = predicted[i]

    test_mask = test_gt > 0
    masked_predictions = predictions[test_mask]
    masked_gt = test_gt[test_mask]

    epoch_results = metrics(
        masked_predictions,
        masked_gt,
        ignored_labels=hyperparams["ignored_labels"],
        n_classes=n_classes,
    )
    return predictions, epoch_results

best_oa = 0
best_oa_epoch = 0
best_oa_aa = 0
best_oa_kappa = 0
def test_callback_function(current_model, current_epoch, save_dir=None):
    global save_path, best_oa, best_oa_epoch, best_oa_aa, best_oa_kappa 

    if save_dir is None:
        if 'save_path' in globals():
            save_dir = save_path
        else:
            save_dir = f"./results/{DATASET}/{MODEL}"
            os.makedirs(save_dir, exist_ok=True)

    test_start_time = time.time()
    test_prediction, epoch_results = truly_optimized_test(
        current_model, img1, img2, test_gt, hyperparams
    )
    if current_epoch == hyperparams["test_interval"]:
        print("Available keys in epoch_results:", list(epoch_results.keys()))
    current_oa = epoch_results['Accuracy']
    if 'Average accuracy' in epoch_results:
        current_aa = epoch_results['Average accuracy']
    elif 'Average_accuracy' in epoch_results:
        current_aa = epoch_results['Average_accuracy']
    elif 'AA' in epoch_results:
        current_aa = epoch_results['AA']
    else:
        class_acc = [epoch_results[f'Class {i}'] for i in range(hyperparams['n_classes']) if
                     f'Class {i}' in epoch_results]
        current_aa = sum(class_acc) / len(class_acc) if class_acc else 0
    if 'Kappa' in epoch_results:
        current_kappa = epoch_results['Kappa']
    elif 'kappa' in epoch_results:
        current_kappa = epoch_results['kappa']
    else:
        current_kappa = 0
        print("Warning: Kappa metric not found in results")

    if current_oa > best_oa:
        best_oa = current_oa
        best_oa_epoch = current_epoch
        best_oa_aa = current_aa
        best_oa_kappa = current_kappa

    mask = np.zeros(gt.shape, dtype="bool")
    for l in IGNORED_LABELS:
        mask[gt == l] = True
    test_prediction[mask] = 0

    color_test_prediction = convert_to_color(test_prediction)
    display_predictions(
        color_test_prediction,
        viz,
        gt=convert_to_color(test_gt),
        caption=f"Epoch {current_epoch} - 预测结果与测试真值对比",
    )

    show_results(epoch_results, viz, label_values=LABEL_VALUES)

    epoch_save_path = os.path.join(save_dir, f"epoch_{current_epoch}")
    os.makedirs(epoch_save_path, exist_ok=True)

    cv2.imwrite(
        os.path.join(epoch_save_path, "prediction.png"),
        cv2.cvtColor(color_test_prediction, cv2.COLOR_RGB2BGR)
    )

    metrics_save_path = os.path.join(epoch_save_path, "metrics.json")
    json_serializable_results = convert_numpy_to_json_serializable(epoch_results)
    with open(metrics_save_path, 'w') as f:
        json.dump(json_serializable_results, f, indent=4)

    test_time = time.time() - test_start_time
    print(f"Epoch {current_epoch} 测试耗时: {test_time:.2f}秒")
    print(f"Epoch {current_epoch} 的测试结果已保存至 {epoch_save_path}")
    print(
        f"当前最高的OA: {best_oa:.4f}，出现在第 {best_oa_epoch} 轮，对应的AA: {best_oa_aa:.4f}，Kappa: {best_oa_kappa:.4f}")

start_time = time.time()

dataset_names = [
    v["name"] if "name" in v.keys() else k for k, v in DATASETS_CONFIG.items()
]

parser = argparse.ArgumentParser(
    description="Run deep learning experiments on" " various hyperspectral datasets"
)
parser.add_argument(
    "--dataset", type=str, default='MUUFL', choices=dataset_names, help="Dataset to use."
)
parser.add_argument(
    "--folder",
    type=str,
    help="Folder where to store the "
    "datasets (defaults to the current working directory).",
    default="./Datasets/",
)
parser.add_argument(
    "--cuda",
    type=int,
    default=0,
    help="Specify CUDA device (defaults to -1, which learns on CPU)",
)
parser.add_argument("--runs", type=int, default=1, help="Number of runs (default: 1)")
parser.add_argument(
    "--restore",
    type=str,
    default=None,
    help="Weights to use for initialization, e.g. a checkpoint",
)
parser.add_argument(
    "--seed",
    type=int,
    default=0,
    help="Set random seed",
)

# Dataset options
group_dataset = parser.add_argument_group("Dataset")
group_dataset.add_argument(
    "--train_val_split",
    type=float,
    default=1,
    help="Percentage of samples to use for training and validation, "
         "'1' means all training data are used to train",
)
group_dataset.add_argument(
    "--training_sample",
    type=float,
    default=10,
    help="Percentage of samples to use for training (default: 10%%) and testing",
)
group_dataset.add_argument(
    "--sampling_mode",
    type=str,
    help="Sampling mode" " (random sampling or disjoint, default: random)",
    default="random",
)
group_dataset.add_argument(
    "--train_set",
    type=str,
    default='dataset/MUUFL/muufl_tr.mat',
    help="Path to the train ground truth (optional, this "
    "supersedes the --sampling_mode option)",
)
group_dataset.add_argument(
    "--test_set",
    type=str,
    default='dataset/MUUFL/muufl_ts.mat',
    help="Path to the test set (optional, by default "
    "the test_set is the entire ground truth minus the training)",
)
# Training options
group_train = parser.add_argument_group("Training")
group_train.add_argument(
    "--epoch",
    type=int,
    default='100',
    help="Training epochs (optional, if" " absent will be set by the model)",
)
group_train.add_argument(
    "--patch_size",
    type=int,
    default='7',
    help="Size of the spatial neighbourhood (optional, if "
    "absent will be set by the model)",
)
group_train.add_argument(
    "--lr", type=float, default='0.0001', help="Learning rate, set by the model if not specified."
)
group_train.add_argument(
    "--class_balancing",
    action="store_true",
    help="Inverse median frequency class balancing (default = False)",
)
group_train.add_argument(
    "--batch_size",
    type=int,
    default='64',
    help="Batch size (optional, if absent will be set by the model",
)
group_train.add_argument(
    "--test_stride",
    type=int,
    default=1,
    help="Sliding window step stride during inference (default = 1)",
)
# Data augmentation parameters
group_da = parser.add_argument_group("Data augmentation")
group_da.add_argument(
    "--flip_augmentation", action="store_true", help="Random flips (if patch_size > 1)"
)
group_da.add_argument(
    "--radiation_augmentation",
    action="store_true",
    help="Random radiation noise (illumination)",
)
group_da.add_argument(
    "--mixture_augmentation", action="store_true", help="Random mixes between spectra"
)

parser.add_argument(
    "--with_exploration", action="store_true", help="See data exploration visualization"
)
parser.add_argument(
    "--download",
    type=str,
    default=None,
    nargs="+",
    choices=dataset_names,
    help="Download the specified datasets and quits.",
)
parser.add_argument(
    "--model",
    type=str,
    default='SDSM',
)
group_train.add_argument(
    "--test_interval",
    type=int,
    default=20,
)

args = parser.parse_args()

CUDA_DEVICE = get_device(args.cuda)

SAMPLE_PERCENTAGE = args.training_sample
SAMPLE_TRAIN_VALID = args.train_val_split
FLIP_AUGMENTATION = args.flip_augmentation
RADIATION_AUGMENTATION = args.radiation_augmentation
MIXTURE_AUGMENTATION = args.mixture_augmentation
DATASET = "Houston2013" if args.dataset is None else args.dataset
MODEL = args.model
N_RUNS = args.runs
PATCH_SIZE = args.patch_size
DATAVIZ = args.with_exploration
FOLDER = args.folder
EPOCH = args.epoch
SAMPLING_MODE = args.sampling_mode
CHECKPOINT = args.restore
LEARNING_RATE = args.lr
CLASS_BALANCING = args.class_balancing
TRAIN_GT = args.train_set
TEST_GT = args.test_set
TEST_STRIDE = args.test_stride
TEST_INTERVAL = args.test_interval

seed_torch(seed=args.seed)

if args.download is not None and len(args.download) > 0:
    for dataset in args.download:
        get_dataset(dataset, target_folder=FOLDER)
    quit()

viz = visdom.Visdom(env=DATASET + " " + MODEL)
if not viz.check_connection:
    print("Visdom is not connected. Did you run 'python -m visdom.server' ?")


hyperparams = vars(args)
img1, img2, gt, LABEL_VALUES, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(DATASET, FOLDER)


if palette is None:
    palette = {0: (0, 0, 0)}
    for k, color in enumerate(sns.color_palette("hls", len(LABEL_VALUES) - 1)):
        palette[k + 1] = tuple(np.asarray(255 * np.array(color), dtype="uint8"))
invert_palette = {v: k for k, v in palette.items()}


def convert_to_color(x):
    return convert_to_color_(x, palette=palette)

def convert_from_color(x):
    return convert_from_color_(x, palette=invert_palette)

if DATAVIZ:
    mean_spectrums = explore_spectrums(
        img1, gt, LABEL_VALUES, viz, ignored_labels=IGNORED_LABELS
    )
    plot_spectrums(mean_spectrums, viz, title="Mean spectrum/class")

N_CLASSES = len(LABEL_VALUES)
N_BANDS = (img1.shape[-1], img2.shape[-1])

hyperparams.update(
    {
        "n_classes": N_CLASSES,
        "n_bands": N_BANDS,
        "ignored_labels": IGNORED_LABELS,
        "device": CUDA_DEVICE,
    }
)
hyperparams = dict((k, v) for k, v in hyperparams.items() if v is not None)

results = []
for run in range(N_RUNS):
    if TRAIN_GT is not None and TEST_GT is not None:
        if DATASET == 'Augsburg' or DATASET == 'Berlin':
            train_gt = open_file(TRAIN_GT)['TrainImage']
            test_gt = open_file(TEST_GT)['TestImage']
        elif DATASET == 'Houston2013' or DATASET == 'YR' or DATASET == 'Italy':
            train_gt = open_file(TRAIN_GT)['Tr']
            test_gt = open_file(TEST_GT)['Te']
        elif DATASET == 'MUUFL':
            train_gt = open_file(TRAIN_GT)['training_map']
            test_gt = open_file(TEST_GT)['testing_map']
    elif TRAIN_GT is not None:
        train_gt = open_file(TRAIN_GT)
        test_gt = np.copy(gt)
        w, h = test_gt.shape
        test_gt[(train_gt > 0)[:w, :h]] = 0
    elif TEST_GT is not None:
        test_gt = open_file(TEST_GT)
    else:
        train_gt, test_gt = sample_gt(gt, SAMPLE_PERCENTAGE, mode=SAMPLING_MODE)
    print(
        "{} samples selected (over {})".format(
            np.count_nonzero(train_gt), np.count_nonzero(gt)
        )
    )
    print(
        "Running an experiment with the {} model".format(MODEL),
        "run {}/{}".format(run + 1, N_RUNS),
    )

    display_predictions(convert_to_color(train_gt), viz, caption="Train ground truth")
    display_predictions(convert_to_color(test_gt), viz, caption="Test ground truth")

    if CLASS_BALANCING:
        weights = compute_imf_weights(train_gt, N_CLASSES, IGNORED_LABELS)
        hyperparams["weights"] = torch.from_numpy(weights)
    model, optimizer, loss, hyperparams = get_model(MODEL, **hyperparams)

    if SAMPLE_TRAIN_VALID != 1:
        train_gt, val_gt = sample_gt(train_gt, SAMPLE_TRAIN_VALID, mode="random")
    else:
        _, val_gt = sample_gt(train_gt, 0.95, mode="random")

    train_dataset = MultiModalX(img1, img2, train_gt, **hyperparams)
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=hyperparams["batch_size"],
        shuffle=True,
    )
    val_dataset = MultiModalX(img1, img2, val_gt, **hyperparams)
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=hyperparams["batch_size"],
    )

    print(hyperparams)
    print("Network :")
    with torch.no_grad():
        for input, input2,  _ in train_loader:
            break
        print("Input 1", input.size())
        print("Input 2", input2.size())
    if CHECKPOINT is not None:
        model.load_state_dict(torch.load(CHECKPOINT))

    train_start_time = time.time()

    save_path = f"./results/{DATASET}/{MODEL}"
    os.makedirs(save_path, exist_ok=True)
    try:
        train(
            model,
            optimizer,
            loss,
            train_loader,
            hyperparams["epoch"],
            scheduler=hyperparams["scheduler"],
            device=hyperparams["device"],
            supervision=hyperparams["supervision"],
            val_loader=val_loader,
            display=viz,
            test_interval=TEST_INTERVAL,
            test_callback=lambda model, epoch: test_callback_function(model, epoch, save_path),
        )
    except KeyboardInterrupt:
        pass

    test_prediction, epoch_results = truly_optimized_test(
        model, img1, img2, test_gt, hyperparams
    )
    prediction = test_prediction
    run_results = epoch_results




    mask = np.zeros(gt.shape, dtype="bool")
    for l in IGNORED_LABELS:
        mask[gt == l] = True
    prediction[mask] = 0

    color_prediction = convert_to_color(prediction)
    display_predictions(
        color_prediction,
        viz,
        gt=convert_to_color(test_gt),
        caption="Prediction vs. test ground truth",
    )

    flt_test_gt = test_gt.reshape(-1)
    flt_prediction = prediction.reshape(-1)

    results.append(run_results)
    show_results(run_results, viz, label_values=LABEL_VALUES)

if N_RUNS > 1:
    show_results(results, viz, label_values=LABEL_VALUES, agregated=True)

