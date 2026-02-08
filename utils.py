import random
import numpy as np
from sklearn.metrics import confusion_matrix
import sklearn.model_selection
import itertools
import spectral
import visdom
import matplotlib.pyplot as plt
from scipy import io, misc
import os
import re
import torch


def get_device(ordinal):
    if ordinal < 0:
        print("Computation on CPU")
        device = torch.device('cpu')
    elif torch.cuda.is_available():
        print("Computation on CUDA GPU device {}".format(ordinal))
        device = torch.device('cuda:{}'.format(ordinal))
    else:
        print("/!\\ CUDA was requested but is not available! Computation will go on CPU. /!\\")
        device = torch.device('cpu')
    return device


def open_file(dataset):
    _, ext = os.path.splitext(dataset)
    ext = ext.lower()
    if ext == '.mat':
        return io.loadmat(dataset)
    elif ext == '.tif' or ext == '.tiff':
        return misc.imread(dataset)
    elif ext == '.hdr':
        img = spectral.open_image(dataset)
        return img.load()
    else:
        raise ValueError("Unknown file format: {}".format(ext))

def convert_to_color_(arr_2d, palette=None):
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)
    if palette is None:
        raise Exception("Unknown color palette")

    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d


def convert_from_color_(arr_3d, palette=None):
    if palette is None:
        raise Exception("Unknown color palette")

    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i

    return arr_2d


def display_predictions(pred, vis, gt=None, caption=""):
    if gt is None:
        vis.images([np.transpose(pred, (2, 0, 1))],
                    opts={'caption': caption})
    else:
        vis.images([np.transpose(pred, (2, 0, 1)),
                    np.transpose(gt, (2, 0, 1))],
                    nrow=2,
                    opts={'caption': caption})

def display_dataset(img, gt, bands, labels, palette, vis):
    print("Image 1 has dimensions {}x{} and {} channels".format(*img.shape))
    rgb = spectral.get_rgb(img, bands)
    rgb /= np.max(rgb)
    rgb = np.asarray(255 * rgb, dtype='uint8')
    caption = "RGB (bands {}, {}, {})".format(*bands)
    vis.images([np.transpose(rgb, (2, 0, 1))],
                opts={'caption': caption})

def display_lidar_data(img, vis):
    print("Image 2 has dimensions {}x{} and {} channels".format(*img.shape))
    gray = img / np.max(img)
    gray = np.asarray(255 * gray, dtype='uint8')
    caption = "LiDAR"
    vis.images([np.transpose(gray, (2, 0, 1))],
               opts={'caption': caption})


def explore_spectrums(img, complete_gt, class_names, vis,
                      ignored_labels=None):
    mean_spectrums = {}
    for c in np.unique(complete_gt):
        if c in ignored_labels:
            continue
        mask = complete_gt == c
        class_spectrums = img[mask].reshape(-1, img.shape[-1])
        step = max(1, class_spectrums.shape[0] // 100)
        fig = plt.figure()
        plt.title(class_names[c])

        for spectrum in class_spectrums[::step, :]:
            plt.plot(spectrum, alpha=0.25)
        mean_spectrum = np.mean(class_spectrums, axis=0)
        std_spectrum = np.std(class_spectrums, axis=0)
        lower_spectrum = np.maximum(0, mean_spectrum - std_spectrum)
        higher_spectrum = mean_spectrum + std_spectrum

        plt.fill_between(range(len(mean_spectrum)), lower_spectrum,
                         higher_spectrum, color="#3F5D7D")
        plt.plot(mean_spectrum, alpha=1, color="#FFFFFF", lw=2)
        vis.matplot(plt)
        mean_spectrums[class_names[c]] = mean_spectrum
    return mean_spectrums


def plot_spectrums(spectrums, vis, title=""):
    win = None
    for k, v in spectrums.items():
        n_bands = len(v)
        update = None if win is None else 'append'
        win = vis.line(X=np.arange(n_bands), Y=v, name=k, win=win, update=update,
                       opts={'title': title})


def build_dataset(mat, gt, ignored_labels=None):
    samples = []
    labels = []
    assert mat.shape[:2] == gt.shape[:2]

    for label in np.unique(gt):
        if label in ignored_labels:
            continue
        else:
            indices = np.nonzero(gt == label)
            samples += list(mat[indices])
            labels += len(indices[0]) * [label]
    return np.asarray(samples), np.asarray(labels)


def get_random_pos(img, window_shape):
    w, h = window_shape
    W, H = img.shape[:2]
    x1 = random.randint(0, W - w - 1)
    x2 = x1 + w
    y1 = random.randint(0, H - h - 1)
    y2 = y1 + h
    return x1, x2, y1, y2


def padding_image(image, patch_size=None, mode="symmetric", constant_values=None):
    if patch_size is None:
        patch_size = [1, 1]
    h = patch_size[0] // 2
    w = patch_size[1] // 2
    pad_width = [[h, h], [w, w]]
    [pad_width.append([0, 0]) for i in image.shape[2:]]
    padded_image = np.pad(image, pad_width, mode=mode)
    return padded_image

def restore_from_padding(image, patch_size=None):
    if patch_size is None:
        patch_size = [1, 1]
    h = patch_size[0] // 2
    w = patch_size[1] // 2
    W, H = image.shape[:2]
    restore_img = image[w:W-w, h:H-h, :]

    return restore_img



def sliding_window(image1, image2, step=10, window_size=(20, 20), with_data=True):
    w, h = window_size
    W, H = image1.shape[:2]
    offset_w = (W - w) % step
    offset_h = (H - h) % step
    for x in range(0, W - w + offset_w + 1, step):
        if x + w > W:
            x = W - w
        for y in range(0, H - h + offset_h + 1, step):
            if y + h > H:
                y = H - h
            if with_data:
                yield image1[x:x + w, y:y + h], image2[x:x + w, y:y + h], x, y, w, h
            else:
                yield x, y, w, h


def count_sliding_window(top, top2, step=10, window_size=(20, 20)):
    sw = sliding_window(top, top2, step, window_size, with_data=False)
    return sum(1 for _ in sw)


def grouper(n, iterable):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def metrics(prediction, target, ignored_labels=[], n_classes=None):
    ignored_mask = np.zeros(target.shape[:2], dtype=bool)
    for l in ignored_labels:
        ignored_mask[target == l] = True
    ignored_mask = ~ignored_mask
    target = target[ignored_mask]
    prediction = prediction[ignored_mask]

    results = {}

    n_classes = np.max(target) + 1 if n_classes is None else n_classes

    cm = confusion_matrix(
        target,
        prediction,
        labels=range(n_classes))

    results["Confusion matrix"] = cm
    total = np.sum(cm)
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy *= 100 / float(total)

    results["Accuracy"] = accuracy

    F1scores = np.zeros(len(cm))
    for i in range(len(cm)):
        try:
            F1 = 2. * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
        except ZeroDivisionError:
            F1 = 0.
        F1scores[i] = F1

    results["F1 scores"] = F1scores

    Precisions = np.zeros(len(cm))
    for i in range(len(cm)):
        try:
            Precision = 1. * cm[i, i] / np.sum(cm[i, :])
        except ZeroDivisionError:
            Precision = 0.
        Precisions[i] = Precision

    results["Precisions"] = Precisions

    AAs = []
    for i in range(len(cm)):
        try:
            recall = cm[i][i] / np.sum(cm[i, :])
            if np.isnan(recall):
                continue
        except ZeroDivisionError:
            recall = 0.
        AAs.append(recall)
    results['AA'] = np.mean(AAs)

    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / \
        float(total * total)
    kappa = (pa - pe) / (1 - pe)
    results["Kappa"] = kappa

    return results


def show_results(results, vis, label_values=None, agregated=False):
    text = ""

    if agregated:
        accuracies = [r["Accuracy"] for r in results]
        AAs = [r["AA"] for r in results]
        kappas = [r["Kappa"] for r in results]
        F1_scores = [r["F1 scores"] for r in results]
        Precisions = [r["Precisions"] for r in results]

        F1_scores_mean = np.mean(F1_scores, axis=0)
        F1_scores_std = np.std(F1_scores, axis=0)
        Precisions_mean = np.mean(Precisions, axis=0)
        Precisions_std = np.std(Precisions, axis=0)
        cm = np.mean([r["Confusion matrix"] for r in results], axis=0)
        text += "Agregated results :\n"
    else:
        cm = results["Confusion matrix"]
        accuracy = results["Accuracy"]
        F1scores = results["F1 scores"]
        Precision = results['Precisions']
        AA = results['AA']
        kappa = results["Kappa"]

    vis.heatmap(cm, opts={'title': "Confusion matrix",
                          'marginbottom': 150,
                          'marginleft': 150,
                          'width': 500,
                          'height': 500,
                          'rownames': label_values, 'columnnames': label_values})
    text += "Confusion matrix :\n"
    text += str(cm)
    text += "---\n"

    class_labels = [f"C{i}" for i in range(1, len(label_values))]

    text += "F1 scores :\n"
    if agregated:
        for label, score, std in zip(class_labels, F1_scores_mean[1:],
                                     F1_scores_std[1:]):
            text += "\t{}: {:.04f} +- {:.04f}\n".format(label, score, std)
    else:
        for label, score in zip(class_labels, F1scores[1:]):
            text += "\t{}: {:.04f}\n".format(label, score)
    text += "---\n"

    text += "Precisions :\n"
    if agregated:
        for label, score, std in zip(class_labels, Precisions_mean[1:],
                                     Precisions_std[1:]):
            text += "\t{}: {:.04f} +- {:.04f}\n".format(label, score, std)
    else:
        for label, score in zip(class_labels, Precision[1:]):
            text += "\t{}: {:.04f}\n".format(label, score)
    text += "---\n"

    if agregated:
        text += ("OA: {:.04f} +- {:.04f}\n".format(np.mean(accuracies) / 100,
                                                   np.std(accuracies) / 100))
        text += ("AA: {:.04f} +- {:.04f}\n".format(np.mean(AAs),
                                                    np.std(AAs)))
        text += ("Kappa: {:.04f} +- {:.04f}\n".format(np.mean(kappas),
                                                       np.std(kappas)))
    else:
        text += "OA : {:.04f}\n".format(accuracy / 100)
        text += "AA : {:.04f}\n".format(AA)
        text += "Kappa: {:.04f}\n".format(kappa)

    vis.text(text.replace('\n', '<br/>'))
    print(text)

def sample_gt(gt, train_size, mode='random'):
    indices = np.nonzero(gt)
    X = list(zip(*indices))
    y = gt[indices].ravel()
    train_gt = np.zeros_like(gt)
    test_gt = np.zeros_like(gt)
    if train_size > 1:
       train_size = int(train_size)

    if mode == 'random':
       train_indices, test_indices = sklearn.model_selection.train_test_split(X, train_size=train_size, stratify=y)
       train_row_indices, train_col_indices = zip(*train_indices)
       test_row_indices, test_col_indices = zip(*test_indices)

       train_gt[train_row_indices, train_col_indices] = gt[train_row_indices, train_col_indices]
       test_gt[test_row_indices, test_col_indices] = gt[test_row_indices, test_col_indices]

    elif mode == 'fixed':
       print("Sampling {} with train size = {}".format(mode, train_size))
       train_indices, test_indices = [], []
       for c in np.unique(gt):
           if c == 0:
              continue
           indices = np.nonzero(gt == c)
           X = list(zip(*indices))

           train, test = sklearn.model_selection.train_test_split(X, train_size=train_size)
           train_indices += train
           test_indices += test
       train_indices = [list(t) for t in zip(*train_indices)]
       test_indices = [list(t) for t in zip(*test_indices)]
       train_gt[train_indices] = gt[train_indices]
       test_gt[test_indices] = gt[test_indices]

    elif mode == 'disjoint':
        train_gt = np.copy(gt)
        test_gt = np.copy(gt)
        for c in np.unique(gt):
            mask = gt == c
            for x in range(gt.shape[0]):
                first_half_count = np.count_nonzero(mask[:x, :])
                second_half_count = np.count_nonzero(mask[x:, :])
                try:
                    ratio = first_half_count / (first_half_count + second_half_count)
                    if ratio > 0.9 * train_size:
                        break
                except ZeroDivisionError:
                    continue
            mask[:x, :] = 0
            train_gt[mask] = 0

        test_gt[train_gt > 0] = 0
    else:
        raise ValueError("{} sampling is not implemented yet.".format(mode))
    return train_gt, test_gt


def compute_imf_weights(ground_truth, n_classes=None, ignored_classes=[]):
    n_classes = np.max(ground_truth) if n_classes is None else n_classes
    weights = np.zeros(n_classes)
    frequencies = np.zeros(n_classes)

    for c in range(0, n_classes):
        if c in ignored_classes:
            continue
        frequencies[c] = np.count_nonzero(ground_truth == c)

    frequencies /= np.sum(frequencies)
    idx = np.nonzero(frequencies)
    median = np.median(frequencies[idx])
    weights[idx] = median / frequencies[idx]
    weights[frequencies == 0] = 0.
    return weights

def camel_to_snake(name):
    s = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s).lower()

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



def random_seed_setting(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


