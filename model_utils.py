import torch.nn as nn
import torch
import torch.optim as optim
import os
import datetime
import numpy as np
import joblib
from tqdm import tqdm
from utils import grouper, sliding_window, count_sliding_window, camel_to_snake
from model.SDSM import SDSM

def get_model(name, **kwargs):
    device = kwargs.setdefault("device", torch.device("cpu"))
    n_classes = kwargs["n_classes"]
    (n_bands, n_bands2) = kwargs["n_bands"]
    weights = torch.ones(n_classes)
    weights[torch.LongTensor(kwargs["ignored_labels"])] = 0.0
    weights = weights.to(device)
    weights = kwargs.setdefault("weights", weights)

    if name == "SDSM":
        kwargs.setdefault("patch_sizes", [7])
        if isinstance(kwargs["patch_sizes"], int):
            kwargs["patch_sizes"] = [kwargs["patch_sizes"]]
        patch_size = kwargs["patch_sizes"][0]
        kwargs.setdefault("encoder_embed_dim", 64)
        kwargs.setdefault("en_depth", 4)
        kwargs.setdefault("en_heads", 4)
        kwargs.setdefault("dim_head", 16)
        kwargs.setdefault("emb_dropout", 0.1)
        num_patches = (patch_size ** 2)
        center_pixel = True

        model = SDSM(
            l1=n_bands,
            l2=n_bands2,
            patch_size=patch_size,
            num_patches=num_patches,
            num_classes=n_classes,
            encoder_embed_dim=kwargs["encoder_embed_dim"],
            en_depth=kwargs["en_depth"],
            en_heads=kwargs["en_heads"],
            mlp_dim=128,
            dim_head=kwargs["dim_head"],
            dropout=0.1,
            emb_dropout=kwargs["emb_dropout"]
        )

        lr = kwargs.setdefault("lr", 0.001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])

        kwargs.setdefault("epoch", 128)
        kwargs.setdefault("batch_size", 64)


    else:
        raise KeyError("{} model is unknown.".format(name))

    model = model.to(device)
    epoch = kwargs.setdefault("epoch", 100)
    kwargs.setdefault(
        "scheduler",

        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch),
    )
    kwargs.setdefault("batch_size", 64)
    kwargs.setdefault("supervision", "full")
    kwargs.setdefault("flip_augmentation", False)
    kwargs.setdefault("radiation_augmentation", False)
    kwargs.setdefault("mixture_augmentation", False)
    kwargs["center_pixel"] = center_pixel
    return model, optimizer, criterion, kwargs



def train(
    net,
    optimizer,
    criterion,
    data_loader,
    epoch,
    scheduler=None,
    display_iter=100,
    device=torch.device("cpu"),
    display=None,
    val_loader=None,
    supervision="full",
    test_interval=20,
    test_callback=None,
):
    if criterion is None:
        raise Exception("Missing criterion. You must specify a loss function.")

    net.to(device)

    save_epoch = epoch // 20 if epoch > 20 else 10

    losses = np.zeros(1000000)
    mean_losses = np.zeros(100000000)
    iter_ = 1
    loss_win, val_win = None, None
    val_accuracies = []

    for e in tqdm(range(1, epoch + 1), desc="Training the network"):
        net.train()
        avg_loss = 0.0

        for batch_idx, (data, data2, target) in tqdm(
            enumerate(data_loader), total=len(data_loader)
        ):
            data, data2, target = data.to(device), data2.to(device), target.to(device)

            optimizer.zero_grad()
            if supervision == "full":
                output = net(data, data2)
                loss = criterion(output, target)
            elif supervision == "semi":
                outs = net(data, data2)
                output, rec = outs
                loss = criterion[0](output, target) + net.aux_loss_weight * criterion[
                    1
                ](rec, data)
            else:
                raise ValueError(
                    'supervision mode "{}" is unknown.'.format(supervision)
                )
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()
            losses[iter_] = loss.item()
            mean_losses[iter_] = np.mean(losses[max(0, iter_ - 100) : iter_ + 1])

            if display_iter and iter_ % display_iter == 0:
                string = "Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}"
                string = string.format(
                    e,
                    epoch,
                    batch_idx * len(data),
                    len(data) * len(data_loader),
                    100.0 * batch_idx / len(data_loader),
                    mean_losses[iter_],
                )
                update = None if loss_win is None else "append"
                loss_win = display.line(
                    X=np.arange(iter_ - display_iter, iter_),
                    Y=mean_losses[iter_ - display_iter : iter_],
                    win=loss_win,
                    update=update,
                    opts={
                        "title": "Training loss",
                        "xlabel": "Iterations",
                        "ylabel": "Loss",
                    },
                )
                tqdm.write(string)

                if len(val_accuracies) > 0:
                    val_win = display.line(
                        Y=np.array(val_accuracies),
                        X=np.arange(len(val_accuracies)),
                        win=val_win,
                        opts={
                            "title": "Validation accuracy",
                            "xlabel": "Epochs",
                            "ylabel": "Accuracy",
                        },
                    )
            iter_ += 1
            del (data, target, loss, output)

        avg_loss /= len(data_loader)
        if val_loader is not None:
            val_acc = val(net, val_loader, device=device, supervision=supervision)
            val_accuracies.append(val_acc)
            metric = -val_acc
        else:
            metric = avg_loss

        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(metric)
        elif scheduler is not None:
            scheduler.step()

        if test_callback is not None and e % test_interval == 0:
            print(f"\n在第 {e} 轮进行测试...")
            net.eval()
            test_callback(net, e)
            net.train()
        if e % save_epoch == 0:
            save_model(
                net,
                camel_to_snake(str(net.__class__.__name__)),
                data_loader.dataset.name,
                epoch=e,
                metric=abs(metric),
            )


def save_model(model, model_name, dataset_name, **kwargs):
    model_dir = "./checkpoints/" + model_name + "/" + dataset_name + "/"
    time_str = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    if isinstance(model, torch.nn.Module):
        filename = time_str + "_epoch{epoch}_{metric:.2f}".format(
            **kwargs
        )
        tqdm.write("Saving neural network weights in {}".format(filename))
        torch.save(model.state_dict(), model_dir + filename + ".pth")
    else:
        filename = time_str
        tqdm.write("Saving model params in {}".format(filename))
        joblib.dump(model, model_dir + filename + ".pkl")


def test(net, img1, img2, hyperparams):
    net.eval()
    patch_size = hyperparams["patch_size"]
    center_pixel = hyperparams["center_pixel"]
    batch_size, device = hyperparams["batch_size"], hyperparams["device"]
    n_classes = hyperparams["n_classes"]

    kwargs = {
        "step": hyperparams["test_stride"],
        "window_size": (patch_size, patch_size),
    }
    probs = np.zeros(img1.shape[:2] + (n_classes,))

    iterations = count_sliding_window(img1, img2, **kwargs) // batch_size
    for batch in tqdm(
        grouper(batch_size, sliding_window(img1, img2, **kwargs)),
        total=(iterations),
        desc="Inference on the image",
    ):
        with torch.no_grad():
            if patch_size == 1:
                data = [b[0][0, 0] for b in batch]
                data = np.copy(data)
                data = torch.from_numpy(data)

                data2 = [b[1][0, 0] for b in batch]
                data2 = np.copy(data2)
                data2 = torch.from_numpy(data2)
            else:
                data = [b[0] for b in batch]
                data = np.copy(data)
                data = data.transpose(0, 3, 1, 2)
                data = torch.from_numpy(data)

                data2 = [b[1] for b in batch]
                data2 = np.copy(data2)
                data2 = data2.transpose(0, 3, 1, 2)
                data2 = torch.from_numpy(data2)

            indices = [b[2:] for b in batch]
            data = data.to(device)
            data2 = data2.to(device)
            output = net(data, data2)
            if isinstance(output, tuple):
                output = output[0]
            output = output.to("cpu")

            if patch_size == 1 or center_pixel:
                output = output.numpy()
            else:
                output = np.transpose(output.numpy(), (0, 2, 3, 1))
            for (x, y, w, h), out in zip(indices, output):
                if center_pixel:
                    probs[x + w // 2, y + h // 2] += out
                else:
                    probs[x : x + w, y : y + h] += out
    return probs



def val(net, data_loader, device="cpu", supervision="full"):
    # TODO : fix me using metrics()
    accuracy, total = 0.0, 0.0
    ignored_labels = data_loader.dataset.ignored_labels
    for batch_idx, (data, data2, target) in enumerate(data_loader):
        with torch.no_grad():
            data, data2, target = data.to(device), data2.to(device), target.to(device)
            if supervision == "full":
                output = net(data, data2)
            elif supervision == "semi":
                outs = net(data)
                output, rec = outs

            if isinstance(output, tuple):
                output = output[0]
            _, output = torch.max(output, dim=1)
            for out, pred in zip(output.view(-1), target.view(-1)):
                if out.item() in ignored_labels:
                    continue
                else:
                    accuracy += out.item() == pred.item()
                    total += 1
    return accuracy / total

