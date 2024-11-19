import os
import argparse
from tqdm import tqdm
import torch.nn as nn
import tensorflow as tf
import torch.optim as optim

from models.TMC import ETMC, ce_loss
import torchvision.transforms as transforms
from data.dfdt_dataset import FakeAVCelebDatasetTrain, FakeAVCelebDatasetVal


from utils.utils import *
from utils.logger import create_logger
from sklearn.metrics import accuracy_score
from torch.utils.tensorboard import SummaryWriter

# Define the audio_args dictionary
audio_args = {
    'nb_samp': 64600,
    'first_conv': 1024,
    'in_channels': 1,
    'filts': [20, [20, 20], [20, 128], [128, 128]],
    'blocks': [2, 4],
    'nb_fc_node': 1024,
    'gru_node': 1024,
    'nb_gru_layer': 3,
}


def get_args(parser):
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--data_dir", type=str, default="datasets/train/fakeavceleb*")
    parser.add_argument("--LOAD_SIZE", type=int, default=256)
    parser.add_argument("--FINE_SIZE", type=int, default=224)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--hidden", nargs="*", type=int, default=[])
    parser.add_argument("--hidden_sz", type=int, default=768)
    parser.add_argument("--img_embed_pool_type", type=str, default="avg", choices=["max", "avg"])
    parser.add_argument("--img_hidden_sz", type=int, default=1024)
    parser.add_argument("--include_bn", type=int, default=True)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_factor", type=float, default=0.3)
    parser.add_argument("--lr_patience", type=int, default=10)
    parser.add_argument("--max_epochs", type=int, default=500)
    parser.add_argument("--n_workers", type=int, default=12)
    parser.add_argument("--name", type=str, default="MMDF")
    parser.add_argument("--num_image_embeds", type=int, default=1)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--savedir", type=str, default="./savepath/")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n_classes", type=int, default=2)
    parser.add_argument("--annealing_epoch", type=int, default=10)
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--pretrained_image_encoder", type=bool, default = False)
    parser.add_argument("--freeze_image_encoder", type=bool, default = True)
    parser.add_argument("--pretrained_audio_encoder", type = bool, default=False)
    parser.add_argument("--freeze_audio_encoder", type = bool, default = True)
    parser.add_argument("--augment_dataset", type = bool, default = True)

    for key, value in audio_args.items():
        parser.add_argument(f"--{key}", type=type(value), default=value)

def get_optimizer(model, args):
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    return optimizer


def get_scheduler(optimizer, args):
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "max", patience=args.lr_patience, verbose=True, factor=args.lr_factor
    )

def model_forward(i_epoch, model, args, ce_loss, batch):
    rgb, spec, tgt = batch['video_reshaped'], batch['spectrogram'], batch['label_map']
    rgb_pt = torch.Tensor(rgb.numpy())
    spec = spec.numpy()
    spec_pt = torch.Tensor(spec)
    tgt_pt = torch.Tensor(tgt.numpy())

    if torch.cuda.is_available():
        rgb_pt, spec_pt, tgt_pt = rgb_pt.cuda(), spec_pt.cuda(), tgt_pt.cuda()
        
    # depth_alpha, rgb_alpha, depth_rgb_alpha = model(rgb_pt, spec_pt)

    # loss = ce_loss(tgt_pt, depth_alpha, args.n_classes, i_epoch, args.annealing_epoch) + \
    #        ce_loss(tgt_pt, rgb_alpha, args.n_classes, i_epoch, args.annealing_epoch) + \
    #        ce_loss(tgt_pt, depth_rgb_alpha, args.n_classes, i_epoch, args.annealing_epoch)
    # return loss, depth_alpha, rgb_alpha, depth_rgb_alpha, tgt_pt

    depth_alpha, rgb_alpha, pseudo_alpha, depth_rgb_alpha = model(rgb_pt, spec_pt)

    loss = ce_loss(tgt_pt, depth_alpha, args.n_classes, i_epoch, args.annealing_epoch) + \
           ce_loss(tgt_pt, rgb_alpha, args.n_classes, i_epoch, args.annealing_epoch) + \
           ce_loss(tgt_pt, pseudo_alpha, args.n_classes, i_epoch, args.annealing_epoch) + \
           ce_loss(tgt_pt, depth_rgb_alpha, args.n_classes, i_epoch, args.annealing_epoch)
    return loss, depth_alpha, rgb_alpha, depth_rgb_alpha, tgt_pt



def model_eval(i_epoch, data, model, args, criterion):
    model.eval()
    with torch.no_grad():
        losses, depth_preds, rgb_preds, depthrgb_preds, tgts = [], [], [], [], []
        for batch in tqdm(data):
            loss, depth_alpha, rgb_alpha, depth_rgb_alpha, tgt = model_forward(i_epoch, model, args, criterion, batch)
            losses.append(loss.item())

            depth_pred = depth_alpha.argmax(dim=1).cpu().detach().numpy()
            rgb_pred = rgb_alpha.argmax(dim=1).cpu().detach().numpy()
            depth_rgb_pred = depth_rgb_alpha.argmax(dim=1).cpu().detach().numpy()

            depth_preds.append(depth_pred)
            rgb_preds.append(rgb_pred)
            depthrgb_preds.append(depth_rgb_pred)
            tgt = tgt.cpu().detach().numpy()
            tgts.append(tgt)

    metrics = {"loss": np.mean(losses)}
    print(f"Mean loss is: {metrics['loss']}")

    tgts = [l for sl in tgts for l in sl]
    depth_preds = [l for sl in depth_preds for l in sl]
    rgb_preds = [l for sl in rgb_preds for l in sl]
    depthrgb_preds = [l for sl in depthrgb_preds for l in sl]
    metrics["spec_acc"] = accuracy_score(tgts, depth_preds)
    metrics["rgb_acc"] = accuracy_score(tgts, rgb_preds)
    metrics["specrgb_acc"] = accuracy_score(tgts, depthrgb_preds)
    return metrics

def write_weight_histograms(writer, step, model):
    for idx, item in enumerate(model.named_parameters()):
        name = item[0]
        weights = item[1].data
        if weights.size(dim = 0) > 2:
            try:
                writer.add_histogram(name, weights, idx)
            except ValueError as e:
                continue

writer = SummaryWriter()

def train(args):
    set_seed(args.seed)
    args.savedir = os.path.join(args.savedir, args.name)
    os.makedirs(args.savedir, exist_ok=True)

    train_ds = FakeAVCelebDatasetTrain(args)
    train_ds = train_ds.load_features_from_tfrec()

    val_ds = FakeAVCelebDatasetVal(args)
    val_ds = val_ds.load_features_from_tfrec()
    
    model = ETMC(args)
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)
    logger = create_logger("%s/logfile.log" % args.savedir, args)
    if torch.cuda.is_available():
        model.cuda()

    torch.save(args, os.path.join(args.savedir, "checkpoint.pt"))
    start_epoch, global_step, n_no_improve, best_metric = 0, 0, 0, -np.inf

    for i_epoch in range(start_epoch, args.max_epochs):
        train_losses = []
        model.train()
        optimizer.zero_grad()

        for index, batch in tqdm(enumerate(train_ds)):
            loss, depth_out, rgb_out, depthrgb, tgt = model_forward(i_epoch, model, args, ce_loss, batch)
            if args.gradient_accumulation_steps > 1:
                 loss = loss / args.gradient_accumulation_steps

            train_losses.append(loss.item())
            loss.backward()
            global_step += 1
            if global_step % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        #Write weight histograms to Tensorboard.
        write_weight_histograms(writer, i_epoch, model)

        model.eval()
        metrics = model_eval(
            np.inf, val_ds, model, args, ce_loss
        )
        logger.info("Train Loss: {:.4f}".format(np.mean(train_losses)))
        log_metrics("val", metrics, logger)
        logger.info(
            "{}: Loss: {:.5f} | spec_acc: {:.5f}, rgb_acc: {:.5f}, depth rgb acc: {:.5f}".format(
                "val", metrics["loss"], metrics["spec_acc"], metrics["rgb_acc"], metrics["specrgb_acc"]
            )
        )
        tuning_metric = metrics["specrgb_acc"]

        scheduler.step(tuning_metric)
        is_improvement = tuning_metric > best_metric
        if is_improvement:
            best_metric = tuning_metric
            n_no_improve = 0
        else:
            n_no_improve += 1

        save_checkpoint(
            {
                "epoch": i_epoch + 1,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "n_no_improve": n_no_improve,
                "best_metric": best_metric,
            },
            is_improvement,
            args.savedir,
        )

        if n_no_improve >= args.patience:
            logger.info("No improvement. Breaking out of loop.")
            break
    writer.close()
    # load_checkpoint(model, os.path.join(args.savedir, "model_best.pt"))
    model.eval()
    test_metrics = model_eval(
        np.inf, val_ds, model, args, ce_loss
    )
    logger.info(
        "{}: Loss: {:.5f} | spec_acc: {:.5f}, rgb_acc: {:.5f}, depth rgb acc: {:.5f}".format(
            "Test", test_metrics["loss"], test_metrics["spec_acc"], test_metrics["rgb_acc"],
            test_metrics["depthrgb_acc"]
        )
    )
    log_metrics(f"Test", test_metrics, logger)


def cli_main():
    parser = argparse.ArgumentParser(description="Train Models")
    get_args(parser)
    args, remaining_args = parser.parse_known_args()
    assert remaining_args == [], remaining_args
    train(args)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    cli_main()
