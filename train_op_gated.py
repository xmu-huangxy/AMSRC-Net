import os
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torch.nn as nn
import numpy as np
import yaml
import shutil
from eval import evaluate
from tqdm import tqdm
from losses.loss import *
from datasets.dataset import Chunked_sample_dataset, img_batch_tensor2numpy
from models.pix2pix_networks import PixelDiscriminator
from models.generator_op_gated import Net
from utils.initialization_utils import weights_init_kaiming
from utils.vis_utils import visualize_sequences
from utils.model_utils import loader, saver, only_model_saver
import random
import argparse


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train(config, training_chunked_samples_dir, testing_chunked_samples_file):
    paths = dict(log_dir="%s/%s" % (config["log_root"], config["exp_name"]),
                 ckpt_dir="%s/%s" % (config["ckpt_root"], config["exp_name"]))

    os.makedirs(paths["ckpt_dir"], exist_ok=True)

    batch_size = config["batchsize"]
    epochs = config["num_epochs"]
    num_workers = config["num_workers"]
    device = config["device"]
    lr = config["lr"]
    training_chunk_samples_files = sorted(os.listdir(training_chunked_samples_dir))

    # loss functions
    gradient_loss = Gradient_Loss(3).cuda()
    intensity_loss = Intensity_Loss().cuda()
    two_streams_similar_loss = Two_streams_similarity_Loss().cuda()

    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8, last_epoch=-1)

    step = 0
    epoch_last = 0
    if not config["pretrained"]:
        model.apply(weights_init_kaiming)
    else:
        assert (config["pretrained"] is not None)
        model.load_state_dict(torch.load(config["pretrained_model"])['net'])
        optimizer.load_state_dict(torch.load(config["pretrained_model"])['optimizer'])
        print(f'Pre-trained model have been loaded.\n')

    writer = SummaryWriter(paths["log_dir"])
    shutil.copyfile("./cfgs/cfg.yaml", os.path.join(config["log_root"], config["exp_name"], "cfg.yaml"))

    best_auc = -1
    for epoch in range(epoch_last, epochs + epoch_last):
        for chunk_file_idx, chunk_file in enumerate(training_chunk_samples_files):
            dataset = Chunked_sample_dataset(os.path.join(training_chunked_samples_dir, chunk_file))
            dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
            for idx, train_data in tqdm(enumerate(dataloader),
                                        desc="Training Epoch %d, Chunked File %d" % (epoch + 1, chunk_file_idx),
                                        total=len(dataloader)):
                model = model.train()

                sample_frames, sample_ofs, _, _, _ = train_data
                sample_ofs = sample_ofs.to(device)  # [batch, (t_length-1)*2, patch_H, patch_W]
                sample_frames = sample_frames.to(device)  # [batch, t_length*3, patch_H, patch_W]

                out = model(sample_frames, sample_ofs)

                inte_l = intensity_loss(out["frame_pred"], out["frame_target"])
                grad_l = gradient_loss(out["frame_pred"], out["frame_target"])
                two_l = two_streams_similar_loss(out["op_embedding"], out["RGB_embedding"])
                l2_l = l2_regularization(model, l2_alpha=0.001)

                loss_t = config['lam_frame'] * inte_l + config['lam_grad'] * grad_l + config['lam_similar'] * two_l + config['lam_l2'] * l2_l

                optimizer.zero_grad()
                loss_t.backward()
                optimizer.step()

                if step % config["logevery"] == config["logevery"] - 1:
                    print("[Step: {}/ Epoch: {}]: Loss: {:.4f} ".format(step + 1, epoch + 1, loss_t))

                    writer.add_scalar('loss_total/train', loss_t, global_step=step + 1)

                    num_vis = 6
                    writer.add_figure("img/train_sample_frames",
                                      visualize_sequences(img_batch_tensor2numpy(
                                          sample_frames.cpu()[:num_vis, :, :, :]),
                                          seq_len=sample_frames.size(1) // 3,
                                          return_fig=True),
                                      global_step=step + 1)
                    writer.add_figure("img/train_frame_recon",
                                      visualize_sequences(img_batch_tensor2numpy(
                                          out["frame_pred"].detach().cpu()[:num_vis, :, :, :]),
                                          seq_len=config["model_paras"]["clip_pred"],
                                          return_fig=True),
                                      global_step=step + 1)
                    writer.add_figure("img/train_of_target",
                                      visualize_sequences(img_batch_tensor2numpy(
                                          sample_ofs.cpu()[:num_vis, :, :, :]),
                                          seq_len=sample_ofs.size(1) // 2,
                                          return_fig=True),
                                      global_step=step + 1)

                step += 1

            del dataset
        scheduler.step()
        '''
        if epoch % config["saveevery"] == config["saveevery"] - 1:
            model_save_path = os.path.join(paths["ckpt_dir"], config["model_savename"])
            saver(model.state_dict(), optimizer.state_dict(), model_save_path, epoch + 1, step, max_to_save=5)

            # computer training stats
            stats_save_path = os.path.join(paths["ckpt_dir"], "training_stats.npy-%d" % (epoch + 1))
            cal_training_stats(config, model_save_path + "-%d" % (epoch + 1), training_chunked_samples_dir,
                               stats_save_path)

            with torch.no_grad():
                auc = evaluate(config, model_save_path + "-%d" % (epoch + 1),
                               testing_chunked_samples_file,
                               stats_save_path,
                               suffix=str(epoch + 1))
                if auc > best_auc:
                    best_auc = auc
                    only_model_saver(model.state_dict(), os.path.join(paths["ckpt_dir"], "best.pth"))

                writer.add_scalar("auc", auc, global_step=epoch + 1)
        '''
        with torch.no_grad():
            auc = evaluate(config, ckpt_path=None, testing_chunked_samples_file=testing_chunked_samples_file, training_chunked_samples_dir=training_chunked_samples_dir, suffix=str(epoch + 1), model=model)
            print(auc)
            if auc > best_auc:
                best_auc = auc
                only_model_saver(model.state_dict(), os.path.join(paths["ckpt_dir"], "best.pth"))

            writer.add_scalar("auc", auc, global_step=epoch + 1)

    print("================ Best AUC %.4f ================" % best_auc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='two_streams for Anomaly Prediction')
    parser.add_argument('--dataset', default='ped2', type=str, help='The name of the dataset to train.')
    args = parser.parse_args()
    if args.dataset == "ped2":
        config = yaml.safe_load(open("./cfgs/ped2_cfg.yaml"))
    elif args.dataset == "avenue":
        config = yaml.safe_load(open("./cfgs/avenue_cfg.yaml"))
    elif args.dataset == "shanghaitech":
        config = yaml.safe_load(open("./cfgs/shanghaitech_cfg.yaml"))
    else:
        raise NotImplementedError
        print("No other datasets can be trained!")
    dataset_name = config["dataset_name"]
    dataset_base_dir = config["dataset_base_dir"]
    training_chunked_samples_dir = os.path.join(dataset_base_dir, dataset_name, "training/chunked_samples")
    testing_chunked_samples_file = os.path.join(dataset_base_dir, dataset_name,
                                                "testing/chunked_samples/chunked_samples_00.pkl")
    setup_seed(2021)
    train(config, training_chunked_samples_dir, testing_chunked_samples_file)
