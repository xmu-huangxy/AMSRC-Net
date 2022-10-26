import argparse
import os
import torch
import cv2
import joblib
import pickle
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import yaml
from models.generator_op_gated import Net
from datasets.dataset import Chunked_sample_dataset
from utils.eval_utils import save_evaluation_curves
import gc


METADATA = {
    "ped2": {
        "testing_video_num": 12,
        "testing_frames_cnt": [180, 180, 150, 180, 150, 180, 180, 180, 120, 150,
                               180, 180]
    },
    "avenue": {
        "testing_video_num": 21,
        "testing_frames_cnt": [1439, 1211, 923, 947, 1007, 1283, 605, 36, 1175, 841,
                               472, 1271, 549, 507, 1001, 740, 426, 294, 248, 273,
                               76],
    },
    "shanghaitech": {
        "testing_video_num": 107,
        "testing_frames_cnt": [265, 433, 337, 601, 505, 409, 457, 313, 409, 337,
                               337, 457, 577, 313, 529, 193, 289, 289, 265, 241,
                               337, 289, 265, 217, 433, 409, 529, 313, 217, 241,
                               313, 193, 265, 317, 457, 337, 361, 529, 409, 313,
                               385, 457, 481, 457, 433, 385, 241, 553, 937, 865,
                               505, 313, 361, 361, 529, 337, 433, 481, 649, 649,
                               409, 337, 769, 433, 241, 217, 265, 265, 217, 265,
                               409, 385, 481, 457, 313, 601, 241, 481, 313, 337,
                               457, 217, 241, 289, 337, 313, 337, 265, 265, 337,
                               361, 433, 241, 433, 601, 505, 337, 601, 265, 313,
                               241, 289, 361, 385, 217, 337, 265]
    },

}


def cal_training_stats(config, ckpt_path, training_chunked_samples_dir, model=None):
    device = config["device"]
    # load weights
    if model:
        model = model.cuda()
        model.eval()
    else:
        model = Net().to(device).to(config["device"]).eval()
        model_weights = torch.load(ckpt_path)["model_state_dict"]
        model.load_state_dict(model_weights)
    # print("load pre-trained success!")

    score_func = nn.MSELoss(reduction="none")
    training_chunk_samples_files = sorted(os.listdir(training_chunked_samples_dir))

    frame_training_stats = []
    fea_training_stats = []

    print("=========Forward pass for training stats ==========")
    with torch.no_grad():
        for chunk_file_idx, chunk_file in enumerate(training_chunk_samples_files):
            dataset = Chunked_sample_dataset(os.path.join(training_chunked_samples_dir, chunk_file))
            dataloader = DataLoader(dataset=dataset, batch_size=128, num_workers=0, shuffle=False)

            for idx, data in tqdm(enumerate(dataloader),
                                  desc="Training stats calculating, Chunked File %02d" % chunk_file_idx,
                                  total=len(dataloader)):
                sample_frames, sample_ofs, _, _, _ = data
                sample_frames = sample_frames.to(device)
                sample_ofs = sample_ofs.to(device)
                out = model(sample_frames, sample_ofs)
                # frame_score
                loss_frame = score_func(out["frame_pred"], out["frame_target"]).cpu().data.numpy()
                frame_scores = np.sum(np.sum(np.sum(loss_frame, axis=3), axis=2), axis=1)
                # fea_score
                op_embedding, RGB_embedding = out["op_embedding"], out["RGB_embedding"]
                op_embedding = op_embedding.reshape(op_embedding.size(0), -1)
                RGB_embedding = RGB_embedding.reshape(RGB_embedding.size(0), -1)
                similaritys = torch.cosine_similarity(op_embedding, RGB_embedding, dim=1).cpu().numpy()
                fea_scores = (1 - similaritys)
                frame_training_stats.append(frame_scores)
                fea_training_stats.append(fea_scores)
            del dataset
            gc.collect()

    print("=========Forward pass for training stats done!==========")
    frame_training_stats = np.concatenate(frame_training_stats, axis=0)
    fea_training_stats = np.concatenate(fea_training_stats, axis=0)
    training_stats = dict(frame_training_stats=frame_training_stats, fea_training_stats=fea_training_stats)
    return training_stats


def evaluate(config, ckpt_path, testing_chunked_samples_file, training_chunked_samples_dir, suffix, model=None):
    dataset_name = config["dataset_name"]
    dataset_base_dir = config["dataset_base_dir"]
    device = config["device"]
    num_workers = config["num_workers"]

    testset_num_frames = np.sum(METADATA[dataset_name]["testing_frames_cnt"])

    eval_dir = os.path.join(config["eval_root"], config["exp_name"])
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    if model:
        model = model.cuda()
        model.eval()
    else:
        model = Net().to(device).eval()
        model_weights = torch.load(ckpt_path)["model_state_dict"]
        model.load_state_dict(model_weights)
    # print("load pre-trained success!")

    training_stats = cal_training_stats(config, ckpt_path=None, training_chunked_samples_dir=training_chunked_samples_dir,  model=model)

    if training_stats is not None:
        frame_mean, frame_std = np.mean(training_stats["frame_training_stats"]), \
                                np.std(training_stats["frame_training_stats"])
        fea_mean, fea_std = np.mean(training_stats["fea_training_stats"]), \
                                np.std(training_stats["fea_training_stats"])

    score_func = nn.MSELoss(reduction="none")
    dataset_test = Chunked_sample_dataset(testing_chunked_samples_file)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=128, num_workers=num_workers, shuffle=False)

    # bbox anomaly scores for each frame
    frame_bbox_scores = [{} for i in range(testset_num_frames.item())]
    for test_data in tqdm(dataloader_test, desc="Eval: ", total=len(dataloader_test)):

        sample_frames_test, sample_ofs_test, bbox_test, pred_frame_test, indices_test = test_data
        sample_frames_test = sample_frames_test.to(device)
        sample_ofs_test = sample_ofs_test.to(device)

        out_test = model(sample_frames_test, sample_ofs_test)

        loss_frame_test = score_func(out_test["frame_pred"], out_test["frame_target"]).cpu().data.numpy()
        frame_scores = np.sum(np.sum(np.sum(loss_frame_test, axis=3), axis=2), axis=1)
        op_embedding, RGB_embedding = out_test["op_embedding"], out_test["RGB_embedding"]
        op_embedding = op_embedding.reshape(op_embedding.size(0), -1)
        RGB_embedding = RGB_embedding.reshape(RGB_embedding.size(0), -1)
        similaritys = torch.cosine_similarity(op_embedding, RGB_embedding, dim=1).cpu().numpy()
        fea_scores = (1 - similaritys)

        if training_stats is not None:
            frame_scores = (frame_scores - frame_mean) / frame_std
            fea_scores = (fea_scores - fea_mean) / fea_std
            # print(frame_scores)

        scores = config["w_p"] * frame_scores + config["w_s"] * fea_scores

        for i in range(len(scores)):
            frame_bbox_scores[pred_frame_test[i][-1].item()][i] = scores[i]

    del dataset_test

    # frame-level anomaly score
    frame_scores = np.empty(len(frame_bbox_scores))
    for i in range(len(frame_scores)):
        if len(frame_bbox_scores[i].items()) == 0:
            frame_scores[i] = (0 - frame_mean) / frame_std
        else:
            frame_scores[i] = np.max(list(frame_bbox_scores[i].values()))

    joblib.dump(frame_scores,
                os.path.join(config["eval_root"], config["exp_name"], "frame_scores_%s.json" % suffix))

    # ================== Calculate AUC ==============================
    # load gt labels
    gt = pickle.load(
        open(os.path.join(config["dataset_base_dir"], "%s/ground_truth_demo/gt_label.json" % dataset_name), "rb"))
    gt_concat = np.concatenate(list(gt.values()), axis=0)

    new_gt = np.array([])
    new_frame_scores = np.array([])

    start_idx = 0
    for cur_video_id in range(METADATA[dataset_name]["testing_video_num"]):
        gt_each_video = gt_concat[start_idx:start_idx + METADATA[dataset_name]["testing_frames_cnt"][cur_video_id]][4:]
        scores_each_video = frame_scores[
                            start_idx:start_idx + METADATA[dataset_name]["testing_frames_cnt"][cur_video_id]][4:]

        start_idx += METADATA[dataset_name]["testing_frames_cnt"][cur_video_id]

        new_gt = np.concatenate((new_gt, gt_each_video), axis=0)
        new_frame_scores = np.concatenate((new_frame_scores, scores_each_video), axis=0)

    gt_concat = new_gt
    frame_scores = new_frame_scores

    curves_save_path = os.path.join(config["eval_root"], config["exp_name"], 'anomaly_curves_%s' % suffix)
    auc = save_evaluation_curves(frame_scores, gt_concat, curves_save_path,
                                 np.array(METADATA[dataset_name]["testing_frames_cnt"]) - 4)

    return auc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
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

    testing_chunked_samples_file = os.path.join("./data", config["dataset_name"],
                                                "testing/chunked_samples/chunked_samples_00.pkl")
    training_chunked_samples_dir = os.path.join("./data", config["dataset_name"],
                                                "training/chunked_samples")
    model_to_eval_path = config["pretrained_model"]

    with torch.no_grad():
        auc = evaluate(config, model_to_eval_path, testing_chunked_samples_file, training_chunked_samples_dir, suffix="best")
        print(auc)
