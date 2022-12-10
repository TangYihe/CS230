# Evaluation on output images from conditional diffusion model
# Using DeepFace emotion prediction from https://github.com/serengil/deepface

import os
import wandb, torch
from ddpm_conditional import *
from fastcore.all import *
from modules import *
from fer_data import fer_dataset
from embedding_utils import prepare_cnn, cnn_embed, prepare_vae, vae_embed

from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt
import json
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, precision_recall_fscore_support

## Classwise evaluation results
def calc_on_label(y_true, y_pred, model_name):
    print('-------------'+model_name+'------------------')
    precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred, labels = emotion_label)

    print(emotion_label)
    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('fscore: {}'.format(fscore))
    print('support: {}'.format(support))
    print()

##  Score Calculation
def calc_score(y_true, y_pred):
    print("fscore:", f1_score(y_true, y_pred, average="macro"))
    print("precision:", precision_score(y_true, y_pred, average="macro"))
    print("recall:", recall_score(y_true, y_pred, average="macro"))  
    
## Input file from the directory for complete evaluation pipeline
def total_eval(directory):
    ## Correspondance from label to emotion
    emotion_label = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
    y_true_total = []
    y_pred_total = []

    y_true_label = []
    y_pred_label = []

    y_true_cnn = []
    y_pred_cnn = []

    y_true_vae = []
    y_pred_vae = []

    ## Without Face detection -> take the image as a whole for emotion detection
    for orig_label in os.listdir(directory):
        f = os.path.join(directory, orig_label)
        if os.path.isdir(f):
            print(f)
            for img_path in os.listdir(f):
                if img_path[-5:] != '.jpeg':
                    continue
                desired_label = img_path[:-5].split('_')[-1]
                model = img_path[:-5].split('_')[0]
                if (model == 'ema') or (model == 'non'):
                    continue
                if int(img_path[:-5].split('_')[1]) > 5:
                    continue
                full_img_path = os.path.join(f, img_path)
                img = cv2.imread(full_img_path)
                ## Analyze emotion
                demoraphy = DeepFace.analyze(img_path=full_img_path, 
                actions = ['emotion'], enforce_detection = False)
                demo_json = json.loads(json.dumps(demoraphy))
                dominate_emotion = demo_json['dominant_emotion']

                if model == 'label':
                    y_true_label.append(emotion_label[int(desired_label)])
                    y_pred_label.append(dominate_emotion)
                elif model == 'cnn':
                    y_true_cnn.append(emotion_label[int(desired_label)])
                    y_pred_cnn.append(dominate_emotion)
                elif model == 'vae':
                    y_true_vae.append(emotion_label[int(desired_label)])
                    y_pred_vae.append(dominate_emotion)

    # Calculate score for label embedding
    calc_on_label(y_true_label, y_pred_label, 'LABEL')
    calc_score(y_true_label, y_pred_label)
    # Calculate score for CNN embedding
    calc_on_label(y_true_cnn, y_pred_cnn, 'CNN')
    calc_score(y_true_cnn, y_pred_cnn)
    # Calculate score for VAE embedding
    calc_on_label(y_true_vae, y_pred_vae, 'VAE')
    calc_score(y_true_vae, y_pred_vae)


def main():  
    torch.cuda.empty_cache()
    # Number of labels
    n = 7
    device = "cuda"
    directory = 'output_img'
    
    # Load prevoius checkpoints
    diffuser = Diffusion(noise_steps=1000, img_size=64, num_classes=7, c_in=1, c_out=1, use_sem=None)
    ckpt = torch.load("/home/ubuntu/CS230/model_ckpt/DDPM_conditional_aug/ckpt.pt")
    ema_ckpt = torch.load("/home/ubuntu/CS230/model_ckpt/DDPM_conditional_aug/ema_ckpt.pt")
    diffuser.model.load_state_dict(ckpt)
    diffuser.ema_model.load_state_dict(ema_ckpt)

    vae_diffuser = Diffusion(noise_steps=1000, img_size=64, num_classes=10, c_in=1, c_out=1, use_sem='vae')
    ckpt = torch.load("/home/ubuntu/CS230/models/vae_resume/ckpt.pt")
    ema_ckpt = torch.load("/home/ubuntu/CS230/models/vae_resume/ema_ckpt.pt")
    vae_diffuser.model.load_state_dict(ckpt)
    vae_diffuser.ema_model.load_state_dict(ema_ckpt)

    cnn_diffuser = Diffusion(noise_steps=1000, img_size=64, num_classes=10, c_in=1, c_out=1, use_sem='cnn')
    ckpt = torch.load("/home/ubuntu/CS230/models/cnn_encode/ckpt.pt")
    ema_ckpt = torch.load("/home/ubuntu/CS230/models/cnn_encode/ema_ckpt.pt")
    cnn_diffuser.model.load_state_dict(ckpt)
    cnn_diffuser.ema_model.load_state_dict(ema_ckpt)

    # load images
    
    train_dataloader = torch.load('/home/ubuntu/CS230/dataset/fer_train_32.pt')
    val_dataloader = torch.load('/home/ubuntu/CS230/dataset/fer_val_32.pt')

    # Generate 5 samples for each label
    orig_sample_count = [0]*7
    device = "cuda"
    for idx, (img, label) in enumerate(val_dataloader):
        if orig_sample_count == [5]*7:
            break

        # process input
        t = torch.randint(low=999, high=1000, size=(img.shape[0],))
        t = t.to(device)
        img = img.to(device).float()
        label = label.to(device)

        # get noised imgs
        x_t, noise = vae_diffuser.noise_images(img, t)

        print(img.shape, x_t.shape)

        # denoise for all labels
        for i in range(2, x_t.shape[0]):
            if orig_sample_count == [5]*7:
                break
            orig_sample_count[label[i]] += 1
            print(orig_sample_count)

            z = x_t[i].unsqueeze(axis=0).expand(7, -1, -1, -1)
            x = img[i].unsqueeze(axis=0).expand(7, -1, -1, -1) # [7, 1, 48, 48]

            labels = torch.arange(7).long().to(device)
            sampled_images = diffuser.sample(use_ema=True, labels=labels, seed=0, init_noise=z)
            vae_sampled_images = vae_diffuser.sample(use_ema=True, labels=labels, seed=0, init_noise=z, ref_images=x)
            cnn_sampled_images = cnn_diffuser.sample(use_ema=True, labels=labels, seed=0, init_noise=z, ref_images=x)

            # Save images in orig_label folder, from current label count to desired label
            for j in range(n):
                im = Image.fromarray(sampled_images[j].squeeze().cpu().numpy())
                im.save(f"./{directory}/{label[i]}/label_{orig_sample_count[label[i]]}_{j}.jpeg")
                im = Image.fromarray(vae_sampled_images[j].squeeze().cpu().numpy())
                im.save(f"./{directory}/{label[i]}/vae_{orig_sample_count[label[i]]}_{j}.jpeg")
                im = Image.fromarray(cnn_sampled_images[j].squeeze().cpu().numpy())
                im.save(f"./{directory}/{label[i]}/cnn_{orig_sample_count[label[i]]}_{j}.jpeg")
                
    # Go through evaluation pipeline
    total_eval(directory)


if __name__ == '__main__':
    main()