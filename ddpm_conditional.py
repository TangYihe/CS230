import argparse
from contextlib import nullcontext
import os
import copy
import numpy as np
import torch
import torch.nn as nn
from types import SimpleNamespace
from fastprogress import progress_bar, master_bar
from torch import optim
from utils import *
from modules import UNet_conditional, EMA, UNet_semEmb
import logging
import wandb
from IPython import embed # for sake of debugging
from embedding_utils import prepare_cnn, cnn_embed, prepare_vae, vae_embed
from torchvision import transforms

config = SimpleNamespace(    
    run_name = "DDPM_conditional",
    epochs = 100,
    noise_steps=1000,
    seed = 42,
    batch_size = 10,
    img_size = 48,
    num_classes = 7,
    # dataset_path = get_cifar(img_size=64),
    dataset_path = None,
    train_folder = "train",
    val_folder = "test",
    device = "cuda",
    slice_size = 1,
    use_wandb = True,
    do_validation = True,
    fp16 = True,
    log_every_epoch = 10,
    num_workers=10,
    lr = 3e-4)


logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, num_classes=10, c_in=3, c_out=3, use_sem=None, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size

        self.use_sem = use_sem
        if self.use_sem == 'cnn':
            sem_dim = 256
        elif self.use_sem == 'vae':
            sem_dim = 128

        if self.use_sem is not None:
            self.model = UNet_semEmb(c_in, c_out, num_classes=num_classes, sem_dim=sem_dim).to(device)
        else:
            self.model = UNet_conditional(c_in, c_out, num_classes=num_classes).to(device)
        self.ema_model = copy.deepcopy(self.model).eval().requires_grad_(False)
        self.device = device
        self.c_in = c_in
        self.num_classes = num_classes

        # prepare semantic embedding models if applicable
        self.emb_model = None
        if use_sem == 'cnn':
            self.emb_model = prepare_cnn()
        elif use_sem == 'vae':
            self.emb_model = prepare_vae()

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
    
    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def noise_images(self, x, t):
        "Add noise to images at instant t"
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ
    
    @torch.inference_mode()
    def sample(self, use_ema, labels, cfg_scale=3, seed=None, init_noise=None, ref_images=None):
        model = self.ema_model if use_ema else self.model
        n = len(labels)
        if ref_images is not None:
            ref_images = ref_images[:n]
            print(ref_images.size())
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.inference_mode():
            # include manual_seed to increase reproducibility 
            if seed:
                torch.manual_seed(seed)
                
            if init_noise is None:
                x = torch.randn((n, self.c_in, self.img_size, self.img_size)).to(self.device)
            else: # support selected noise
                x = init_noise.to(self.device)

            # check if has passed in reference images
            sem_encoding = None
            if ref_images is not None:
                if self.use_sem == 'cnn':
                    sem_encoding = torch.tensor(cnn_embed(self.emb_model, ref_images.expand(-1,3,-1,-1))).to(self.device)
                elif self.use_sem == 'vae':
                    sem_encoding = torch.tensor(vae_embed(self.emb_model, ref_images)).to(self.device)

            for i in progress_bar(reversed(range(1, self.noise_steps)), total=self.noise_steps-1, leave=False):
                t = (torch.ones(n) * i).long().to(self.device)
                if self.use_sem is not None:
                    predicted_noise = model(x, t, labels, sem_encoding=sem_encoding)
                else:
                    predicted_noise = model(x, t, labels)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                    # predicted_noise = uncond_predicted_noise
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise

        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x

    def train_step(self, loss):
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.ema.step_ema(self.ema_model, self.model)
        self.scheduler.step()

    def one_epoch(self, train=True, use_wandb=False):
        avg_loss = 0.
        if train: self.model.train()
        else: self.model.eval()
        pbar = progress_bar(self.train_dataloader, leave=False)
        
        first_batch = None
        for i, (images, labels) in enumerate(pbar):
            with torch.autocast("cuda") and (torch.inference_mode() if not train else torch.enable_grad()):
                images = images.to(self.device)
                if first_batch is None:
                    first_batch = images

                sem_encoding = None
                if self.use_sem == 'cnn':
                    sem_encoding = torch.tensor(cnn_embed(self.emb_model, images.expand(-1,3,-1,-1))).to(self.device)
                elif self.use_sem == 'vae':
                    sem_encoding = torch.tensor(vae_embed(self.emb_model, images)).to(self.device)
                labels = labels.to(self.device)
                t = self.sample_timesteps(images.shape[0]).to(self.device)
                x_t, noise = self.noise_images(images, t)
                if np.random.random() < 0.1:
                    labels = None
                predicted_noise = self.model(x_t, t, labels, sem_encoding=sem_encoding)
                loss = self.mse(noise, predicted_noise)
                avg_loss += loss
            if train:
                self.train_step(loss)
                if use_wandb: 
                    wandb.log({"train_mse": loss.item(),
                                "learning_rate": self.scheduler.get_last_lr()[0]})
            pbar.comment = f"MSE={loss.item():2.3f}"  

        # returned the last batch of images for progress tracking      
        return (avg_loss.mean().item(), first_batch)

    def log_images(self, use_wandb=False, ref_images=None):
        "Log images to wandb and save them to disk"
        labels = torch.arange(self.num_classes).long().to(self.device)
        sampled_images = self.sample(use_ema=False, labels=labels, ref_images=ref_images)
        ema_sampled_images = self.sample(use_ema=True, labels=labels, ref_images=ref_images)
        # plot_images(sampled_images)  #to display on jupyter if available
        if use_wandb:
            wandb.log({"sampled_images":     [wandb.Image(img.permute(1,2,0).squeeze().cpu().numpy()) for img in sampled_images]})
            wandb.log({"ema_sampled_images": [wandb.Image(img.permute(1,2,0).squeeze().cpu().numpy()) for img in ema_sampled_images]})

    def load(self, model_cpkt_path, model_ckpt="ckpt.pt", ema_model_ckpt="ema_ckpt.pt"):
        self.model.load_state_dict(torch.load(os.path.join(model_cpkt_path, model_ckpt)))
        self.ema_model.load_state_dict(torch.load(os.path.join(model_cpkt_path, ema_model_ckpt)))

    def save_model(self, run_name, use_wandb=False, epoch=-1):
        if not os.path.exists(os.path.join("models", run_name)):
            os.makedirs(os.path.join("models", run_name))
        "Save model locally and on wandb"
        torch.save(self.model.state_dict(), os.path.join("models", run_name, f"ckpt.pt"))
        torch.save(self.ema_model.state_dict(), os.path.join("models", run_name, f"ema_ckpt.pt"))
        torch.save(self.optimizer.state_dict(), os.path.join("models", run_name, f"optim.pt"))
        if use_wandb:
            at = wandb.Artifact("model", type="model", description="Model weights for DDPM conditional", metadata={"epoch": epoch})
            at.add_dir(os.path.join("models", run_name))
            wandb.log_artifact(at)

    def prepare(self, args):
        # mk_folders(args.run_name)
        if args:
            device = args.device
        else:
            device = "cuda"
        self.train_dataloader, self.val_dataloader= get_fer_data(args)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=0.001)
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=args.lr, 
                                                 steps_per_epoch=len(self.train_dataloader), epochs=args.epochs)
        self.mse = nn.MSELoss()
        self.ema = EMA(0.995)
        self.scaler = torch.cuda.amp.GradScaler()

    def fit(self, args):
        for epoch in progress_bar(range(args.epochs), total=args.epochs, leave=True):
            logging.info(f"Starting epoch {epoch}:")
            _, train_images  = self.one_epoch(train=True, use_wandb=args.use_wandb)
            
            ## validation
            if args.do_validation:
                avg_loss, val_images = self.one_epoch(train=False, use_wandb=args.use_wandb)
                if args.use_wandb:
                    wandb.log({"val_mse": avg_loss})
            
            # log predicitons
            if epoch % args.log_every_epoch == 0:
                self.log_images(use_wandb=args.use_wandb, ref_images=train_images)
                self.save_model(run_name=args.run_name, use_wandb=args.use_wandb, epoch=epoch)

        # save model
        self.save_model(run_name=args.run_name, use_wandb=args.use_wandb, epoch=epoch)




def parse_args(config):
    parser = argparse.ArgumentParser(description='Process hyper-parameters')
    parser.add_argument('--run_name', type=str, default=config.run_name, help='name of the run')
    parser.add_argument('--epochs', type=int, default=config.epochs, help='number of epochs')
    parser.add_argument('--seed', type=int, default=config.seed, help='random seed')
    parser.add_argument('--batch_size', type=int, default=config.batch_size, help='batch size')
    parser.add_argument('--img_size', type=int, default=config.img_size, help='image size')
    parser.add_argument('--num_classes', type=int, default=config.num_classes, help='number of classes')
    parser.add_argument('--dataset_path', type=str, default=config.dataset_path, help='path to dataset')
    parser.add_argument('--device', type=str, default=config.device, help='device')
    parser.add_argument('--use_wandb', type=bool, default=config.use_wandb, help='use wandb')
    parser.add_argument('--lr', type=float, default=config.lr, help='learning rate')
    parser.add_argument('--slice_size', type=int, default=config.slice_size, help='slice size')
    parser.add_argument('--noise_steps', type=int, default=config.noise_steps, help='noise steps')
    parser.add_argument('--use_sem', type=str, default='vae', help='type of semantic encoding to use')
    args = vars(parser.parse_args())
    
    # update config with parsed args
    for k, v in args.items():
        setattr(config, k, v)


if __name__ == '__main__':
    parse_args(config)

    ## seed everything
    set_seed(config.seed)

    
    diffuser = Diffusion(config.noise_steps, img_size=config.img_size, num_classes=config.num_classes, c_in=1, c_out=1, use_sem=config.use_sem)
    print('after initialize')

    with wandb.init(project="train_sd", group="train", config=config) if config.use_wandb else nullcontext():
        diffuser.prepare(config)
        print('after prepare data')
        diffuser.fit(config)
