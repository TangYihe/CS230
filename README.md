# Conditional Diffusion Models on Emotion Generation
This is an easy-to-understand implementation of diffusion models within 100 lines of code adapted from https://github.com/tcapelle/Diffusion-Models-pytorch/. Different from other implementations, this code doesn't use the lower-bound formulation for sampling and strictly follows Algorithm 1 from the [DDPM](https://arxiv.org/pdf/2006.11239.pdf) paper, which makes it extremely short and easy to follow. There are two implementations: `conditional` and `unconditional`. We use the `conditional` implementation to include the labels for different emotions. The conditional code also implements Classifier-Free-Guidance (CFG) and Exponential-Moving-Average (EMA). Below you can find two explanation videos for the theory behind diffusion models and the implementation.

<a href="https://www.youtube.com/watch?v=HoKDTa5jHvg">
   <img alt="Qries" src="https://user-images.githubusercontent.com/61938694/191407922-f613759e-4bea-4ac9-9135-d053a6312421.jpg"
   width="300">
</a>

<a href="https://www.youtube.com/watch?v=TBCRlnwJtZU">
   <img alt="Qries" src="https://user-images.githubusercontent.com/61938694/191407849-6d0376c7-05b2-43cd-a75c-1280b0e33af1.png"
   width="300">
</a>

<hr>

## Train a Diffusion Model on your own data:

### Conditional Training
1. (optional) Configure Hyperparameters in ```ddpm_conditional.py```
2. Set path to dataset in ```ddpm_conditional.py```
3. ```python ddpm_conditional.py```

## Sampling
The following examples show how to sample images using the models trained in the video on the [fer-2013](https://www.kaggle.com/datasets/deadskull7/fer2013). You can download the checkpoints for the models [here]().

### Conditional Model
This model was trained on [fer-2013 48x48](https://www.kaggle.com/datasets/joaopauloschuler/cifar10-64x64-resized-via-cai-super-resolution) with 7 classes ```"Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"```
```python
    n = 6
    device = "cuda"
    model = UNet_conditional(num_classes=10, c_in=1, c_out=1).to(device)
    model.load_state_dict(ckpt)
    diffusion = Diffusion(img_size=64, device=device)
    labels = torch.Tensor([6] * n).long().to(device)
    x = diffusion.sample(model, n, labels, cfg_scale=3)
    plot_images(x)
```
