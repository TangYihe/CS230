# Facial Expression Manipulation with Conditional Diffusion Model

This is our CS230 final project code base. 

To see a demo of how our trained model could manipulate emotions for face images, please follow `demo.ipynb` for an interactive experience.  

To run experiements of the diffusion model (with semantic encoding), first run `python fer_data.py` to create dataloaders for datasets, then run `python ddpm_conditional --use_sem='vae'` (if prefer to use CNN semantic encoders, pass in `--use_sem='cnn'`, else if would like to view the baseline model, pass in `--use_sem='None'`)

### Referenced implementations
During development we referenced to the VAE implementation by Subramanian, A.K. at https://github.com/AntixK/PyTorch-VAE, and the DDPM implementation by Capelle, T. at https://github.com/tcapelle/Diffusion-Models-pytorch. Part of the CNN model is from Yihe's previous course assignments which is not publically released.  
Other than these, we implemented the FER dataset manipulation pipeline (jointly with Yihe's final project in CS229), evaluation and visualization pipeline independently, as well as independently integrated the semantic encoder to the current DDPM. 
