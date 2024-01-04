import argparse, torch, gc, os
import torch.nn.functional as F
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel 
from torchvision import transforms
from torch.utils.data import DataLoader

from image_meta_dataset import ImageMetaDataset

## for the sake of memory
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"
torch.cuda.init()  # Initialize CUDA
torch.cuda.empty_cache()  # Clear cache if needed

## parsing arguments
parser = argparse.ArgumentParser(description="Finetuning diffusion model for image generation")

parser.add_argument(
    "--base_model", type=str, default="CompVis/stable-diffusion-v1-4", 
    help="the diffusion model to fine-tune")

parser.add_argument(
    "--epochs", type=int, default=10, 
    help="Number of total epochs for training, default value: 10")

parser.add_argument(
    "--batch_size", type=int, default=16, 
    help="Batch size for training, default value: 16")

parser.add_argument(
    "--resolution", type=int, default=256, 
    help="The resolution for images")

parser.add_argument(
    "--learning_rate", type=float, default=1e-4, 
    help="Initial learning rate for training")

parser.add_argument(
    "--csv_file", type=str, 
    help="csv file with image text and path")

parser.add_argument(
    "--image_text_col", type=str, default='Title', 
    help="column name in the csv file for the image text")

parser.add_argument(
    "--image_file_col", type=str, default='path',
    help="column name in the csv file for the image path")

parser.add_argument(
    "--output_dir", type=str, 
    help="Direction to save the fine-tuned model")

args = parser.parse_args()

# data preparation
print('Data preparation..')

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((args.resolution, args.resolution), antialias = True),
    transforms.CenterCrop(args.resolution),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize([0.5], [0.5])
])

comic_dataset = ImageMetaDataset(
    csv_file = args.csv_file, 
    meta_columns = [args.image_text_col], 
    path_column = args.image_file_col,
    transform = img_transform
)

dataloader = DataLoader(comic_dataset, batch_size = args.batch_size, shuffle = True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## initilize diffusion models
print('Initializing diffusion models..')

tokenizer = CLIPTokenizer.from_pretrained(
    args.base_model, 
    subfolder="tokenizer")

text_encoder = CLIPTextModel.from_pretrained(
    args.base_model, 
    subfolder="text_encoder").to(device)

vae = AutoencoderKL.from_pretrained(
    args.base_model, 
    subfolder="vae").to(device)

noise_scheduler = DDPMScheduler.from_pretrained(
    args.base_model, 
    subfolder="scheduler")

unet = UNet2DConditionModel.from_pretrained(
	args.base_model,
	subfolder="unet").to(device)

# training unet model only
text_encoder.requires_grad_(False)
vae.requires_grad_(False)
unet.train()

## training start
print('Start fine-tuning..')

optimizer = torch.optim.AdamW(
    unet.parameters(),
    lr=args.learning_rate)

n_steps = len(dataloader)

for epoch in range(args.epochs):
    for i, data in enumerate(dataloader):
        images = data['image'].to(device)
        
        inputs = tokenizer(
            data[args.image_text_col], 
            max_length=tokenizer.model_max_length, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt")
        
        input_ids = inputs.input_ids.to(device)
        
        latents = vae.encode(images).latent_dist.sample()
        latents = latents * vae.config.scaling_factor
        noise = torch.randn_like(latents)
        batch_size = latents.shape[0]
        timesteps = torch.randint(0, 
                                  noise_scheduler.config.num_train_timesteps, 
                                  (batch_size,), 
                                  device=latents.device)
        
        timesteps = timesteps.long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        del latents
        torch.cuda.empty_cache()
        
        encoder_hidden_states = text_encoder(input_ids)[0]

        print(torch.cuda.mem_get_info())
        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
        del noisy_latents, timesteps, encoder_hidden_states
        torch.cuda.empty_cache()

        # compute loss and gradients
        loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
        loss.backward()
        
        # update parameters
        optimizer.step()
        optimizer.zero_grad()
        
        # print status
        if i % 10 == 0:
            print(f'epoch: [{epoch+1}/{args.epochs}]')
            print(f'[{i+1}/{n_steps}]')    
            print(f'total loss: {loss.item()}')
        
        gc.collect()
        torch.cuda.empty_cache()

## saving fine-tuned model
pipeline = StableDiffusionPipeline.from_pretrained(
    args.base_model,
    text_encoder=text_encoder,
    vae=vae,
    unet=unet
)

pipeline.save_pretrained(args.output_dir)