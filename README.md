# fine_tune_diffusion_model

## usage
```bash
python finetune_diffusion_model.py --csv_file <input_file> ---output_dir <output_directory>
```

## Options

- `--csv_file`: the csv file must have two columns, one speficies the text description (default: Title) of the image and another specifies the path to the image (default: path)
- `--output_file`: directory to save the fune-tuned diffusion model

## Other options
- `--image_text_col`: the column name in teh csv file tp specify the text description of the image
- `--image_file_col`: the column name in the csv file to specify the path to the image file
- `--base_model`: the base model to fine-tune, default: CompVis/stable-diffusion-v1-4
- `--epochs`: number of epochs for training, default: 10
