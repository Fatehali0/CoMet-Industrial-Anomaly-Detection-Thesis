# ------------------------------------------------------------------
# CoMet: Towards Real Unsupervised Anomaly Detection Via Confident Meta-Learning
# # Import libraries and project modules needed to configure, run, and evaluate CoMet.

import logging
import os
import sys

import click
import numpy as np
import torch

sys.path.append("src")
import backbones
import common
import metrics
import utils
import comet

LOGGER = logging.getLogger(__name__)

# Save evaluation results for all processed datasets into the final results file.
def persist_results(run_save_path, results):
    if not results:
        return
# extracts metric names ,extracts dataset names ,extracts scores calls

    metric_names = [k for k in results[-1].keys() if k != "dataset_name"]
    dataset_names = [r["dataset_name"] for r in results]
    scores = [list(r.values())[1:] for r in results]

    utils.compute_and_store_final_results(
        run_save_path,
        scores,
        column_names=metric_names,
        row_names=dataset_names,
    )
# Load optional pre-generated anomalous images and masks for augmented training.

def load_augmented_data(augmented_dir, batch_size, resize=329, imagesize=288):
    """Load pre-generated anomalous images for training."""
    if not augmented_dir or not os.path.exists(augmented_dir):
        return None

    img_dir = os.path.join(augmented_dir, "imgs")
    mask_dir = os.path.join(augmented_dir, "masks")

    if not os.path.exists(img_dir):
        return None

    from torchvision import transforms
    from PIL import Image

    transform_img = transforms.Compose([
        transforms.Resize((int(resize * 2.5 + 0.5), resize)),
        transforms.CenterCrop((int(imagesize * 2.5 + 0.5), imagesize)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_mask = transforms.Compose([
        transforms.Resize((int(resize * 2.5 + 0.5), resize)),
        transforms.CenterCrop((int(imagesize * 2.5 + 0.5), imagesize)),
        transforms.ToTensor(),
    ])

    class AugmentedDataset(torch.utils.data.Dataset):
        def __init__(self, img_dir, mask_dir):
            self.img_dir = img_dir
            self.mask_dir = mask_dir
            self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])

        def __len__(self):
            return len(self.img_files)


        def __getitem__(self, idx):
            img_name = self.img_files[idx]
            img_path = os.path.join(self.img_dir, img_name)
            mask_path = os.path.join(self.mask_dir, img_name)

            image = Image.open(img_path).convert("RGB")
            image = transform_img(image)

            if os.path.exists(mask_path):
                mask = Image.open(mask_path)
                mask = transform_mask(mask)
            else:
                mask = torch.ones([1, *image.size()[1:]])

            return {
                "image": image,
                "mask": mask,
                "is_anomaly": True,
                "image_path": img_path,
            }
        
        dataset = AugmentedDataset(img_dir, mask_dir)
            
    #creates a dataloader for those augmented images
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

# Map dataset names from the command line to their corresponding dataset classes.
_DATASETS = {
    "mvtec": ["datasets.mvtec", "MVTecDataset"],
    "visa": ["datasets.visa", "VisaDataset"],
    "sdd2": ["datasets.sdd2", "SDD2Dataset"],
    "pothole": ["datasets.pothole", "PotholeDataset"]
}

# Define the top-level command-line interface and global experiment settings.
# “The main() click group defines the global experiment options and allows the
#  net and dataset subcommands to be chained together.”
@click.group(chain=True)
@click.option("--results_path", type=str, required=True)
@click.option("--gpu", type=int, default=[0], multiple=True)
@click.option("--seed", type=int, default=0)
@click.option("--log_group", type=str, default="group")
@click.option("--log_project", type=str, default="project")
@click.option("--run_name", type=str, default="test")
@click.option("--test", is_flag=True)
@click.option("--save_segmentation_images", is_flag=True, default=False)
@click.option("--augmented_path", type=str, default='')
@click.option("--checkpoint_run_name", type=str, default="")

def main(**kwargs):
    pass


# Main execution controller: create loaders and models,
#  then run training or testing and save results.
@main.result_callback()
def run(
    methods,
    results_path,
    gpu,
    seed,
    log_group,
    log_project,
    run_name,
    test,
    save_segmentation_images,
    augmented_path,
    checkpoint_run_name
):
    # Collect helper functions returned by the chained net and dataset commands
    methods = dict(methods)
    # Create the folder where this experiment will store checkpoints and results.

    run_save_path = utils.create_storage_folder(
        results_path, log_project, log_group, run_name, mode="overwrite"
    )
# Select the execution device and prepare dataset loaders for the selected classes.
    device = utils.set_torch_device(gpu)
    dataloaders_list = methods["get_dataloaders"](seed)

    results = []
    # Process each selected dataset/class one by one.
    for dataloader_count, dataloaders in enumerate(dataloaders_list):
        dataset_name = dataloaders["training"].name
        LOGGER.info(f"Evaluating dataset [{dataset_name}] ({dataloader_count + 1}/{len(dataloaders_list)})")

        utils.fix_seeds(seed, device)
        imagesize = dataloaders["training"].dataset.imagesize
        # Build the CoMet model configuration for the current dataset
        # .This builds one or more CoMet model objects depending on the
        #  selected backbone settings.
        comet_list = methods["get_comet"](imagesize, device)

        models_dir = os.path.join(run_save_path, "models")
        os.makedirs(models_dir, exist_ok=True)

        for i, comet_model in enumerate(comet_list):
            if comet_model.backbone.seed is not None:
                utils.fix_seeds(comet_model.backbone.seed, device)

            checkpoint_model_dir = None
            if checkpoint_run_name:
                checkpoint_model_dir = os.path.join(
                    results_path,
                    log_project,
                    log_group,
                    checkpoint_run_name,
                    "models",
                    str(i),
                )
# Assign a dedicated checkpoint/results directory to the current CoMet model
# .Each model gets its own save directory for checkpoints and outputs.

            comet_model.set_model_dir(
                os.path.join(models_dir, str(i)),
                dataset_name,
                checkpoint_model_dir=checkpoint_model_dir,
            )
# Either train the model or load/test it depending on the selected run mode.
            if not test:
                augmented_loader = None
                if augmented_path:
                    augmented_loader = load_augmented_data(
                        augmented_path,
                        dataloaders["training"].batch_size,
                        resize=329,
                        imagesize=288
                    )
                i_auroc, p_auroc, pro = comet_model.train(
                    dataloaders["training"],
                    dataloaders["testing"],
                    augmented_loader
                )
            else:
                i_auroc, p_auroc, pro = comet_model.test(
                    dataloaders["training"],
                    dataloaders["testing"],
                    save_segmentation_images
                )
# Store the final metrics for the current dataset/class.
            result = {
                "dataset_name": dataset_name,
                "image_auroc": i_auroc,
                "Pixel_auroc": p_auroc,
                "Pro": pro,
            }
            results.append(result)
            # Save the updated results table and print the main evaluation metrics.
            persist_results(run_save_path, results)

            for key, val in result.items():
                if key != "dataset_name":
                    LOGGER.info(f"{key}: {val:.3f}")

        LOGGER.info("-----\n")

    persist_results(run_save_path, results)
# Define model architecture, training hyperparameters, and ablation flags for CoMet
# .This command configures how CoMet should be built.
@main.command("net")
@click.option("--backbone_names", "-b", type=str, multiple=True, default=["wide_resnet50_2"])
@click.option("--layers_to_extract_from", "-le", type=str, multiple=True, default=["layer2", "layer3"])
@click.option("--pretrain_embed_dimension", type=int, default=1024)
@click.option("--target_embed_dimension", type=int, default=1024)
@click.option("--patchsize", type=int, default=3)
@click.option("--embedding_size", type=int, default=256)
@click.option("--meta_epochs", type=int, default=40)
@click.option("--aed_meta_epochs", type=int, default=1)
@click.option("--gan_epochs", type=int, default=4)
@click.option("--dsc_layers", type=int, default=2)
@click.option("--dsc_hidden", type=int, default=None)
@click.option("--noise_std", type=float, default=0.05)
@click.option("--dsc_margin", type=float, default=0.8)
@click.option("--dsc_lr", type=float, default=0.0002)
@click.option("--auto_noise", type=float, default=0)
@click.option("--train_backbone", is_flag=True)
@click.option("--cos_lr", is_flag=True)
@click.option("--pre_proj", type=int, default=1)
@click.option("--proj_layer_type", type=int, default=0)
@click.option("--mix_noise", type=int, default=1)
# Existing switch for enabling or disabling confidence learning.
@click.option("--use_confidence_learning/--no_confidence_learning", default=True)
# Separate switch for ablating model-confidence regularization only.
@click.option("--use_model_confidence/--no_model_confidence", default=True)
# Separate switch for ablating data-confidence sample weighting only.
@click.option("--use_data_confidence/--no_data_confidence", default=True)
# Existing switch for enabling or disabling the meta-learning process.
@click.option("--use_meta_learning/--no_meta_learning", default=True)
@click.option("--fixed_lambda", type=float, default=0.01)
@click.option("--use_mask_refinement/--no_mask_refinement", default=False)
@click.option("--mask_threshold", type=float, default=0.6)
@click.option("--min_component_area", type=int, default=20)
def net(
    backbone_names,
    layers_to_extract_from,
    pretrain_embed_dimension,
    target_embed_dimension,
    patchsize,
    embedding_size,
    meta_epochs,
    aed_meta_epochs,
    gan_epochs,
    noise_std,
    dsc_layers,
    dsc_hidden,
    dsc_margin,
    dsc_lr,
    auto_noise,
    train_backbone,
    cos_lr,
    pre_proj,
    proj_layer_type,
    mix_noise,
    use_confidence_learning,
    use_data_confidence,
    use_model_confidence,
    use_meta_learning,
    use_mask_refinement,
    mask_threshold,
    min_component_area,
    fixed_lambda,
):
    # Organize feature-extraction layers for each selected backbone.
    backbone_names = list(backbone_names)
    if len(backbone_names) > 1:
        layers_coll = [[] for _ in backbone_names]
        for layer in layers_to_extract_from:
            idx = int(layer.split(".")[0])
            layer_name = ".".join(layer.split(".")[1:])
            layers_coll[idx].append(layer_name)
    else:
        layers_coll = [layers_to_extract_from]
# Build and initialize CoMet model objects using
#  the selected architecture and training settings.creates comet.CoMet(device) loads backbone
# ,calls model.load(...) passes all settings into the model
#  returns model list 

    def get_comet(input_shape, device):
        models = []
        for name, layers in zip(backbone_names, layers_coll):
            seed = None
            if ".seed-" in name:
                name, seed = name.split(".seed-")[0], int(name.split("-")[-1])
            backbone = backbones.load(name)
            backbone.name, backbone.seed = name, seed

            model = comet.CoMet(device)
            # Pass all selected experiment settings from main.py into the CoMet implementation.
            model.load (
             backbone=backbone,
                layers_to_extract_from=layers,
                device=device,
                input_shape=input_shape,
                pretrain_embed_dimension=pretrain_embed_dimension,
                target_embed_dimension=target_embed_dimension,
                patchsize=patchsize,
                embedding_size=embedding_size,
                meta_epochs=meta_epochs,
                aed_meta_epochs=aed_meta_epochs,
                gan_epochs=gan_epochs,
                noise_std=noise_std,
                dsc_layers=dsc_layers,
                dsc_hidden=dsc_hidden,
                dsc_margin=dsc_margin,
                dsc_lr=dsc_lr,
                auto_noise=auto_noise,
                train_backbone=train_backbone,
                cos_lr=cos_lr,
                pre_proj=pre_proj,
                proj_layer_type=proj_layer_type,
                mix_noise=mix_noise,
                # Pass the confidence-learning setting into the CoMet model.
                use_confidence_learning=use_confidence_learning,
                # Pass the model-confidence ablation flag into CoMet.
                use_model_confidence=use_model_confidence,
                # Pass the data-confidence ablation flag into CoMet.
                use_data_confidence=use_data_confidence,
                # Pass the meta-learning setting into the CoMet model.
                use_meta_learning=use_meta_learning,
                fixed_lambda=fixed_lambda,
                use_mask_refinement=use_mask_refinement,
                mask_threshold=mask_threshold,
                min_component_area=min_component_area,
            )
            models.append(model)
        return models

    return ("get_comet", get_comet)

# Define dataset path, class selection, loader settings, and image augmentations.
@main.command("dataset")
@click.argument("name", type=str)
@click.argument("data_path", type=click.Path(exists=True, file_okay=False))
@click.option("--subdatasets", "-d", multiple=True, type=str, required=True)
@click.option("--train_val_split", type=float, default=1.0)
@click.option("--batch_size", default=2, type=int)
@click.option("--num_workers", default=8, type=int)
@click.option("--resize", default=256, type=int)
@click.option("--imagesize", default=224, type=int)
@click.option("--rotate_degrees", default=0, type=int)
@click.option("--translate", default=0.0, type=float)
@click.option("--scale", default=0.0, type=float)
@click.option("--brightness", default=0.0, type=float)
@click.option("--contrast", default=0.0, type=float)
@click.option("--saturation", default=0.0, type=float)
@click.option("--gray", default=0.0, type=float)
@click.option("--hflip", default=0.0, type=float)
@click.option("--vflip", default=0.0, type=float)
@click.option("--augment", is_flag=True)
def dataset(
    name, data_path, subdatasets, train_val_split, batch_size, num_workers,
    resize, imagesize, rotate_degrees, translate, scale, brightness,
    contrast, saturation, gray, hflip, vflip, augment
):
    # Import the correct dataset implementation based on the selected dataset name.
    dataset_info = _DATASETS[name]
    dataset_lib = __import__(dataset_info[0], fromlist=[dataset_info[1]])
# Build train, test, and optional validation dataloaders for each selected class
# .This is the real data-builder function.

    def get_dataloaders(seed):
        dataloaders = []
        for sub in subdatasets:
            # Create dataset objects for training and testing splits.
            train_dataset = dataset_lib.__dict__[dataset_info[1]](
                data_path, classname=sub, resize=resize, train_val_split=train_val_split,
                imagesize=imagesize, split=dataset_lib.DatasetSplit.TRAIN, seed=seed,
                rotate_degrees=rotate_degrees, translate=translate, scale=scale,
                brightness_factor=brightness, contrast_factor=contrast,
                saturation_factor=saturation, gray_p=gray, h_flip_p=hflip,
                v_flip_p=vflip, augment=augment
            )
            test_dataset = dataset_lib.__dict__[dataset_info[1]](
                data_path, classname=sub, resize=resize, imagesize=imagesize,
                split=dataset_lib.DatasetSplit.TEST, seed=seed
            )

            loader_kwargs = {
                "batch_size": batch_size,
                "num_workers": num_workers,
                "pin_memory": torch.cuda.is_available(),
            }
            if num_workers > 0:
                loader_kwargs["prefetch_factor"] = 2
           # Wrap dataset objects into PyTorch dataloaders for batch processing.
            train_loader = torch.utils.data.DataLoader(
                train_dataset, shuffle=True, **loader_kwargs
            )
            test_loader = torch.utils.data.DataLoader(
                test_dataset, shuffle=False, **loader_kwargs
            )
       # Package the loaders in a standard dictionary format used by the main execution flow
       # .This prepares the structure that run() expects.
            train_loader.name = f"{name}_{sub}" if sub else name
            dataloader_dict = {
                "training": train_loader,
                "testing": test_loader,
                "validation": None
            }
            if train_val_split < 1:
                val_dataset = dataset_lib.__dict__[dataset_info[1]](
                    data_path, classname=sub, resize=resize, train_val_split=train_val_split,
                    imagesize=imagesize, split=dataset_lib.DatasetSplit.VAL, seed=seed
                )
                val_loader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=torch.cuda.is_available(),
                )
                dataloader_dict["validation"] = val_loader

            dataloaders.append(dataloader_dict)
        return dataloaders

    return ("get_dataloaders", get_dataloaders)

# Start the command-line interface and enable runtime logging
#  when the script is executed directly.
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    main()

