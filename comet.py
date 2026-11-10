# ------------------------------------------------------------------
# CoMet: Towards Real Unsupervised Anomaly Detection Via Confident Meta-Learning (https://openaccess.thecvf.com/content/ICCV2025/html/Aqeel_Towards_Real_Unsupervised_Anomaly_Detection_Via_Confident_Meta-Learning_ICCV_2025_paper.html)
# Licensed under the MIT License
# ------------------------------------------------------------------

"""detection methods."""
import logging
import os
import pickle
from collections import OrderedDict

import math
from networkx import is_planar
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch.utils.tensorboard import SummaryWriter
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from collections import deque
import psutil

import common
import metrics
from utils import plot_segmentation_images
# Start the command-line interface and enable runtime  logging when the script is executed directly.
LOGGER = logging.getLogger(__name__)
# Initialize learnable layer weights before training starts.
def init_weight(m):

    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)
# Compute model confidence from train/validation loss behavior and use it for
#  confidence-aware regularization. this class measures how reliable the current training behavior is
class ConfidenceLearning:
    # Initialize the confidence-learning module and its loss-history buffers.
    def __init__(self, model, window_size=10, l2_lambda=0.01):
        """
        Initialize confidence learning module
        
        Args:
            model: The model to apply confidence learning to
            window_size: Number of batches to consider for covariance calculation
            l2_lambda: L2 regularization strength
        """
        self.model = model
        self.window_size = window_size
        self.l2_lambda = l2_lambda
        
        # Tracking loss values for covariance calculation
        self.train_losses = deque(maxlen=window_size)
        self.val_losses = deque(maxlen=window_size)
        
        # Tracking confidence metrics
        self.confidence_scores = []
        self.covariance_matrices = []
    # Update normalized train and validation loss history for confidence estimation.
    def update_losses(self, train_loss, val_loss):
        # Normalize losses to zero mean, unit variance
        if not hasattr(self, 'loss_stats'):
            self.loss_stats = {'mean': 0.0, 'std': 1.0, 'count': 0}
        
        # Online update of mean/std
        count = self.loss_stats['count']
        mean = self.loss_stats['mean']
        std = self.loss_stats['std']
        
        new_count = count + 1
        new_mean = (count * mean + train_loss + val_loss) / (new_count * 2)
        new_std = ((count * std**2 + (train_loss - new_mean)**2 + (val_loss - new_mean)**2) / (new_count * 2))**0.5
        if new_std < 1e-6:
            new_std = 1.0
        
        self.loss_stats['mean'] = new_mean
        self.loss_stats['std'] = new_std
        self.loss_stats['count'] = new_count
        
        # Z-score normalize
        train_loss = (train_loss - new_mean) / new_std
        val_loss = (val_loss - new_mean) / new_std
        
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
    
    def compute_covariance(self):
        """
        Compute the covariance matrix between training and validation losses
        Returns None if not enough data points
        """
        if len(self.train_losses) < 2 or len(self.val_losses) < 2:
            return None
            
        # Convert deques to numpy arrays
        train_losses = np.array(self.train_losses)
        val_losses = np.array(self.val_losses)
        
        # Ensure we have matching number of samples
        min_samples = min(len(train_losses), len(val_losses))
        train_losses = train_losses[-min_samples:]
        val_losses = val_losses[-min_samples:]
        
        # Stack losses to create multivariate data
        data = np.vstack([train_losses, val_losses]).T
        
        # Compute covariance matrix
        cov_matrix = np.cov(data, rowvar=False)
        return cov_matrix
    # Convert covariance of train/validation losses into a model-confidence score.
    def compute_confidence_metric(self):
        """
        Compute confidence metric based on determinant of covariance matrix
        Lower determinant = higher confidence (less uncertainty)
        """
        cov_matrix = self.compute_covariance()
        if cov_matrix is None:
            return 0.5  # Default value when not enough data
            
        # Ensure numerical stability
        cov_matrix = cov_matrix + np.eye(cov_matrix.shape[0]) * 1e-6
        
        # Compute determinant - lower means higher confidence
        det = np.linalg.det(cov_matrix)
        
        # Invert and normalize to get confidence score (0-1 range)
        confidence = 1.0 / (1.0 + det)
        
        # Store the confidence score and covariance matrix
        self.confidence_scores.append(confidence)
        self.covariance_matrices.append(cov_matrix)
        
        return confidence
        # Apply confidence-weighted regularization to the training loss.
    def apply_l2_regularization(self, loss, confidence=None):
        """
        Apply L2 regularization weighted by confidence
        
        Args:
            loss: The current loss value
            confidence: Optional confidence score (if None, will compute)
        """
        if confidence is None:
            confidence = self.compute_confidence_metric()
        
        # Apply stronger regularization when confidence is low
        adaptive_lambda = self.l2_lambda * (1.0 - confidence)
        
        # Compute L2 regularization term
        l2_reg = 0.0
        for param in self.model.parameters():
            l2_reg += torch.norm(param, 2) ** 2
            
        # Add weighted regularization to loss
        regularized_loss = loss + adaptive_lambda * l2_reg
        
        return regularized_loss, confidence
    
    def get_confidence_history(self): # “This function is used to Return the stored history of model-confidence scores.
        """Return the history of confidence scores"""
        return self.confidence_scores
    
    def reset(self): #“This function clears all previously stored loss and confidence history before a fresh training phase.
        """Reset loss history for clean start"""
        self.train_losses.clear()
        self.val_losses.clear()
        self.confidence_scores.clear()
        self.covariance_matrices.clear()
# Main scoring network that converts patch embeddings into anomaly-discrimination scores so simply say:
#.backbone extracts features CoMet converts them into embeddings Discriminator looks at those embeddingsthen gives an anomaly-related score

class Discriminator(torch.nn.Module): # This defines a PyTorch neural network module called Discriminator.

    def __init__(self, in_planes, n_layers=1, hidden=None):# This constructor builds the discriminator network.

        super(Discriminator, self).__init__()

        _hidden = in_planes if hidden is None else hidden# If no hidden size is given, use the input size as the hidden size.

        self.body = torch.nn.Sequential()#“This creates the main sequence of layers that will process the feature embeddings.”

        for i in range(n_layers - 1):#If the discriminator has multiple layers, create each hidden block here.

            _in = in_planes if i == 0 else _hidden #For the first layer, input size is the original feature size. for later i/p from previous hidden layer

            _hidden = int(_hidden // 1.5) if hidden is None else hidden
            self.body.add_module(
                'block%d' % (i + 1),
                torch.nn.Sequential(
                    torch.nn.Linear(_in, _hidden),
                    torch.nn.BatchNorm1d(_hidden),
                    torch.nn.LeakyReLU(0.2)
                )
            )
        self.tail = torch.nn.Linear(_hidden, 1, bias=False)
        self.apply(init_weight)

    def forward(self, x):
        x = self.body(x)
        x = self.tail(x)
        return x

# Project feature embeddings into the representation space used for anomaly training.
class Projection(torch.nn.Module):
    def __init__(self, in_planes, out_planes=None, n_layers=1, layer_type=0):
        super(Projection, self).__init__()
        
        if out_planes is None:
            out_planes = in_planes
        self.layers = torch.nn.Sequential()
        _in = None
        _out = None
        for i in range(n_layers):
            _in = in_planes if i == 0 else _out
            _out = out_planes 
            self.layers.add_module(f"{i}fc", 
                                   torch.nn.Linear(_in, _out))
            if i < n_layers - 1:
                # if layer_type > 0:
                #     self.layers.add_module(f"{i}bn", 
                #                            torch.nn.BatchNorm1d(_out))
                if layer_type > 1:
                    self.layers.add_module(f"{i}relu",
                                           torch.nn.LeakyReLU(.2))
        self.apply(init_weight)
    def forward(self, x):
        
        # x = .1 * self.layers(x) + x
        x = self.layers(x)
        return x

# Handle TensorBoard logging during training.This is a helper for logging training progress.

class TBWrapper:
    
    def __init__(self, log_dir):
        self.g_iter = 0
        # Ensure the TensorBoard log directory exists before creating the writer.
        os.makedirs(log_dir, exist_ok=True)
        self.logger = SummaryWriter(log_dir=log_dir)
    
    def step(self):
        self.g_iter += 1
# Main CoMet class: handles model setup, training, testing, prediction,
#  confidence, and meta-learning.
class CoMet(torch.nn.Module):
    # Initialize the CoMet object and store the execution device.
    def __init__(self, device):
        """anomaly detection class."""
        super(CoMet, self).__init__()
        self.device = device
# Build and configure all core CoMet components, training settings, and ablation switches.
    def load(
        self,
        backbone,
        layers_to_extract_from,
        device,
        input_shape,
        pretrain_embed_dimension, # 1536
        target_embed_dimension, # 1536
        patchsize=3, # 3
        patchstride=1, 
        embedding_size=None, # 256
        meta_epochs=1, # 40
        aed_meta_epochs=1,
        gan_epochs=1, # 4
        noise_std=0.05,
        mix_noise=1,
        noise_type="GAU",
        dsc_layers=2, # 2
        dsc_hidden=None, # 1024
        dsc_margin=.8, # .5
        dsc_lr=0.0002,
        train_backbone=False,
        auto_noise=0,
        cos_lr=False,
        lr=1e-3,
        pre_proj=0, # 1
        proj_layer_type=0,
        confidence_window_size=5,  
        l2_lambda=0.01,
        use_confidence_learning=True,
        use_model_confidence=True,
        use_data_confidence=True,
        use_meta_learning=True,
        use_mask_refinement=False,
        mask_threshold=0.6,
        min_component_area=20,
        fixed_lambda=0.01,
        **kwargs,
    ):
        pid = os.getpid()
        def show_mem():
            return(psutil.Process(pid).memory_info())
# Build the feature extraction and embedding pipeline used by CoMet.
        self.backbone = backbone.to(device)
        self.backbone_name = getattr(backbone, "name", backbone.__class__.__name__)
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape

        self.device = device
        self.patch_maker = PatchMaker(patchsize, stride=patchstride)

        self.forward_modules = torch.nn.ModuleDict({})
          # This module takes the backbone and extracts intermediate outputs from the selected layers.
        feature_aggregator = common.NetworkFeatureAggregator (
            self.backbone, self.layers_to_extract_from, self.device, train_backbone
        )
        feature_dimensions = feature_aggregator.feature_dimensions(input_shape)
        self.forward_modules["feature_aggregator"] = feature_aggregator

        preprocessing = common.Preprocessing(
            feature_dimensions, pretrain_embed_dimension
        )
        self.forward_modules["preprocessing"] = preprocessing

        self.target_embed_dimension = target_embed_dimension
        preadapt_aggregator = common.Aggregator(
            target_dim=target_embed_dimension
        )

        _ = preadapt_aggregator.to(self.device)

        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

        self.anomaly_segmentor = common.RescaleSegmentor(
            device=self.device, target_size=input_shape[-2:]
        )

        self.embedding_size = embedding_size if embedding_size is not None else self.target_embed_dimension
        self.meta_epochs = meta_epochs
        self.lr = lr
        self.cos_lr = cos_lr
        self.train_backbone = train_backbone
        if self.train_backbone:
            self.backbone_opt = torch.optim.AdamW(self.forward_modules["feature_aggregator"].backbone.parameters(), lr)
        # AED
        self.aed_meta_epochs = aed_meta_epochs
# Create the projection  modules used for anomaly scoring.
        self.pre_proj = pre_proj
        if self.pre_proj > 0:
            self.pre_projection = Projection(self.target_embed_dimension, self.target_embed_dimension, pre_proj, proj_layer_type)
            self.pre_projection.to(self.device)
            self.proj_opt = torch.optim.AdamW(self.pre_projection.parameters(), lr*.1)
# Create the  discriminator modules used for anomaly scoring.
        # Discriminator
        self.auto_noise = [auto_noise, None]
        self.dsc_lr = dsc_lr
        self.gan_epochs = gan_epochs
        self.mix_noise = mix_noise
        self.noise_type = noise_type
        self.noise_std = noise_std
        self.discriminator = Discriminator(self.target_embed_dimension, n_layers=dsc_layers, hidden=dsc_hidden)
        self.discriminator.to(self.device)
        # Prepare the optimizer and learning-rate scheduler for discriminator training.
        self.dsc_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.dsc_lr, weight_decay=1e-5)
        self.dsc_schl = torch.optim.lr_scheduler.CosineAnnealingLR(self.dsc_opt, (meta_epochs - aed_meta_epochs) * gan_epochs, self.dsc_lr*.4)
        self.dsc_margin= dsc_margin 

        self.model_dir = ""
        self.dataset_name = ""
        self.tau = 1
        self.logger = None

         # At the end, add the confidence module
        from collections import deque
        # Store confidence and meta-learning switches so
        #  later training blocks can enable or disable each mechanism.
        self.confidence_window_size = confidence_window_size
        self.l2_lambda = l2_lambda
        self.confidence_module = None
        # Store whether confidence learning should remain active during training.
        self.use_confidence_learning = use_confidence_learning
        self.use_model_confidence = use_model_confidence
        self.use_data_confidence = use_data_confidence
        # Store whether inner-loop and outer-loop meta-learning should be used.
        self.use_meta_learning = use_meta_learning
        self.fixed_lambda = fixed_lambda
        self.current_adaptive_lambda = fixed_lambda
        self.use_mask_refinement = use_mask_refinement
        self.mask_threshold = mask_threshold
        self.min_component_area = min_component_area
        self._debug_feature_stats_logged = False
        self._debug_prediction_stats_logged = False

        LOGGER.info(f"backbone: {self.backbone_name}")
        LOGGER.info(f"layers_to_extract_from: {self.layers_to_extract_from}")
        # Runtime check: confirm whether model confidence and data confidence are enabled.
        LOGGER.info(f"use_model_confidence: {self.use_model_confidence}")
        LOGGER.info(f"use_data_confidence: {self.use_data_confidence}")




        self.model_dir = ""
        self.dataset_name = ""
        self.tau = 1
        self.logger = None
     # Create and assign the checkpoint, output, and TensorBoard directories for this run.
    def set_model_dir(self, model_dir, dataset_name, checkpoint_model_dir=None):

        self.model_dir = model_dir 
        os.makedirs(self.model_dir, exist_ok=True)
        self.output_ckpt_dir = os.path.join(self.model_dir, dataset_name)
        os.makedirs(self.output_ckpt_dir, exist_ok=True)
        self.ckpt_dir = (
            os.path.join(checkpoint_model_dir, dataset_name)
            if checkpoint_model_dir is not None
            else self.output_ckpt_dir
        )
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.tb_dir = os.path.join(self.output_ckpt_dir, "tb")
        os.makedirs(self.tb_dir, exist_ok=True)
        self.logger = TBWrapper(self.tb_dir)
    
# Extract feature embeddings either from a dataloader or directly from input tensors.
    def embed(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            features = []
            for image in data:
                if isinstance(image, dict):
                    image = image["image"]
                    input_image = image.to(torch.float).to(self.device)
                with torch.no_grad():
                    features.append(self._embed(input_image))
            return features
        return self._embed(data)
# Convert input images into aligned patch-level feature embeddings for anomaly scoring.
    def _embed(self, images, detach=True, provide_patch_shapes=False, evaluation=False):
        """Returns feature embeddings for images."""

        B = len(images)
        if not evaluation and self.train_backbone:
            self.forward_modules["feature_aggregator"].train()
            features = self.forward_modules["feature_aggregator"](images, eval=evaluation)
        else:
            _ = self.forward_modules["feature_aggregator"].eval()
            with torch.no_grad():
                features = self.forward_modules["feature_aggregator"](images)
 
                # DINOv3 ViT features may include a CLS token.
                # Remove it before reshaping tokens into a spatial feature map.
        features = [features[layer] for layer in self.layers_to_extract_from]

        for i, feat in enumerate(features):
            if len(feat.shape) == 3:
                B, L, C = feat.shape
                # Remove prefix tokens (CLS / register tokens) until the
                # remaining token count forms a square spatial grid.
                prefix_tokens = 0
                for possible_prefix in range(0, 10):
                    remaining = L - possible_prefix
                    if remaining > 0 and int(math.sqrt(remaining)) ** 2 == remaining:
                        prefix_tokens = possible_prefix
                        break

                if prefix_tokens > 0:
                    feat = feat[:, prefix_tokens:, :]
                    B, L, C = feat.shape

                spatial_size = int(math.sqrt(L))
                features[i] = feat.reshape(B, spatial_size, spatial_size, C).permute(0, 3, 1, 2)

        if not self._debug_feature_stats_logged:
            for layer_name, feat in zip(self.layers_to_extract_from, features):
                LOGGER.info(
                    f"Feature debug - {layer_name}: shape={tuple(feat.shape)}, "
                    f"min={feat.min().item():.6f}, max={feat.max().item():.6f}"
                )
            self._debug_feature_stats_logged = True
        features = [
            self.patch_maker.patchify(x, return_spatial_info=True) for x in features # herefeature maps become patches After feature maps are obtained, they are converted into patch-level features.
        ]
        patch_shapes = [x[1] for x in features]
        features = [x[0] for x in features]
        ref_num_patches = patch_shapes[0]

        for i in range(1, len(features)):
            _features = features[i]
            patch_dims = patch_shapes[i]

            # TODO(pgehler): Add comments
            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )
            _features = _features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            features[i] = _features
        features = [x.reshape(-1, *x.shape[-3:]) for x in features]
        
        features = self.forward_modules["preprocessing"](features)
        features = self.forward_modules["preadapt_aggregator"](features)       

        return features, patch_shapes  # At this point, the feature maps have already been processed into the final patch-level representation.


# Load a trained checkpoint, run prediction on test data,
#  evaluate metrics, and optionally save segmentations.
    def test(self, training_data, test_data, save_segmentation_images):

        ckpt_paths = [
            os.path.join(self.ckpt_dir, "ckpt.pth"),
            os.path.join(self.ckpt_dir, "models.ckpt"),
        ]
        for ckpt_path in ckpt_paths:
            if not os.path.exists(ckpt_path):
                continue

            state_dicts = torch.load(ckpt_path, map_location=self.device)
            if "discriminator" in state_dicts:
                self.discriminator.load_state_dict(state_dicts["discriminator"])
                if self.pre_proj > 0 and "pre_projection" in state_dicts:
                    self.pre_projection.load_state_dict(state_dicts["pre_projection"])
                break
            if "pretrained_enc" in state_dicts:
                self.feature_enc.load_state_dict(state_dicts["pretrained_enc"])
            if "pretrained_dec" in state_dicts:
                self.feature_dec.load_state_dict(state_dicts["pretrained_dec"])
            break

        aggregator = {"scores": [], "segmentations": [], "features": []}
        scores, segmentations, features, labels_gt, masks_gt = self.predict(test_data)
        aggregator["scores"].append(scores)
        aggregator["segmentations"].append(segmentations)
        aggregator["features"].append(features)

        scores = np.array(aggregator["scores"])
        min_scores = scores.min(axis=-1).reshape(-1, 1)
        max_scores = scores.max(axis=-1).reshape(-1, 1)
        scores = (scores - min_scores) / np.maximum(max_scores - min_scores, 1e-12)
        scores = np.asarray(np.mean(scores, axis=0), dtype=np.float32).reshape(-1)

        segmentations = np.array(aggregator["segmentations"])
        min_scores = (
            segmentations.reshape(len(segmentations), -1)
            .min(axis=-1)
            .reshape(-1, 1, 1, 1)
        )
        max_scores = (
            segmentations.reshape(len(segmentations), -1)
            .max(axis=-1)
            .reshape(-1, 1, 1, 1)
        )
        segmentations = (segmentations - min_scores) / np.maximum(max_scores - min_scores, 1e-12)
        segmentations = np.mean(segmentations, axis=0)

        anomaly_labels = labels_gt

        if save_segmentation_images:
            self.save_segmentation_images(test_data, segmentations, scores)
            
        auroc = metrics.compute_imagewise_retrieval_metrics(
            scores, anomaly_labels
        )["auroc"]

        pixel_scores = metrics.compute_pixelwise_retrieval_metrics(
            segmentations, masks_gt
        )
        full_pixel_auroc = pixel_scores["auroc"]
        pro = metrics.compute_pro(np.squeeze(np.array(masks_gt)), segmentations)
        if isinstance(pro, tuple):
            pro = pro[0]

        return auroc, full_pixel_auroc, pro
    
    def _evaluate(self, test_data, scores, segmentations, features, labels_gt, masks_gt):
        
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        img_min_scores = scores.min(axis=-1)
        img_max_scores = scores.max(axis=-1)
        scores = (scores - img_min_scores) / max(img_max_scores - img_min_scores, 1e-12)

        auroc = metrics.compute_imagewise_retrieval_metrics(
            scores, labels_gt 
        )["auroc"]

        if len(masks_gt) > 0:
            segmentations = np.array(segmentations)
            min_scores = (
                segmentations.reshape(len(segmentations), -1)
                .min(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            max_scores = (
                segmentations.reshape(len(segmentations), -1)
                .max(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            norm_segmentations = np.zeros_like(segmentations)
            for min_score, max_score in zip(min_scores, max_scores):
                norm_segmentations += (segmentations - min_score) / max(max_score - min_score, 1e-2)
            norm_segmentations = norm_segmentations / len(scores)

            pixel_scores = metrics.compute_pixelwise_retrieval_metrics(
                norm_segmentations, masks_gt)
            full_pixel_auroc = pixel_scores["auroc"]

            pro = metrics.compute_pro(np.squeeze(np.array(masks_gt)), 
                                norm_segmentations)
            if isinstance(pro, tuple):
                pro = pro[0]

        else:
            full_pixel_auroc = -1 
            pro = -1

        return auroc, full_pixel_auroc, pro
# Split one sampled meta-task into training and validation parts for
#  inner-loop learning and confidence computation.one part for inner-loop learning
#  andone part for validation/confidence This is the function that creates the
#  train/validation structure needed by meta-learning and model
#  confidence.This function takes one sampled meta-task and splits it into
# :train_task and val_task
#  it gets total task size ,computes validation count, shuffles indices, creates two smaller batches


    def _split_task_for_val(self, full_task, val_ratio=0.3):
        """
        Split a single meta-task into a training part (inner-loop) and a
        validation part used for the covariance-based confidence update.
        """
        imgs = full_task["image"]
        n = imgs.shape[0]
        n_val = max(1, int(n * val_ratio))
        perm = torch.randperm(n, device=imgs.device)
        val_idx, train_idx = perm[:n_val], perm[n_val:]

        train_batch = {"image": imgs[train_idx]}
        val_batch = {"image": imgs[val_idx]}

        for k, v in full_task.items():
            if k != "image" and v is not None:
                train_batch[k] = v[train_idx]
                val_batch[k] = v[val_idx]

        return train_batch, val_batch
# Run the full CoMet training pipeline including data confidence, meta-learning,
#  model confidence, evaluation, and checkpoint saving
# .This is the full training engine of CoMet.
    def train(self, training_data, test_data, augmented_loader=None,
              n_way=5, k_shot=1, meta_batch_size=4, meta_lr=0.01):
        """Enhanced Meta Learning with optional model/data confidence switches."""

        LOGGER.info("Starting training process...")# Start training and initialize optional data-confidence weights.
        data_weights = None

        # Only create IQR-based sample weights when data confidence is enabled.
        if self.use_data_confidence:
            # Remove sample weights when data confidence is disabled so training remains unweighted.
            LOGGER.info("Computing IQR-based data weights...")
            self.discriminator.eval()
            anomaly_scores = []

            with torch.no_grad():
                for data in training_data:
                    image = data["image"].to(self.device)
                    features, _ = self._embed(image, evaluation=True)
                    scores = -self.discriminator(features).cpu().numpy()
                    anomaly_scores.extend(scores.squeeze())

            anomaly_scores = np.array(anomaly_scores)
            q1 = np.percentile(anomaly_scores, 25)
            q3 = np.percentile(anomaly_scores, 75)
            kappa = 1.5
            threshold = q3 + kappa * (q3 - q1)

            data_weights = np.minimum(1.0, threshold / (anomaly_scores + 1e-6))
            data_weights = np.clip(data_weights, 0.0, 1.0)
            training_data.dataset.sample_weights = data_weights

            # Runtime check: confirm that data-confidence sample weights are still created.
            LOGGER.info("Data confidence is ON, sample weights were created.")
            LOGGER.info(
                f"Anomaly scores - Q1: {q1:.4f}, Q3: {q3:.4f}, "
                f"IQR: {q3-q1:.4f}, threshold: {threshold:.4f}"
            )
            LOGGER.info(
                f"Data weights - mean: {data_weights.mean():.4f}, "
                f"std: {data_weights.std():.4f}, min: {data_weights.min():.4f}, "
                f"max: {data_weights.max():.4f}"
            )
        elif hasattr(training_data.dataset, "sample_weights"):
            # Runtime check: if data confidence is off, sample weights must not be created.
            LOGGER.info("Data confidence is OFF, so sample weights are not created.")
            delattr(training_data.dataset, "sample_weights")

        self.discriminator.train()
# Initialize or reset the confidence-learning module before epoch-wise training begins.
        if self.use_confidence_learning:
            if not hasattr(self, "confidence_module") or self.confidence_module is None:
                self.confidence_module = ConfidenceLearning(
                    self.discriminator,
                    window_size=getattr(self, "confidence_window_size", 5),
                    l2_lambda=getattr(self, "l2_lambda", 0.01),
                )
            self.confidence_module.reset()

        state_dict = {}
        # Reuse an existing checkpoint if available instead of repeating full training.
        ckpt_path = os.path.join(self.output_ckpt_dir, "ckpt.pth")

        if os.path.exists(ckpt_path):
            state_dict = torch.load(ckpt_path, map_location=self.device)
            if "discriminator" in state_dict:
                self.discriminator.load_state_dict(state_dict["discriminator"])
                if self.pre_proj > 0 and "pre_projection" in state_dict:
                    self.pre_projection.load_state_dict(state_dict["pre_projection"])
                if (
                    self.use_confidence_learning
                    and self.confidence_module is not None
                    and "confidence_scores" in state_dict
                ):
                    self.confidence_module.confidence_scores = state_dict["confidence_scores"]
            else:
                self.load_state_dict(state_dict, strict=False)
            return self._evaluate_loaded_model(training_data, test_data)

        def update_state_dict(_):
            state_dict["discriminator"] = OrderedDict(
                (k, v.detach().cpu()) for k, v in self.discriminator.state_dict().items()
            )
            if self.pre_proj > 0:
                state_dict["pre_projection"] = OrderedDict(
                    (k, v.detach().cpu()) for k, v in self.pre_projection.state_dict().items()
                )
            if self.use_confidence_learning and self.confidence_module is not None:
                state_dict["confidence_scores"] = self.confidence_module.confidence_scores

        best_record = None
        current_training_data = training_data

        alpha = 1e-4
        beta = 2e-4
        # Initialize the model-confidence regularization strength
        #  based on the selected ablation setting.
        adaptive_lambda = (
            0.0
            if (not self.use_confidence_learning or not self.use_model_confidence)
            else self.fixed_lambda
        )
        LOGGER.info(f"Initial lambda: {adaptive_lambda:.6f}")
        # If meta-learning is disabled, the meta-batch loop is skipped by
        #  forcing batch size to zero.
# Disable the meta-learning loop by forcing the meta-batch size to zero when meta-learning is off.
        effective_meta_batch_size = meta_batch_size if self.use_meta_learning else 0
        epochs = self.meta_epochs
# Run epoch-wise training and collect meta-learning and confidence signals for the current epoch.
        for epoch in range(epochs):
            meta_grads = []
            meta_train_losses = []
            meta_val_losses = []
            det_sigma = 0.0
       # outer loop:the model processes multiple sampled tasks ,
        # each task contributes to the meta update
        # ,One full task is sampledinside the outer loop:a ) pick one small task
        #  from the training data.this task is used for meta-learning .
        #  The task is split into train and validation Simple meaning:
        #  one sampled task is divided into: train_task and val_task
# Outer-loop setup: sample one task, split it into train/validation parts, and create a temporary model for inner-loop adaptation.
            for _ in range(effective_meta_batch_size):# Outer meta-learning loop: process multiple sampled tasks and collect task-level update information.
                full_task = self._sample_task(current_training_data, n_way, k_shot)
                train_task, val_task = self._split_task_for_val(full_task, val_ratio=0.3)
             #  here Temporary model is created:create a temporary copy of the
             #  current discriminator this copy is used only inside the task
                temp_model = type(self.discriminator)(
                    in_planes=self.discriminator.body[0][0].in_features,
                    n_layers=len(self.discriminator.body) + 1,
                    hidden=self.discriminator.body[0][0].out_features
                    if len(self.discriminator.body) > 0 else None,
                ).to(self.device)
                temp_model.load_state_dict(self.discriminator.state_dict())
                inner_optimizer = torch.optim.Adam(temp_model.parameters(), lr=alpha) # “This line defines the optimizer that will update the temporary model during the inner loop.
             #   Inner loop starts here :adapt the temporary model for
             #  a few steps on train_task
                for inner_step in range(5):#it means temporary model will be adapted for five updated steps on sampled task.
                    inner_optimizer.zero_grad()#it means clear old gradients before computing new step
                  # inside inner loop from here compute loss on train_task  and update temporary model
            
                    step_loss = self._train_discriminator_step(  # This computes the loss of the temporary model on train if data confidence is ON -> sample weights are used model-confidence-related regularization strength is also passed
                        
                        temp_model,
                        train_task,
                        sample_weights=data_weights if self.use_data_confidence else None,
                        adaptive_lambda=adaptive_lambda,
                    )
                    if inner_step == 0: # Store the inner-loop training loss for later model-confidence computation.
                        train_loss = step_loss.item()
                    step_loss.backward()  #“This line performs backpropagation to compute how the temporary model parameters should change.”

                    inner_optimizer.step()

                temp_model.eval()
                with torch.no_grad():
                 # from here Validation loss is computed
                 #  :After inner-loop adaptation:after adapting on train_task
                 #  check performance on val_task
                 #  This is the meta-learning idea:
                 # learn on one part  and evaluate on another part
                    vloss = self._train_discriminator_step(
                        temp_model,
                        val_task,
                        sample_weights=data_weights if self.use_data_confidence else None,
                        adaptive_lambda=adaptive_lambda,
                    )
                    val_loss = vloss.item()
                temp_model.train()
                 #  here Meta-train and meta-val losses are stored:
                 #  :keep train-task loss and
                 #  keep val-task loss 
                 #  These losses are later used for model confidence.

                meta_train_losses.append(train_loss)
                meta_val_losses.append(val_loss)
                 #from here Meta-gradient is computed
                 #it mean: compare temporary model parameters with original model parameters
                 #store the difference as meta-gradient information

                grad = {
                    name: (orig_param.data - param.data)
                    for (name, param), (_, orig_param) in zip(
                        temp_model.named_parameters(),
                        self.discriminator.named_parameters(),
                    )
                }
                meta_grads.append(grad)

                del temp_model, train_task, val_task
                torch.cuda.empty_cache()
             # Compute model confidence from meta-train and meta-validation loss behavior
             #  and update adaptive regularization strength.
            if (
                self.use_confidence_learning
                and self.use_model_confidence
                and meta_train_losses
                and meta_val_losses
            ):
                avg_train = np.mean(meta_train_losses)
                avg_val = np.mean(meta_val_losses)
                 #from here Model confidence is computed from meta losses
                 #  use average meta-train and meta-val loss
                 #  compute covariance and estimate model confidence
                 # So this is point where meta-learning connects to confidence learning.

                # Update confidence-learning history using average meta-train and meta-validation loss.
                self.confidence_module.update_losses(avg_train, avg_val)
# Compute covariance of recent train and validation losses as the basis for model confidence.
                cov = self.confidence_module.compute_covariance()
                if cov is not None and cov.size > 0:
                    cov = cov + np.eye(cov.shape[0]) * 1e-8
                    try:
                        det_sigma = np.linalg.det(cov)
                        det_sigma = np.clip(det_sigma, 1e-6, 100.0)
                        det_sigma = np.log1p(det_sigma) / np.log1p(100.0)
                    except Exception:
                        det_sigma = 0.0

                lambda_0 = 0.005
                gamma = 0.5
                # Convert model uncertainty into the adaptive confidence-based regularization weight.
                adaptive_lambda = np.clip(lambda_0 * (1.0 + gamma * det_sigma), 0.0, 1.0)
                LOGGER.info(
                    f"Model uncertainty - det(Sigma): {det_sigma:.6f}, "
                    f"lambda: {adaptive_lambda:.6f}"
                )
                # Fall back to fixed or zero regularization when model confidence is not active.
            else:
                adaptive_lambda = (
                    0.0
                    if (not self.use_confidence_learning or not self.use_model_confidence)
                    else self.fixed_lambda
                )
             # Apply meta-update only when meta-learning is enabled
             # .Outer meta update happens here 
             #  use collected task-level meta-gradients
             #  update the real discriminator This is the final outer-loop update.
            if self.use_meta_learning and meta_grads:
                self._apply_meta_update(meta_grads, beta)
            self.current_adaptive_lambda = adaptive_lambda
             # from here Normal discriminator training continues
             #  After meta update:  meaning 
             #  continue the main training after meta-learning step

            self._train_discriminator(current_training_data)
             # Evaluate the current model on test data after each epoch.
            scores, segmentations, features, labels_gt, masks_gt = self.predict(test_data)
            auroc, full_pixel_auroc, pro = self._evaluate(
                test_data, scores, segmentations, features, labels_gt, masks_gt
            )
            if isinstance(pro, tuple):
                pro = pro[0]

            if self.logger:
                self.logger.logger.add_scalar("i-auroc", auroc, epoch)
                self.logger.logger.add_scalar("p-auroc", full_pixel_auroc, epoch)
                self.logger.logger.add_scalar("pro", pro, epoch)
                self.logger.logger.add_scalar("adaptive_lambda", adaptive_lambda, epoch)
                self.logger.logger.add_scalar("det_sigma", det_sigma, epoch)
# Keep the best-performing checkpoint based on image-level and pixel-level evaluation metrics.
            if best_record is None:     # “This block initializes the best performance record during the first epoch.”
                best_record = [auroc, full_pixel_auroc, pro]
                update_state_dict(None)
            else:
                if auroc > best_record[0]:# “ This block updates the best checkpoint when the current epoch achieves a better image-level AUROC than all previous epochs.”
                    best_record = [auroc, full_pixel_auroc, pro]
                    update_state_dict(None)
                elif auroc == best_record[0] and full_pixel_auroc > best_record[1]:# “If the image-level AUROC is tied, the code checks whether the pixel-level AUROC is better and updates the best record accordingly.”

                    best_record[1] = full_pixel_auroc
                    best_record[2] = pro
                    update_state_dict(None)
# During training, each epoch gives different results.This block ensures:the model does not only keep the
#  last epoch it keeps the best-performing epoch So it is a best model selection block.


            print(        #“This print statement summarizes the current epoch performance and shows both the current values and the best values obtained so far.”


                f"----- Epoch {epoch} " #“ print current epoch number

                f"I-AUROC:{round(auroc, 4)}(MAX:{round(best_record[0], 4)}) " #“ print current image AUROC best image AUROC so far

                f"P-AUROC:{round(full_pixel_auroc, 4)}(MAX:{round(best_record[1], 4)}) "#“current pixel AUROC best pixel AUROC so far


                f"PRO-AUROC:{round(pro, 4)}(MAX:{round(best_record[2], 4)}) "#“print current PRO best PRO so far


                f"lambda:{round(adaptive_lambda, 6)} -----" #“ print current lambda

            )

        torch.save(state_dict, ckpt_path)
        return best_record

# Apply the Reptile-style outer-loop meta update using averaged task-level gradients.
    def _apply_meta_update(self, meta_grads, meta_lr):
        """Apply Reptile meta-update with fixed learning rate"""
        # Initialize average gradients
        avg_grad = {
            name: torch.zeros_like(param) 
            for name, param in self.discriminator.named_parameters()
        }
        
        # Compute average of meta-gradients
        for grad in meta_grads:
            for name, param in grad.items():
                avg_grad[name] += param / len(meta_grads)
        
        # Apply the meta-update with fixed learning rate
        for name, param in self.discriminator.named_parameters():
            param.data += meta_lr * avg_grad[name]

    def _refine_dataset(self, dataloader, keep_indices):
        dataset = dataloader.dataset
        if hasattr(dataset, 'data_to_iterate'):
            dataset.data_to_iterate = [d for d, keep in zip(dataset.data_to_iterate, keep_indices) if keep]
        
        # Keep sample_weights in sync
        if hasattr(dataset, 'sample_weights'):
            dataset.sample_weights = np.array(dataset.sample_weights)[keep_indices]

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=dataloader.batch_size,
            shuffle=True,
            num_workers=dataloader.num_workers,
            pin_memory=True
        )
        
        return refined_dataloader
    
    def save_weighted_results(self, dataloader, output_dir="./weighted_results"):

        os.makedirs(output_dir, exist_ok=True)

        # “This block checks whether the dataset already has sample weights. If sample weights are available,
        #  it uses them and prints their minimum and maximum values. Otherwise, it assigns equal weight 1 to all samples.”
        if hasattr(dataloader.dataset, 'sample_weights'):
            weights = dataloader.dataset.sample_weights   # If the weights exist, load them.

            print(f"Sample weights min: {np.min(weights):.6f}, max: {np.max(weights):.6f}")# Show the smallest and largest weights.

        else:  # “If data-confidence weights are not available, this line assigns equal importance to every sample.”

            weights = np.ones(len(dataloader.dataset))

        # Predict scores and segmentations
        scores, segmentations, features, labels_gt, masks_gt = self.predict(dataloader)

        # Normalize scores
        norm_scores = (scores - np.min(scores)) / max(np.max(scores) - np.min(scores), 1e-12)

        # Get image paths and class names
        if hasattr(dataloader.dataset, 'data_to_iterate'):
            image_paths = [x[2] for x in dataloader.dataset.data_to_iterate]
            class_names = []
            for path in image_paths:
                parts = path.split('/')
                try:
                    train_idx = parts.index('train')
                    class_name = parts[train_idx - 1]
                except ValueError:
                    class_name = parts[-2] if parts[-2] != 'good' else parts[-3]
                class_names.append(class_name)
        elif hasattr(dataloader.dataset, 'classes'):
            image_paths = [f"image_{i}" for i in range(len(dataloader.dataset))]
            class_names = [dataloader.dataset.classes[label] for label in labels_gt]
        else:
            image_paths = [f"image_{i}" for i in range(len(dataloader.dataset))]
            class_names = ['unknown'] * len(image_paths)

        alpha = 0.7

        for idx, (image_path, weight, segmentation, score, norm_score, class_name) in enumerate(zip(
            image_paths, weights, segmentations, scores, norm_scores, class_names
        )):
            try:
                # Load original image
                original_image = cv2.imread(image_path)
                if original_image is None:
                    raise ValueError(f"Could not load image {image_path}")
                original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

                norm_seg = cv2.normalize(np.array(segmentation), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                colored_heatmap = cv2.applyColorMap(norm_seg, cv2.COLORMAP_JET)
                if colored_heatmap.shape[:2] != original_image.shape[:2]:
                    colored_heatmap = cv2.resize(colored_heatmap, (original_image.shape[1], original_image.shape[0]))

                overlay_colored = cv2.addWeighted(original_image, alpha, colored_heatmap, 1 - alpha, 0)

                color = (0, 255, 0) if weight > 0.8 else ((0, 255, 255) if weight > 0.5 else (0, 0, 255))

                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(overlay_colored, f"Weight: {weight:.6f}", (10, 30), font, 0.7, color, 2)

                image_name = os.path.basename(image_path) if isinstance(image_path, str) else f"image_{idx}"
                img_dir = os.path.join(output_dir, class_name, f"sample_{idx}")
                os.makedirs(img_dir, exist_ok=True)

                cv2.imwrite(os.path.join(img_dir, "original.png"), cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(img_dir, "segmentation_colored.png"), colored_heatmap)
                cv2.imwrite(os.path.join(img_dir, "overlay_colored.png"), cv2.cvtColor(overlay_colored, cv2.COLOR_RGB2BGR))

                mask_std = np.std(norm_seg)
                mask_mean = np.mean(norm_seg)
                predicted_mask = (
                    (norm_seg > (mask_mean + mask_std)).astype(np.uint8) * 255
                )
                cv2.imwrite(os.path.join(img_dir, "predicted_mask.png"), predicted_mask)

                # Save metadata
                with open(os.path.join(img_dir, "metadata.txt"), "w") as f:
                    f.write(f"Image: {image_name}\n")
                    f.write(f"Class: {class_name}\n")
                    f.write(f"Weight: {weight:.6f}\n")
                    f.write(f"Raw Score: {score:.6f}\n")
                    f.write(f"Normalized Score: {norm_score:.6f}\n")
                    f.write(f"Mask Mean: {mask_mean:.6f}\n")
                    f.write(f"Mask Std: {mask_std:.6f}\n")
                    f.write(f"Is Anomaly: {labels_gt[idx] if idx < len(labels_gt) else 'Unknown'}\n")

            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
                continue

        print(f"Saved {len(image_paths)} weighted images with colored heatmaps and masks to {output_dir}")


    def _evaluate_loaded_model(self, training_data, test_data):
        """Evaluate a loaded model with confidence metrics"""
        # Initialize confidence module if not existing
        if self.use_confidence_learning and self.confidence_module is None:
            self.confidence_module = ConfidenceLearning(
                self.discriminator, 
                window_size=self.confidence_window_size, 
                l2_lambda=self.l2_lambda
            )

        confidence = None
        if self.use_confidence_learning and self.confidence_module is not None:
            train_loss = self._compute_dataset_loss(training_data)
            val_loss = self._compute_dataset_loss(test_data)
            self.confidence_module.update_losses(train_loss, val_loss)
            confidence = self.confidence_module.compute_confidence_metric()
        
        scores, segmentations, features, labels_gt, masks_gt = self.predict(test_data)
        auroc, full_pixel_auroc, pro = self._evaluate(
            test_data, scores, segmentations, features, labels_gt, masks_gt)
        
        print(f"Loaded model evaluation: "
            f"I-AUROC:{round(auroc, 4)}, "
            f"P-AUROC:{round(full_pixel_auroc, 4)}, "
            f"PRO:{round(pro, 4)}, "
            f"Confidence:{round(confidence, 4) if confidence is not None else 'off'}")
        
        return [auroc, full_pixel_auroc, pro]
    
    def _compute_dataset_loss(self, dataloader):
        """Compute average loss on a dataset without updating model parameters"""
        self.discriminator.eval()
        total_loss = 0.0
        batch_count = 0
        
        with torch.no_grad():
            for data in dataloader:
                if isinstance(data, dict):
                    img = data["image"]
                else:
                    img = data
                    
                img = img.to(torch.float).to(self.device)
                
                if self.pre_proj > 0:
                    true_feats = self.pre_projection(self._embed(img, evaluation=True)[0])
                else:
                    true_feats, _ = self._embed(img, evaluation=True)
                    
                noise = torch.normal(0, self.noise_std, true_feats.shape).to(self.device)
                fake_feats = true_feats + noise
                
                scores = self.discriminator(torch.cat([true_feats, fake_feats]))
                true_scores = scores[:len(true_feats)]
                fake_scores = scores[len(fake_feats):]
                
                th = self.dsc_margin
                true_loss = torch.clip(-true_scores + th, min=0).mean()
                fake_loss = torch.clip(fake_scores + th, min=0).mean()
                loss = true_loss + fake_loss
                
                total_loss += loss.item()
                batch_count += 1
        
        self.discriminator.train()
        
        if batch_count > 0:
            return total_loss / batch_count
        return 0.0

# Sample one small meta-learning task from the training data, optionally using data-confidence weights.
    def _sample_task(self, dataloader, n_way, k_shot):
        """
        Sample n-way k-shot task with weighted sampling and diversity check.
        Returns: dict {"image": Tensor[B, C, H, W]}
        """
        if not hasattr(self, "task_memory"):
            self.task_memory = []

        dataset = dataloader.dataset
        N = len(dataset)
        # Apply weighted task sampling only when data confidence is active.
        if self.use_data_confidence and hasattr(dataset, "sample_weights"):  #“This line checks whether the no-data-confidence ablation is off and whether the dataset contains sample weights.”

            raw_weights = np.array(dataset.sample_weights, dtype=np.float64)#“This line reads the data-confidence sample weights into array form for later sampling.”
            if len(raw_weights) != N: #““This line verifies that the weight vector length matches the dataset size.”


                if len(raw_weights) < N:  #“If the weight array is shorter than the dataset, the missing samples are assigned default weight 1.”

                    pad = np.ones(N - len(raw_weights))
                    weights = np.concatenate([raw_weights, pad])
                else:   #“ If there are too many weights, cut them down to dataset size.

                    weights = raw_weights[:N]
            else:   #““If the weight vector already matches the dataset size, it is used as it is.”

                weights = raw_weights.copy()
        else:
            weights = np.ones(N, dtype=np.float64) # “If data confidence is disabled, the task sampling becomes unweighted by assigning equal weight to all samples.”



        # Normalize safely: remove invalid values and enforce a valid probability vector.
        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        weights = np.clip(weights, 0.0, None)
        if weights.sum() <= 0:
            weights = np.ones(N, dtype=np.float64) / N
        else:
            weights = weights / weights.sum()

        max_attempts = 3
        selected_indices = None

        for attempt in range(max_attempts):
            idx_pool = np.arange(N)
            prob_pool = weights.copy()
            chosen = []

            for _ in range(n_way):
                if len(idx_pool) < k_shot:
                    break

                # Re-normalize after every filtering step so np.random.choice
                # always receives a clean probability vector.
                prob_pool = np.nan_to_num(prob_pool, nan=0.0, posinf=0.0, neginf=0.0)
                prob_pool = np.clip(prob_pool, 0.0, None)
                if prob_pool.size == 0:
                    break
                if prob_pool.sum() <= 0:
                    prob_pool = np.ones(len(idx_pool), dtype=np.float64) / len(idx_pool)
                else:
                    prob_pool = prob_pool / prob_pool.sum()

                chosen_class = np.random.choice(
                    idx_pool,
                    size=k_shot,
                    replace=False,
                    p=prob_pool
                )
                chosen.extend(chosen_class)

                # Remove chosen indices
                mask = np.isin(idx_pool, chosen_class, invert=True)
                idx_pool = idx_pool[mask]
                prob_pool = prob_pool[mask]

                if prob_pool.size > 0 and prob_pool.sum() > 0:
                    prob_pool /= prob_pool.sum()

            else:  # All n_way classes sampled
                task_items = [dataset[i] for i in chosen]

                # Diversity check
                if self.task_memory:
                    img = task_items[0]["image"].unsqueeze(0).to(self.device)
                    feat, _ = self._embed(img, evaluation=True)
                    sims = [torch.cosine_similarity(feat, mem, dim=1).mean().item()
                            for mem in self.task_memory]
                    if max(sims) < 0.85:
                        selected_indices = chosen
                        break
                else:
                    selected_indices = chosen
                    break

        if selected_indices is None:
            n_take = min(n_way * k_shot, N)
            selected_indices = np.random.choice(N, size=n_take, replace=False).tolist()
        else:
            selected_indices = selected_indices

        task_items = [dataset[i] for i in selected_indices]

        img = task_items[0]["image"].unsqueeze(0).to(self.device)
        feat, _ = self._embed(img, evaluation=True)
        self.task_memory.append(feat)
        if len(self.task_memory) > 5:
            self.task_memory.pop(0)

        batch = {
            "image": torch.stack([item["image"] for item in task_items]).to(self.device),
        }
        return batch
# Compute one discriminator loss step, optionally using data weights and model-confidence regularization.
    def _train_discriminator_step(self, model, data, sample_weights=None, adaptive_lambda=None):
        """
        Single discriminator training step with SCL loss.
        Accepts either a batch dict or a DataLoader (iterates once).
        """
        if isinstance(data, torch.utils.data.DataLoader):
            # Take the first (and only) batch
            try:
                data_batch = next(iter(data))
            except StopIteration:
                raise ValueError("DataLoader is empty in _train_discriminator_step")
        else:
            data_batch = data  # assume it's already a batch dict

        if isinstance(data_batch, dict):
            img = data_batch["image"]
        elif isinstance(data_batch, (list, tuple)):
            img = data_batch[0]  # fallback: assume first element is image
        else:
            raise TypeError(f"Unsupported data type: {type(data_batch)}")

        img = img.to(torch.float).to(self.device)

        true_feats, _ = self._embed(img, evaluation=False)  # Fixed: unpack tuple

        noise_std_val = self.noise_std[0] if isinstance(self.noise_std, (list, tuple)) else self.noise_std
        noise = torch.normal(0, noise_std_val * 1.2, true_feats.shape).to(self.device)
        fake_feats = true_feats + noise

        scores = model(torch.cat([true_feats, fake_feats]))
        true_scores = scores[:len(true_feats)]
        fake_scores = scores[len(fake_feats):]

        true_loss = torch.clamp(1 - true_scores, min=0).mean()
        fake_loss = torch.clamp(1 + fake_scores, min=0).mean()

        base_loss = true_loss + fake_loss

        if sample_weights is not None:
            batch_size = img.shape[0]
            if len(sample_weights) >= batch_size:
                weights = sample_weights[:batch_size]
            else:
                weights = np.pad(sample_weights, (0, batch_size - len(sample_weights)), constant_values=1.0)
            weights = torch.from_numpy(weights).float().to(self.device)

            # Expand weights to match patch dimension if needed
            if true_loss.shape != weights.shape:
                weights = weights.unsqueeze(1)  # or repeat_interleave if per-image

            true_loss = (true_loss * weights).mean()
            fake_loss = (fake_loss * weights).mean()
            base_loss = true_loss + fake_loss

        if adaptive_lambda is not None and adaptive_lambda > 0:
            l2_reg = sum(p.norm(2) ** 2 for p in model.parameters())
            total_loss = base_loss + adaptive_lambda * l2_reg
        else:
            total_loss = base_loss

        return total_loss
# Run standard discriminator training over the dataset using the current confidence settings.
    def _train_discriminator(self, training_data):
        """
        Train discriminator with Soft Confident Learning.
        Uses: LSCL(?) = S wi · LAD(xi|?) + ?(S) · ||?||²
        """
        # Turn off adaptive lambda when confidence learning or model confidence is disabled.
        adaptive_lambda = getattr(self, 'current_adaptive_lambda', 0.01)
        # Runtime check: in the no-model-confidence ablation, adaptive lambda must be disabled.

        if not self.use_confidence_learning or not self.use_model_confidence:
           LOGGER.info("Model confidence is OFF, so adaptive lambda is set to 0.0")
           adaptive_lambda = 0.0          
        # Use sample weights only in data-confidence-enabled runs.
        # Get data weights
        data_weights = (
            training_data.dataset.sample_weights
            if self.use_data_confidence and hasattr(training_data.dataset, 'sample_weights')
            else None
        )
        # Runtime check: confirm whether weighted loss is active or skipped.
        if data_weights is not None:
            LOGGER.info("Weighted loss is active because data confidence is enabled.")
        else:
           LOGGER.info("Weigted loss is inactive becuase data confidence is  not enabled.")
        
        i_iter = 0
        epoch_metrics = {
            'loss_weighted': [],
            'loss_unweighted': [],
            'weight_mean': [],
            'weight_std': [],
            'confidence': []
        }
        
        
        val_loss = self._compute_dataset_loss(training_data)
        
        with tqdm.tqdm(total=self.gan_epochs) as pbar:
            for i_epoch in range(self.gan_epochs):
                all_loss = []
                all_p_true = []
                all_p_fake = []
                all_p_interp = []
                embeddings_list = []

                # Collect weight statistics only when data-confidence weights are available.
                if self.use_data_confidence and hasattr(training_data.dataset, 'sample_weights'):
                    sample_weights = torch.tensor(training_data.dataset.sample_weights, 
                                            device=self.device)
                    epoch_metrics['weight_mean'].append(sample_weights.mean().item())
                    epoch_metrics['weight_std'].append(sample_weights.std().item())
                else:
                    sample_weights = torch.ones(len(training_data.dataset), 
                                            device=self.device)

                batch_losses = []
                
                for idx, data_item in enumerate(training_data):
                    self.dsc_opt.zero_grad()
                    if self.pre_proj > 0:
                        self.proj_opt.zero_grad()

                    i_iter += 1
                    img = data_item["image"]
                    img = img.to(torch.float).to(self.device)
                    
                    # Get batch-specific weights
                    batch_start = idx * training_data.batch_size
                    batch_end = batch_start + img.shape[0]
                    
                    if data_weights is not None and batch_end <= len(data_weights):
                        batch_weights = data_weights[batch_start:batch_end]
                    else:
                        batch_weights = None
                    
                    if self.pre_proj > 0:
                        true_feats = self.pre_projection(self._embed(img, evaluation=False)[0])
                    else:
                        true_feats = self._embed(img, evaluation=False)[0]
                    
                    noise_idxs = torch.randint(0, self.mix_noise, torch.Size([true_feats.shape[0]]))
                    noise_one_hot = torch.nn.functional.one_hot(noise_idxs, 
                                                            num_classes=self.mix_noise).to(self.device)
                    noise = torch.stack([
                        torch.normal(0, self.noise_std * 1.1**(k), true_feats.shape)
                        for k in range(self.mix_noise)], dim=1).to(self.device)
                    noise = (noise * noise_one_hot.unsqueeze(-1)).sum(1)
                    fake_feats = true_feats + noise

                    scores = self.discriminator(torch.cat([true_feats, fake_feats]))
                    true_scores = scores[:len(true_feats)]
                    fake_scores = scores[len(fake_feats):]
                    
                    th = self.dsc_margin
                    p_true = (true_scores.detach() >= th).sum() / len(true_scores)
                    p_fake = (fake_scores.detach() < -th).sum() / len(fake_scores)
                    true_loss = torch.clip(-true_scores + th, min=0)
                    fake_loss = torch.clip(fake_scores + th, min=0)

                    # Apply data weights if available
                    if batch_weights is not None:
                        patches_per_image = true_loss.shape[0] // len(batch_weights)
                        batch_weights_tensor = torch.from_numpy(batch_weights).float().to(self.device)
                        batch_weights_expanded = batch_weights_tensor.repeat_interleave(patches_per_image).view(-1, 1)
                        
                        # Store unweighted loss
                        unweighted_loss = true_loss.mean() + fake_loss.mean()
                        epoch_metrics['loss_unweighted'].append(unweighted_loss.item())
                        
                        # Apply weights
                        true_loss = (true_loss * batch_weights_expanded).mean()
                        fake_loss = fake_loss.mean()
                    else:
                        true_loss = true_loss.mean()
                        fake_loss = fake_loss.mean()

                    base_loss = true_loss + fake_loss
                    
                    # Add L2 regularization with adaptive lambda: ?(S) · ||?||²
                    l2_reg = sum(torch.norm(p, 2) ** 2 for p in self.discriminator.parameters())
                    loss = base_loss + adaptive_lambda * l2_reg
                    
                    batch_losses.append(base_loss.item())
                    epoch_metrics['loss_weighted'].append(loss.item())

                    self.logger.logger.add_scalar(f"p_true", p_true, self.logger.g_iter)
                    self.logger.logger.add_scalar(f"p_fake", p_fake, self.logger.g_iter)
                    self.logger.logger.add_scalar("loss", loss, self.logger.g_iter)
                    self.logger.step()

                    loss.backward()
                    if self.pre_proj > 0:
                        self.proj_opt.step()
                    if self.train_backbone:
                        self.backbone_opt.step()
                    self.dsc_opt.step()

                    loss = loss.detach().cpu()
                    all_loss.append(loss.item())
                    all_p_true.append(p_true.cpu().item())
                    all_p_fake.append(p_fake.cpu().item())
                
                if hasattr(self, 'confidence_module') and self.confidence_module is not None:
                    avg_train_loss = sum(batch_losses) / len(batch_losses) if batch_losses else 0
                    new_val_loss = self._compute_dataset_loss(training_data)
                    self.confidence_module.update_losses(avg_train_loss, new_val_loss)
                
                if len(embeddings_list) > 0:
                    self.auto_noise[1] = torch.cat(embeddings_list).std(0).mean(-1)
                
                if self.cos_lr:
                    self.dsc_schl.step()
                
                # Calculate epoch statistics
                all_loss = sum(all_loss) / len(training_data)
                all_p_true = sum(all_p_true) / len(training_data)
                all_p_fake = sum(all_p_fake) / len(training_data)
                
                # Log epoch metrics
                if len(epoch_metrics['weight_mean']) > 0:
                    self.logger.logger.add_scalar("epoch_weight_mean", 
                                            np.mean(epoch_metrics['weight_mean']), 
                                            i_epoch)
    
                    self.logger.logger.add_scalar("epoch_weight_std", 
                                            np.mean(epoch_metrics['weight_std']), 
                                            i_epoch)
                
                cur_lr = self.dsc_opt.state_dict()['param_groups'][0]['lr']
                pbar_str = f"epoch:{i_epoch} loss:{round(all_loss, 5)} "
                pbar_str += f"lr:{round(cur_lr, 6)}"
                pbar_str += f" p_true:{round(all_p_true, 3)} p_fake:{round(all_p_fake, 3)}"
                pbar_str += f" ?:{round(adaptive_lambda, 6)}"
                if len(all_p_interp) > 0:
                    pbar_str += f" p_interp:{round(sum(all_p_interp) / len(training_data), 3)}"
                pbar.set_description_str(pbar_str)
                pbar.update(1)
# Run prediction on data and return anomaly scores, segmentations, and supporting outputs.
    def predict(self, data, prefix=""):
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader(data, prefix)
        return self._predict(data)
# Perform batch-wise inference over a dataloader and collect prediction outputs.
    def _predict_dataloader(self, dataloader, prefix):
        """This function provides anomaly scores/maps for full dataloaders."""
        _ = self.forward_modules.eval()

        img_paths = []
        scores = []
        masks = []
        features = []
        labels_gt = []
        masks_gt = []
        from sklearn.manifold import TSNE

        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
            for data in data_iterator:
                if isinstance(data, dict):
                    labels_gt.extend(data["is_anomaly"].numpy().tolist())
                    if data.get("mask", None) is not None:
                        masks_gt.extend(data["mask"].numpy().tolist())
                    image = data["image"]
                    img_paths.extend(data['image_path'])
                _scores, _masks, _feats = self._predict(image)
                for score, mask in zip(_scores, _masks):
                    scores.append(score)
                    masks.append(mask)
                if _feats is not None and len(_feats) > 0:
                    features.extend(_feats)

        return scores, masks, features, labels_gt, masks_gt
# Convert images into feature embeddings, anomaly scores, and segmentation maps.
    def _predict(self, images):
        """Infer score and mask for a batch of images."""
        images = images.to(torch.float).to(self.device)
        _ = self.forward_modules.eval()

        batchsize = images.shape[0]
        if self.pre_proj > 0:
            self.pre_projection.eval()
        self.discriminator.eval()
        with torch.no_grad():
            features, patch_shapes = self._embed(
                images, provide_patch_shapes=True, evaluation=True
            )
            if self.pre_proj > 0:
                features = self.pre_projection(features)

            patch_scores = image_scores = -self.discriminator(features)
            patch_scores = patch_scores.cpu().numpy()
            image_scores = image_scores.cpu().numpy()

            image_scores = self.patch_maker.unpatch_scores(
                image_scores, batchsize=batchsize
            )
            image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
            image_scores = self.patch_maker.score(image_scores)

            patch_scores = self.patch_maker.unpatch_scores(
                patch_scores, batchsize=batchsize
            )
            scales = patch_shapes[0]
            patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])
            features = features.reshape(batchsize, scales[0], scales[1], -1)
            masks, features = self.anomaly_segmentor.convert_to_segmentation(
                patch_scores, None
            )

            if not self._debug_prediction_stats_logged:
                LOGGER.info(
                    f"Prediction debug - image_scores: min={image_scores.min():.6f}, "
                    f"max={image_scores.max():.6f}, mean={image_scores.mean():.6f}"
                )
                LOGGER.info(
                    f"Prediction debug - patch_scores shape={patch_scores.shape}, "
                    f"min={patch_scores.min():.6f}, max={patch_scores.max():.6f}"
                )
                LOGGER.info(
                    f"Prediction debug - masks shape={np.array(masks).shape}, "
                    f"min={np.min(masks):.6f}, max={np.max(masks):.6f}"
                )
                self._debug_prediction_stats_logged = True

            # Optionally refine the predicted anomaly masks to reduce diffuse localization.
            if self.use_mask_refinement:
                masks = self._refine_segmentation_masks(masks)

        return list(image_scores), list(masks), list(features)

    # Refine predicted segmentation masks by removing weak and noisy regions.
    def _refine_segmentation_masks(self, masks):
        refined_masks = []

        for mask in masks:
            mask = np.array(mask, dtype=np.float32)

            # Normalize the anomaly map before thresholding.
            min_val = mask.min()
            max_val = mask.max()
            if max_val > min_val:
                norm_mask = (mask - min_val) / (max_val - min_val)
            else:
                norm_mask = mask.copy()

            # Keep only sufficiently strong anomaly responses.
            binary = (norm_mask >= self.mask_threshold).astype(np.uint8)

            # Remove tiny disconnected regions that are likely noise.
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                binary, connectivity=8
            )

            cleaned = np.zeros_like(binary)
            for label_id in range(1, num_labels):
                area = stats[label_id, cv2.CC_STAT_AREA]
                if area >= self.min_component_area:
                    cleaned[labels == label_id] = 1

            refined = norm_mask * cleaned
            refined_masks.append(refined)

        return refined_masks

    @staticmethod
    def _params_file(filepath, prepend=""):
        return os.path.join(filepath, prepend + "params.pkl")

    def save_to_path(self, save_path: str, prepend: str = ""):
        LOGGER.info("Saving data.")
        self.anomaly_scorer.save(
            save_path, save_features_separately=False, prepend=prepend
        )
        params = {
            "backbone.name": self.backbone.name,
            "layers_to_extract_from": self.layers_to_extract_from,
            "input_shape": self.input_shape,
            "pretrain_embed_dimension": self.forward_modules[
                "preprocessing"
            ].output_dim,
            "target_embed_dimension": self.forward_modules[
                "preadapt_aggregator"
            ].target_dim,
            "patchsize": self.patch_maker.patchsize,
            "patchstride": self.patch_maker.stride,
            "anomaly_scorer_num_nn": self.anomaly_scorer.n_nearest_neighbours,
        }
        with open(self._params_file(save_path, prepend), "wb") as save_file:
            pickle.dump(params, save_file, pickle.HIGHEST_PROTOCOL)

    # Save segmentation visualizations for qualitative anomaly-detection analysis.
    def save_segmentation_images(self, data, segmentations, scores):
        image_paths = [
            x[2] for x in data.dataset.data_to_iterate
        ]
        mask_paths = [
            x[3] for x in data.dataset.data_to_iterate
        ]

        def image_transform(image):
            in_std = np.array(
                data.dataset.transform_std
            ).reshape(-1, 1, 1)
            in_mean = np.array(
                data.dataset.transform_mean
            ).reshape(-1, 1, 1)
            image = data.dataset.transform_img(image)
            return np.clip(
                (image.numpy() * in_std + in_mean) * 255, 0, 255
            ).astype(np.uint8)

        def mask_transform(mask):
            return data.dataset.transform_mask(mask).numpy()

        save_dir = os.path.join(self.output_ckpt_dir, "segmentation_images")
        plot_segmentation_images(
            save_dir,
            image_paths,
            segmentations,
            scores,
            mask_paths,
            image_transform=image_transform,
            mask_transform=mask_transform,
        )

class PatchMaker:
    def __init__(self, patchsize, top_k=0, stride=None):
        self.patchsize = patchsize
        self.stride = stride
        self.top_k = top_k

    def patchify(self, features, return_spatial_info=False):
        """Convert a tensor into a tensor of respective patches.
        Args:
            x: [torch.Tensor, bs x c x w x h]
        Returns:
            x: [torch.Tensor, bs * w//stride * h//stride, c, patchsize,
            patchsize]
        """
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (
                s + 2 * padding - 1 * (self.patchsize - 1) - 1
            ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(self, x, batchsize):
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x):
        was_numpy = False
        if isinstance(x, np.ndarray):
            was_numpy = True
            x = torch.from_numpy(x)
        while x.ndim > 2:
            x = torch.max(x, dim=-1).values
        if x.ndim == 2:
            if self.top_k > 1:
                x = torch.topk(x, self.top_k, dim=1).values.mean(1)
            else:
                x = torch.max(x, dim=1).values
        if was_numpy:
            return x.numpy()
        return x

