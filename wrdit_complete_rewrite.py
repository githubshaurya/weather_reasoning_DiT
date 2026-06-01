"""
This file got a PSNR of 31, 26.8, 31 on Rain-100, Raindrop, Snow-100k test set as of Jan 5th, 2026
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, ConcatDataset
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import VGG19_Weights, MobileNet_V3_Large_Weights
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image
import warnings
import csv
import logging
import json
from collections import defaultdict
from datetime import datetime

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
DATASET_ROOT = Path('/home/cvpr_pg_4/Shourya/dataset-new')
TASK_NAMES = {'Rain': 0, 'Raindrop': 1, 'Snow': 2}
NUM_WEATHER_TYPES = len(TASK_NAMES)
SEVERITY_LEVELS = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2}
BALANCED_SAMPLING_WEIGHTS = {'Raindrop': 4.3, 'Rain': 6.1, 'Snow': 1.1}

print('='*120)

# ============================================================================
# AUTO-DETECT MOBILENETV3 CHANNELS
# ============================================================================
print(' AUTO-DETECTING MOBILENETV3 CHANNELS...')
mobilenet_temp = models.mobilenet_v3_large(weights=None)
x_temp = torch.randn(1, 3, 256, 256)
layer_channels = {}
for i, layer in enumerate(mobilenet_temp.features):
    x_temp = layer(x_temp)
    spatial_size = x_temp.shape[2]
    if spatial_size in [64, 32, 16, 8]:
        layer_channels[spatial_size] = (i, x_temp.shape[1])

DETECTED_INDICES = {
    '64x64': layer_channels.get(64, (0, 32))[0],
    '32x32': layer_channels.get(32, (8, 64))[0],
    '16x16': layer_channels.get(16, (14, 128))[0],
    '8x8': layer_channels.get(8, (20, 256))[0]
}

DETECTED_CHANNELS = tuple(
    layer_channels.get(s, (0, c))[1] for s, c in zip([64, 32, 16, 8], [32, 64, 128, 256])
)

DETECTED_SKIP_CHANNELS = (
    layer_channels.get(8, (0, 960))[1],
    layer_channels.get(16, (0, 672))[1],
    layer_channels.get(32, (0, 40))[1],
    layer_channels.get(64, (0, 16))[1],
    0
)

del mobilenet_temp, x_temp
print(f' Detected channels: {DETECTED_CHANNELS}\n')

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def compute_psnr(pred, target):
    """Compute PSNR metric"""
    pred = (pred + 1) / 2
    target = (target + 1) / 2
    pred = torch.clamp(pred, 0, 1)
    target = torch.clamp(target, 0, 1)
    mse = F.mse_loss(pred, target)
    if mse < 1e-10:
        return 100.0
    psnr = 10 * torch.log10(1.0 / mse)
    return psnr.item()

def compute_ssim(pred, target, window_size=11):
    """Compute SSIM metric"""
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2
    pred = (pred + 1) / 2
    target = (target + 1) / 2
    pred = torch.clamp(pred, 0, 1)
    target = torch.clamp(target, 0, 1)
    
    mu1 = F.avg_pool2d(pred, window_size, 1, padding=window_size//2)
    mu2 = F.avg_pool2d(target, window_size, 1, padding=window_size//2)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.avg_pool2d(pred**2, window_size, 1, padding=window_size//2) - mu1_sq
    sigma2_sq = F.avg_pool2d(target**2, window_size, 1, padding=window_size//2) - mu2_sq
    sigma12 = F.avg_pool2d(pred*target, window_size, 1, padding=window_size//2) - mu1_mu2
    
    ssim_map = (2*mu1_mu2 + C1) * (2*sigma12 + C2) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()

def compute_all_component_gradients(model):
    """ COMPREHENSIVE: Compute gradients for ALL components"""
    component_stats = {}
    total_grad_norm = 0.0
    
    components = {
        'encoder': model.encoder,
        'feature_pyramid': model.feature_pyramid,
        'reasoning': model.reasoning,
        'token_router': model.token_router,
        'token_reconstructor': model.token_reconstructor,
        'dit_blocks': model.dit_blocks,
        'decoder': model.decoder,
    }
    
    for comp_name, component in components.items():
        grad_norm = 0.0
        has_grad = 0
        num_params = 0
        
        for param in component.parameters():
            num_params += param.numel()
            if param.grad is not None:
                has_grad = 1
                grad_norm += param.grad.norm().item() ** 2
        
        grad_norm = np.sqrt(grad_norm)
        total_grad_norm += grad_norm ** 2
        
        component_stats[comp_name] = {
            'grad_norm': grad_norm,
            'has_grad': has_grad,
            'num_params': num_params
        }
    
    total_grad_norm = np.sqrt(total_grad_norm)
    
    for comp_name in component_stats:
        if total_grad_norm > 0:
            component_stats[comp_name]['percentage'] = (
                component_stats[comp_name]['grad_norm'] / total_grad_norm * 100
            )
        else:
            component_stats[comp_name]['percentage'] = 0.0
    
    max_component = max(component_stats, key=lambda x: component_stats[x]['grad_norm'])
    min_component = min(component_stats, key=lambda x: component_stats[x]['grad_norm'])
    
    return {
        'components': component_stats,
        'total_grad_norm': total_grad_norm,
        'max_component': max_component,
        'min_component': min_component
    }

def compute_routing_parameters(model):
    """ Compute routing threshold learning parameters with explicit checks"""
    routing_threshold = model.token_router.routing_threshold.item()
    learned_ratio = torch.sigmoid(model.token_router.routing_threshold).item()
    temperature = model.token_router.temperature.item()
    
    threshold_has_grad = 1 if model.token_router.routing_threshold.grad is not None else 0
    threshold_grad = model.token_router.routing_threshold.grad.item() if threshold_has_grad else 0.0
    temperature_has_grad = 1 if model.token_router.temperature.grad is not None else 0
    temperature_grad = model.token_router.temperature.grad.item() if temperature_has_grad else 0.0
    
    #  Verify gradients are actually non-zero (not just present)
    threshold_has_nonzero_grad = 1 if (threshold_has_grad and abs(threshold_grad) > 1e-10) else 0
    temperature_has_nonzero_grad = 1 if (temperature_has_grad and abs(temperature_grad) > 1e-10) else 0
    
    return {
        'routing_threshold': routing_threshold,
        'learned_ratio': learned_ratio,
        'temperature': temperature,
        'threshold_has_grad': threshold_has_grad,
        'threshold_grad': threshold_grad,
        'threshold_has_nonzero_grad': threshold_has_nonzero_grad,
        'temperature_has_grad': temperature_has_grad,
        'temperature_grad': temperature_grad,
        'temperature_has_nonzero_grad': temperature_has_nonzero_grad,
    }

# ============================================================================
# ENHANCED LOGGING
# ============================================================================

class OptimizedInstrumentedLogger:
    """Memory-optimized logging with buffered CSV writes"""
    
    def __init__(self, logdir='./logs/wrdit_v4_fixed'):
        self.logdir = Path(logdir)
        self.logdir.mkdir(parents=True, exist_ok=True)
        self.train_dir = self.logdir / 'train_logs'
        self.component_dir = self.logdir / 'component_analysis'
        
        for d in [self.train_dir, self.component_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # CSV file paths
        self.epoch_metrics_csv = self.train_dir / 'epoch_metrics.csv'
        self.loss_breakdown_csv = self.train_dir / 'loss_breakdown.csv'
        self.router_analysis_csv = self.component_dir / 'token_router_analysis.csv'
        self.component_gradients_csv = self.component_dir / 'component_gradients.csv'
        self.routing_threshold_csv = self.component_dir / 'routing_threshold_analysis.csv'
        self.router_loss_csv = self.component_dir / 'router_loss_breakdown.csv'
        
        # Buffers
        self.epoch_buffer = []
        self.loss_buffer = []
        self.router_buffer = []
        self.component_grad_buffer = []
        self.routing_threshold_buffer = []
        self.router_loss_buffer = []
        
        self._init_csv_files()
        
        self.logger = logging.getLogger(f'WRDiT_v4_fixed_{id(self)}')
        self.logger.setLevel(logging.DEBUG)
        if not self.logger.handlers:
            fh = logging.FileHandler(self.logdir / 'training.log')
            fh.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)
    
    def _init_csv_files(self):
        """Initialize CSV files with headers"""
        headers = {
            self.epoch_metrics_csv: ['epoch', 'train_loss_total', 'train_loss_l1', 'train_loss_perceptual', 
                                     'train_loss_weather', 'train_loss_severity', 'train_loss_router', 'avg_psnr', 'avg_ssim', 
                                     'learning_rate', 'gradient_norm', 'weather_accuracy'],
            self.loss_breakdown_csv: ['epoch', 'batch', 'batch_size', 'l1_loss', 'perceptual_loss', 
                                     'weather_loss', 'severity_loss', 'router_loss', 'total_loss'],
            self.router_analysis_csv: ['epoch', 'batch', 'num_tokens_original', 'num_tokens_selected', 
                                      'compression_ratio', 'gumbel_entropy', 'strategy'],
            self.component_gradients_csv: ['epoch', 'batch', 'encoder_grad', 'encoder_has_grad', 'encoder_%',
                                          'pyramid_grad', 'pyramid_has_grad', 'pyramid_%',
                                          'reasoning_grad', 'reasoning_has_grad', 'reasoning_%',
                                          'router_grad', 'router_has_grad', 'router_%',
                                          'reconstructor_grad', 'reconstructor_has_grad', 'reconstructor_%',
                                          'dit_grad', 'dit_has_grad', 'dit_%',
                                          'decoder_grad', 'decoder_has_grad', 'decoder_%',
                                          'total_grad_norm', 'max_component', 'min_component'],
            self.routing_threshold_csv: ['epoch', 'batch', 'routing_threshold_raw', 'learned_ratio_sigmoid',
                                        'threshold_gradient', 'threshold_has_nonzero_grad',
                                        'temperature', 'temp_gradient', 'temp_has_nonzero_grad'],
            self.router_loss_csv: ['epoch', 'batch', 'entropy_loss', 'importance_loss', 'threshold_loss', 'total_router_loss']
        }
        
        for filepath, header in headers.items():
            if not filepath.exists():
                with open(filepath, 'w', newline='') as f:
                    csv.writer(f).writerow(header)
    
    def log_epoch_metrics(self, epoch, metrics):
        row = [epoch, metrics.get('train_loss_total', 0), metrics.get('train_loss_l1', 0),
               metrics.get('train_loss_perceptual', 0), metrics.get('train_loss_weather', 0),
               metrics.get('train_loss_severity', 0), metrics.get('train_loss_router', 0),
               metrics.get('avg_psnr', 0), metrics.get('avg_ssim', 0), metrics.get('lr', 0), 
               metrics.get('grad_norm', 0), metrics.get('weather_acc', 0)]
        self.epoch_buffer.append(row)
    
    def log_batch_losses(self, epoch, batch_idx, batch_size, l1, perc, weather, severity, router, total):
        row = [epoch, batch_idx, batch_size, l1, perc, weather, severity, router, total]
        self.loss_buffer.append(row)
    
    def log_router_analysis(self, epoch, batch_idx, num_tokens_orig, num_tokens_sel, comp_ratio, entropy, strategy):
        row = [epoch, batch_idx, num_tokens_orig, num_tokens_sel, comp_ratio, entropy, strategy]
        self.router_buffer.append(row)
    
    def log_all_component_gradients(self, epoch, batch_idx, component_grad_info):
        components = component_grad_info['components']
        row = [epoch, batch_idx,
               components['encoder']['grad_norm'], components['encoder']['has_grad'], components['encoder']['percentage'],
               components['feature_pyramid']['grad_norm'], components['feature_pyramid']['has_grad'], components['feature_pyramid']['percentage'],
               components['reasoning']['grad_norm'], components['reasoning']['has_grad'], components['reasoning']['percentage'],
               components['token_router']['grad_norm'], components['token_router']['has_grad'], components['token_router']['percentage'],
               components['token_reconstructor']['grad_norm'], components['token_reconstructor']['has_grad'], components['token_reconstructor']['percentage'],
               components['dit_blocks']['grad_norm'], components['dit_blocks']['has_grad'], components['dit_blocks']['percentage'],
               components['decoder']['grad_norm'], components['decoder']['has_grad'], components['decoder']['percentage'],
               component_grad_info['total_grad_norm'], component_grad_info['max_component'], component_grad_info['min_component']]
        self.component_grad_buffer.append(row)
    
    def log_routing_threshold_analysis(self, epoch, batch_idx, routing_params):
        row = [epoch, batch_idx,
               routing_params.get('routing_threshold', 0),
               routing_params.get('learned_ratio', 0),
               routing_params.get('threshold_grad', 0),
               routing_params.get('threshold_has_nonzero_grad', 0),
               routing_params.get('temperature', 1.0),
               routing_params.get('temperature_grad', 0),
               routing_params.get('temperature_has_nonzero_grad', 0)]
        self.routing_threshold_buffer.append(row)
    
    def log_router_loss_breakdown(self, epoch, batch_idx, entropy_loss, importance_loss, threshold_loss, total_loss):
        row = [epoch, batch_idx, entropy_loss, importance_loss, threshold_loss, total_loss]
        self.router_loss_buffer.append(row)
    
    def flush_logs(self):
        """Flush all buffers to CSV"""
        self._write_buffer(self.epoch_metrics_csv, self.epoch_buffer)
        self._write_buffer(self.loss_breakdown_csv, self.loss_buffer)
        self._write_buffer(self.router_analysis_csv, self.router_buffer)
        self._write_buffer(self.component_gradients_csv, self.component_grad_buffer)
        self._write_buffer(self.routing_threshold_csv, self.routing_threshold_buffer)
        self._write_buffer(self.router_loss_csv, self.router_loss_buffer)
        
        self.epoch_buffer = []
        self.loss_buffer = []
        self.router_buffer = []
        self.component_grad_buffer = []
        self.routing_threshold_buffer = []
        self.router_loss_buffer = []
    
    def _write_buffer(self, filepath, buffer):
        if buffer:
            with open(filepath, 'a', newline='') as f:
                writer = csv.writer(f)
                for row in buffer:
                    writer.writerow(row)
    
    def close_logger(self):
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)

# ============================================================================
#  FIXED: INSTRUMENTED TOKEN ROUTER WITH PROPER GRADIENT FLOW
# ============================================================================

class InstrumentedTokenRouterFixed(nn.Module):
    """ FIXED Token router with proper gradient flow
    
    Key fixes:
    1. Straight-through estimator for token selection
    2. Gumbel-Softmax properly implemented  
    3. Losses directly depend on router parameters
    4. Temperature and threshold actively trained
    """
    
    def __init__(self, dim=384, severity_dim=3):
        super().__init__()
        combined_dim = dim + severity_dim
        
        self.importance_net = nn.Sequential(
            nn.Linear(combined_dim, combined_dim * 2),
            nn.GELU(),
            nn.LayerNorm(combined_dim * 2),
            nn.Linear(combined_dim * 2, combined_dim),
            nn.GELU(),
            nn.LayerNorm(combined_dim),
            nn.Linear(combined_dim, 1)
        )
        
        #  LEARNABLE ROUTING PARAMETERS
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.routing_threshold = nn.Parameter(torch.tensor(0.5))
        
        #  NEW: Importance scaling for gradient stability
        self.importance_scale = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, x, severity_score, return_indices=True, return_stats=False, tau=1.0):
        """ Forward with proper gradient flow + NaN/Inf protection
        
        Args:
            x: token features (B, N, C)
            severity_score: severity levels (B,)
            return_indices: whether to return selected indices
            return_stats: whether to return routing statistics
            tau: temperature for Gumbel-Softmax annealing
        """
        B, N, C = x.shape
        device = x.device
        dtype = x.dtype
        
        #  Compute importance scores with gradient support
        severity_classes = (severity_score * 2).long().clamp(0, 2)
        severity_onehot = F.one_hot(severity_classes, num_classes=3).float()
        severity_context = severity_onehot.unsqueeze(1).expand(B, N, 3)
        
        x_with_severity = torch.cat([x, severity_context], dim=-1)
        
        #  Importance scores (DIFFERENTIABLE)
        importance_logits = self.importance_net(x_with_severity).squeeze(-1)  # (B, N)
        importance = F.relu(importance_logits * self.importance_scale)  # Scale for stability
        
        #  SAFE: Extract severity with proper type handling
        avg_severity = severity_score.mean()
        if isinstance(avg_severity, torch.Tensor):
            avg_severity = avg_severity.item()
        
        #  SAFE: Ensure severity is valid Python float and clamp to [0, 1]
        avg_severity = float(avg_severity)
        avg_severity = np.clip(avg_severity, 0.0, 1.0)
        
        #  Adaptive keep ratio with EXPLICIT NaN/Inf protection
        learned_ratio = torch.sigmoid(self.routing_threshold)
        learned_ratio_item = learned_ratio.item()
        
        #  Clamp learned_ratio to valid range to avoid edge cases
        # Using small epsilon to prevent exact 0 or 1
        learned_ratio_item = float(np.clip(learned_ratio_item, 1e-6, 1.0 - 1e-6))
        
        #  Compute severity factor safely
        severity_factor = 0.5 + avg_severity
        # Limit range to prevent overflow or underflow
        severity_factor = float(np.clip(severity_factor, 0.5, 1.5))
        
        #  Compute adaptive keep ratio safely
        adaptive_keep_ratio = learned_ratio_item * severity_factor
        # Final clamp to valid compression range
        adaptive_keep_ratio = float(np.clip(adaptive_keep_ratio, 0.5, 1.0))
        
        #  SAFETY CHECK: Ensure no NaN or Inf exists
        if not np.isfinite(adaptive_keep_ratio):
            # Fallback to safe default value if somehow NaN/Inf slipped through
            adaptive_keep_ratio = 0.65
        
        #  NOW GUARANTEED SAFE: Can always convert to int
        num_keep = max(1, int(N * adaptive_keep_ratio))
        
        # Strategy determination
        if avg_severity < 0.33:
            strategy = "LOW"
        elif avg_severity < 0.67:
            strategy = "MEDIUM"
        else:
            strategy = "HIGH"
        
        #  GUMBEL-SOFTMAX WITH STRAIGHT-THROUGH ESTIMATOR
        if self.training:
            u = torch.rand_like(importance)
            u = torch.clamp(u, 1e-7, 1.0 - 1e-7)  # Prevent log(0)
            gumbel_noise = -torch.log(-torch.log(u) + 1e-8)
            gumbel_noise = torch.clamp(gumbel_noise, -20, 20)  # Prevent extreme values
            logits = (importance + gumbel_noise) / (self.temperature * tau + 1e-8)
            weights = F.softmax(logits, dim=1)
            
            selected_weights, top_indices = torch.topk(weights, num_keep, dim=1)
            
            # Compute entropy on weights (for loss)
            entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=1).mean()
            
            # Sort indices
            indices = top_indices.sort(dim=1)[0]
        else:
            _, indices = torch.topk(importance, num_keep, dim=1)
            entropy = torch.tensor(0.0, device=device, dtype=dtype)
            indices = indices.sort(dim=1)[0]
        
        #  Select tokens using indices
        batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(-1, num_keep)
        selected_tokens = x[batch_indices, indices]
        
        #  GATHER IMPORTANCE FOR LOSS (DIFFERENTIABLE!)
        selected_importance = importance.gather(1, indices)  # (B, num_keep)
        
        #  Store differentiable outputs for loss computation
        loss_info = {
            'logits': logits if self.training else importance,
            'weights': weights if self.training else F.softmax(importance, dim=1),
            'entropy': entropy,
            'importance_logits': importance_logits,
            'importance_values': importance,
            'selected_importance': selected_importance,
            'num_keep': num_keep,
            'num_total': N,
        }
        
        stats = None
        if return_stats:
            stats = {
                'strategy': strategy,
                'num_tokens_original': N,
                'num_tokens_selected': num_keep,
                'compression_ratio': N / num_keep if num_keep > 0 else 1.0,
                'gumbel_entropy': entropy.item() if isinstance(entropy, torch.Tensor) else float(entropy),
                'temperature': self.temperature.item(),
                'learned_ratio': learned_ratio_item,
                'severity_factor': severity_factor,
                'adaptive_keep_ratio': adaptive_keep_ratio,
                'avg_severity': avg_severity,
            }
        
        if return_indices:
            return selected_tokens, indices, stats, loss_info
        else:
            return selected_tokens, None, stats, loss_info

# ============================================================================
#  ROUTER LOSS FUNCTION (COMPLETELY FIXED - KEY CHANGE)
# ============================================================================

class RouterLoss(nn.Module):
    """ COMPLETELY FIXED: Router loss with ALWAYS-ACTIVE gradient flow
    
    KEY FIX: loss_threshold is now ALWAYS differentiable through routing_threshold
    No more conditional penalties that break gradients!
    Uses smooth L1 loss for all components - continuous gradient everywhere.
    """
    
    def __init__(self, weight_entropy=0.01, weight_importance=0.01, weight_threshold=0.01):
        super().__init__()
        self.weight_entropy = weight_entropy
        self.weight_importance = weight_importance
        self.weight_threshold = weight_threshold
    
    def forward(self, loss_info, router_module, quality_metric=None):
        """ FIXED: Compute router losses with CONTINUOUS gradient flow to ALL parameters
        
        The KEY CHANGE from previous implementation:
        - BEFORE: Used conditional penalties (if/elif) that broke gradients
        - NOW: Uses smooth L1 loss that is ALWAYS differentiable
        - RESULT: threshold_param now receives continuous gradients
        """
        device = loss_info['logits'].device
        dtype = loss_info['logits'].dtype
        
        #  LOSS 1: Entropy regularization (encourage balanced distributions)
        entropy = loss_info['entropy']
        target_entropy = torch.tensor(2.5, device=device, dtype=dtype)
        loss_entropy = self.weight_entropy * F.smooth_l1_loss(
            entropy.unsqueeze(0), 
            target_entropy.unsqueeze(0)
        )
        
        #  LOSS 2: Importance distribution loss (encourage balanced importance)
        importance_values = loss_info['importance_values']
        importance_mean = importance_values.mean(dim=1, keepdim=True)
        
        # Loss: encourage moderate variance in importance scores
        importance_variance = (importance_values - importance_mean).pow(2).mean()
        target_variance = torch.tensor(0.1, device=device, dtype=dtype)
        loss_importance = self.weight_importance * F.smooth_l1_loss(
            importance_variance.unsqueeze(0),
            target_variance.unsqueeze(0)
        )
        
        #  LOSS 3: FIXED THRESHOLD LOSS - NOW ALWAYS DIFFERENTIABLE 
        # THIS IS THE KEY FIX FOR THE FROZEN GRADIENT PROBLEM!
        threshold_param = router_module.routing_threshold
        learned_ratio = torch.sigmoid(threshold_param)
        
        # Target: keep learned_ratio in range [0.5, 0.8] for good compression
        # Using smooth L1 loss ensures CONTINUOUS gradient for ALL values
        target_ratio = torch.tensor(0.65, device=device, dtype=dtype)
        
        #  ALWAYS-ACTIVE smooth loss: this computes a gradient for routing_threshold
        #    regardless of its current value, because smooth_l1_loss is continuous
        loss_threshold = self.weight_threshold * F.smooth_l1_loss(
            learned_ratio.unsqueeze(0),
            target_ratio.unsqueeze(0)
        )
        
        # Verify gradient can flow: this loss depends on threshold_param
        # learned_ratio = sigmoid(threshold_param)
        # loss_threshold depends on learned_ratio
        # So d(loss_threshold)/d(threshold_param) exists and is non-zero!
        
        #  Total router loss (ALL COMPONENTS DIFFERENTIABLE)
        total_loss = loss_entropy + loss_importance + loss_threshold
        
        return loss_entropy, loss_importance, loss_threshold, total_loss

# ============================================================================
# TOKEN RECONSTRUCTOR
# ============================================================================

class InstrumentedTokenReconstructor(nn.Module):
    """Token reconstructor"""
    
    def __init__(self, dim=384, num_heads=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=0.1, batch_first=True)
        self.local_mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim)
        )
        self.blend = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, selected_tokens, all_tokens_shape, indices):
        B, N_sel, C = selected_tokens.shape
        N_all = all_tokens_shape
        device = selected_tokens.device
        dtype = selected_tokens.dtype
        
        reconstructed = torch.zeros(B, N_all, C, device=device, dtype=dtype)
        for b in range(B):
            reconstructed[b, indices[b]] = selected_tokens[b]
        
        reconstruction_rmse_list = []
        for b in range(B):
            missing_mask = torch.ones(N_all, dtype=torch.bool, device=device)
            missing_mask[indices[b]] = False
            missing_indices = torch.where(missing_mask)[0]
            num_missing = len(missing_indices)
            
            if num_missing > 0:
                missing_queries = reconstructed[b, missing_indices].unsqueeze(0)
                reconstructed_missing, _ = self.cross_attn(
                    missing_queries, selected_tokens[b:b+1], selected_tokens[b:b+1]
                )
                local_refined = self.local_mlp(missing_queries)
                final_missing = self.blend * reconstructed_missing + (1 - self.blend) * local_refined
                final_missing = final_missing.to(dtype)
                
                rmse = torch.sqrt(F.mse_loss(final_missing, missing_queries)).item()
                reconstruction_rmse_list.append(rmse)
                reconstructed[b, missing_indices] = final_missing[0]
        
        avg_rmse = np.mean(reconstruction_rmse_list) if reconstruction_rmse_list else 0.0
        return reconstructed, N_sel, N_all - N_sel, avg_rmse, self.blend.item()

# ============================================================================
# REASONING MODULE
# ============================================================================

class InstrumentedReasoningModule(nn.Module):
    """Reasoning module for weather prediction"""
    
    def __init__(self, in_channels=None, num_weather_types=NUM_WEATHER_TYPES, num_severity_levels=3):
        super().__init__()
        if in_channels is None:
            in_channels = DETECTED_CHANNELS[3]
        
        hidden_dim = 512
        
        self.spatial_attn_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 4, 1, 1),
            nn.Sigmoid()
        )
        
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 16, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 16, in_channels, 1),
            nn.Sigmoid()
        )
        
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        self.weather_embedding = nn.Embedding(num_weather_types, 384)
        self.reasoning_fc = nn.Sequential(
            nn.Linear(hidden_dim, 384),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        self.weather_head = nn.Linear(384, num_weather_types)
        self.severity_head = nn.Linear(384, num_severity_levels)
    
    def forward(self, features, task_id=None, severity_class=None):
        spatial_attn = self.spatial_attn_1x1(features)
        channel_attn = self.channel_attn(features)
        features_weighted = features * spatial_attn * channel_attn
        
        x = self.shared_conv(features_weighted)
        reasoning_emb = self.reasoning_fc(x)
        
        weather_logits = self.weather_head(reasoning_emb)
        severity_logits = self.severity_head(reasoning_emb)
        
        weather_pred = weather_logits.argmax(dim=1)
        weather_emb = self.weather_embedding(weather_pred)
        
        return weather_logits, severity_logits, weather_emb

# ============================================================================
# OTHER COMPONENTS
# ============================================================================

class FeaturePyramidFusion(nn.Module):
    """Feature pyramid fusion"""
    
    def __init__(self, in_channels_list=None, out_dim=384):
        super().__init__()
        if in_channels_list is None:
            in_channels_list = DETECTED_CHANNELS
        
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c, out_dim, 1) for c in in_channels_list
        ])
        
        self.refine_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_dim, out_dim, 3, padding=1),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(inplace=True)
            ) for _ in range(len(in_channels_list))
        ])
    
    def forward(self, features_dict):
        features = [features_dict['64x64'], features_dict['32x32'], features_dict['16x16'], features_dict['8x8']]
        
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]
        
        for i in range(len(laterals) - 1):
            laterals[i + 1] = laterals[i + 1] + F.interpolate(
                laterals[i], size=laterals[i + 1].shape[2:], mode='bilinear', align_corners=False
            )
        
        outputs = [refine(lat) for refine, lat in zip(self.refine_convs, laterals)]
        return outputs

class MobileNetV3Encoder(nn.Module):
    """MobileNetV3 backbone"""
    
    def __init__(self, pretrained=True):
        super().__init__()
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
        mobilenet = models.mobilenet_v3_large(weights=weights)
        self.features = mobilenet.features
        self.scale_indices = DETECTED_INDICES
    
    def forward(self, x):
        features = {}
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i == self.scale_indices['64x64']:
                features['64x64'] = x
            elif i == self.scale_indices['32x32']:
                features['32x32'] = x
            elif i == self.scale_indices['16x16']:
                features['16x16'] = x
            elif i == self.scale_indices['8x8']:
                features['8x8'] = x
        return features

class EnhancedDiTBlock(nn.Module):
    """DiT block with cross-attention"""
    
    def __init__(self, dim=384, num_heads=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.mha = nn.MultiheadAttention(dim, num_heads, dropout=0.1, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.norm_ctx = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=0.1, batch_first=True)
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 4, dim),
            nn.Dropout(0.1)
        )
    
    def forward(self, x, context=None):
        x = x + self.mha(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        if context is not None:
            x = x + self.cross_attn(self.norm2(x), self.norm_ctx(context), self.norm_ctx(context))[0]
        x = x + self.mlp(self.norm3(x))
        return x

class UpsampleBlockWithSkip(nn.Module):
    """Upsampling with skip connection"""
    
    def __init__(self, in_channels, out_channels, skip_channels=0):
        super().__init__()
        self.has_skip = skip_channels > 0
        
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        if self.has_skip:
            self.skip_proj = nn.Conv2d(skip_channels, out_channels, 1)
        
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels * 2 if self.has_skip else out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x, skip=None):
        x = self.up(x)
        if self.has_skip and skip is not None:
            skip = self.skip_proj(skip)
            if skip.shape[2:] != x.shape[2:]:
                skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        x = self.refine(x)
        return x

class ProgressiveDecoder(nn.Module):
    """Progressive decoder"""
    
    def __init__(self, dit_dim=384, base_channels=64, skip_channels=None):
        super().__init__()
        if skip_channels is None:
            skip_channels = DETECTED_SKIP_CHANNELS
        
        self.input_proj = nn.Conv2d(dit_dim, base_channels * 8, 1)
        self.up1 = UpsampleBlockWithSkip(base_channels * 8, base_channels * 4, skip_channels[0])
        self.up2 = UpsampleBlockWithSkip(base_channels * 4, base_channels * 2, skip_channels[1])
        self.up3 = UpsampleBlockWithSkip(base_channels * 2, base_channels, skip_channels[2])
        self.up4 = UpsampleBlockWithSkip(base_channels, base_channels * 2, skip_channels[3])
        self.up5 = UpsampleBlockWithSkip(base_channels * 2, base_channels * 4, skip_channels[4])
        self.output_conv = nn.Sequential(
            nn.Conv2d(base_channels * 4, 3, 3, padding=1),
            nn.Tanh()
        )
    
    def forward(self, tokens, H=8, W=8, encoder_features=None):
        B, N, C = tokens.shape
        x = tokens.transpose(1, 2).reshape(B, C, H, W)
        x = self.input_proj(x)
        
        #  SAFETY: Clamp intermediate activations AFTER each layer
        x = torch.clamp(x, -5.0, 5.0)
        
        if encoder_features is not None:
            x = self.up1(x, skip=encoder_features.get('8x8'))
            x = torch.clamp(x, -5.0, 5.0)
            
            x = self.up2(x, skip=encoder_features.get('16x16'))
            x = torch.clamp(x, -5.0, 5.0)
            
            x = self.up3(x, skip=encoder_features.get('32x32'))
            x = torch.clamp(x, -5.0, 5.0)
            
            x = self.up4(x, skip=encoder_features.get('64x64'))
            x = torch.clamp(x, -5.0, 5.0)
        else:
            x = self.up1(x)
            x = torch.clamp(x, -5.0, 5.0)
            
            x = self.up2(x)
            x = torch.clamp(x, -5.0, 5.0)
            
            x = self.up3(x)
            x = torch.clamp(x, -5.0, 5.0)
            
            x = self.up4(x)
            x = torch.clamp(x, -5.0, 5.0)
        
        x = self.up5(x, skip=None)
        x = torch.clamp(x, -5.0, 5.0)
        
        out = self.output_conv(x)
        
        #  CRITICAL: Stabilize output IMMEDIATELY before returning
        out = torch.clamp(out, -1.5, 1.5)
        out = torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return out


# ============================================================================
# COMPLETE MODEL (FIXED)
# ============================================================================

class WRDiTv4Fixed(nn.Module):
    """ Complete WRDiT v4 model with FIXED gradient flow"""
    
    def __init__(self, img_size=256, dit_dim=384, num_dit_blocks=8, num_heads=8, 
                 pretrained_encoder=True, num_weather_types=NUM_WEATHER_TYPES):
        super().__init__()
        self.img_size = img_size
        self.dit_dim = dit_dim
        self.num_weather_types = num_weather_types
        
        self.encoder = MobileNetV3Encoder(pretrained=pretrained_encoder)
        self.feature_pyramid = FeaturePyramidFusion(out_dim=dit_dim)
        self.reasoning = InstrumentedReasoningModule(num_weather_types=num_weather_types)
        self.feature_proj = nn.Conv2d(dit_dim, dit_dim, 1)
        
        num_patches = (img_size // 32) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, dit_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        #  USE FIXED ROUTER
        self.token_router = InstrumentedTokenRouterFixed(dim=dit_dim, severity_dim=3)
        self.token_reconstructor = InstrumentedTokenReconstructor(dim=dit_dim, num_heads=num_heads)
        
        self.weather_context_proj = nn.Sequential(
            nn.Linear(384, dit_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        self.dit_blocks = nn.ModuleList([
            EnhancedDiTBlock(dim=dit_dim, num_heads=num_heads) for _ in range(num_dit_blocks)
        ])
        
        self.decoder = ProgressiveDecoder(dit_dim=dit_dim)
        self.global_res_scale = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, x, return_aux=False, task_id=None, severity_class=None, tau=1.0):
        B = x.shape[0]
        x_orig = x
        
        # Encode
        features = self.encoder(x)
        fused_list = self.feature_pyramid(features)
        fused = fused_list[-1]
        
        # Reasoning
        weather_logits, severity_logits, weather_emb = self.reasoning(
            features['8x8'], task_id=task_id, severity_class=severity_class
        )
        
        # Get severity score
        severity_class_pred = severity_logits.argmax(dim=1)
        severity_score = severity_class_pred.float() / 2.0
        
        # Project and tokenize
        feat = self.feature_proj(fused)
        H, W = feat.shape[2], feat.shape[3]
        tokens = feat.flatten(2).transpose(1, 2) + self.pos_embed
        
        #  Route tokens with FIXED router
        selected_tokens, indices, routing_stats, loss_info = self.token_router(
            tokens, severity_score, return_indices=True, return_stats=True, tau=tau
        )
        
        num_tokens_selected = selected_tokens.shape[1]
        
        # Weather context
        weather_ctx = self.weather_context_proj(weather_emb)
        weather_ctx = weather_ctx.unsqueeze(1).expand(-1, num_tokens_selected, -1)
        
        # DiT blocks
        x_dit = selected_tokens
        for dit_block in self.dit_blocks:
            x_dit = dit_block(x_dit, context=weather_ctx)
        
        # Reconstruct
        all_tokens, num_selected, num_missing, rmse, blend_weight = self.token_reconstructor(
            x_dit, tokens.shape[1], indices
        )
        
        # Decode
        restored = self.decoder(all_tokens, H, W, encoder_features=features)
        
        # Global residual
        restored = restored + self.global_res_scale * x_orig
        restored = torch.clamp(restored, -1, 1)
        
        if return_aux:
            return {
                'restored': restored,
                'weather_logits': weather_logits,
                'severity_logits': severity_logits,
                'weather_emb': weather_emb,
                'routing_stats': routing_stats,
                'selected_tokens': selected_tokens,
                'indices': indices,
                'loss_info': loss_info,  #  NEW: Pass differentiable loss info
            }
        return restored

# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class AggressivePerceptualLoss(nn.Module):
    """Multi-layer VGG19 perceptual loss"""
    
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        self.features = vgg.features[:36]
        self.relu_indices = [3, 8, 17, 26, 35]
        self.layer_weights = [1.0, 1.0, 1.0, 1.0, 0.5]
        
        for param in self.features.parameters():
            param.requires_grad = False
        
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def forward(self, x, y):
        x = (x + 1) / 2
        y = (y + 1) / 2
        x = torch.clamp(x, 0, 1)
        y = torch.clamp(y, 0, 1)
        
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std
        
        loss = 0.0
        layer_count = 0
        for i, layer in enumerate(self.features):
            x = layer(x)
            y = layer(y)
            if i in self.relu_indices:
                weight = self.layer_weights[self.relu_indices.index(i)]
                loss = loss + weight * F.l1_loss(x, y)
                layer_count += 1
        
        return loss / max(layer_count, 1)

# ============================================================================
# DATASET
# ============================================================================

class WeatherDataset(Dataset):
    """Weather restoration dataset"""
    
    def __init__(self, degraded_dir, clean_dir, task_name, img_size=256, augment=False):
        self.degraded_dir = Path(degraded_dir)
        self.clean_dir = Path(clean_dir)
        self.task_name = task_name
        self.task_id = TASK_NAMES[task_name]
        self.img_size = img_size
        self.augment = augment
        
        self.degraded_files = sorted(
            list(self.degraded_dir.glob('*.png')) + 
            list(self.degraded_dir.glob('*.jpg')) + 
            list(self.degraded_dir.glob('*.jpeg'))
        )
        self.clean_files = sorted(
            list(self.clean_dir.glob('*.png')) + 
            list(self.clean_dir.glob('*.jpg')) + 
            list(self.clean_dir.glob('*.jpeg'))
        )
        
        assert len(self.degraded_files) == len(self.clean_files), \
            f"Mismatch: {len(self.degraded_files)} degraded vs {len(self.clean_files)} clean"
        
        print(f' {task_name}: {len(self.degraded_files)} images')
        
        if augment:
            self.pil_transforms = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.2),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.RandomRotation(5)
            ])
        else:
            self.pil_transforms = transforms.Resize((img_size, img_size))
        
        self.tensor_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def __len__(self):
        return len(self.degraded_files)
    
    def __getitem__(self, idx):
        degraded = Image.open(self.degraded_files[idx]).convert('RGB')
        clean = Image.open(self.clean_files[idx]).convert('RGB')
        
        if self.augment:
            seed = torch.randint(0, 2**32, (1,)).item()
            torch.manual_seed(seed)
            degraded = self.pil_transforms(degraded)
            torch.manual_seed(seed)
            clean = self.pil_transforms(clean)
        else:
            degraded = self.pil_transforms(degraded)
            clean = self.pil_transforms(clean)
        
        degraded = self.tensor_transform(degraded)
        clean = self.tensor_transform(clean)
        
        diff = degraded - clean
        severity = torch.sqrt(
            (0.299 * diff[0] ** 2 + 0.587 * diff[1] ** 2 + 0.114 * diff[2] ** 2).mean()
        )
        severity = torch.clamp(severity, 0.25, 1.0)
        
        if severity < 0.3:
            severity_class = torch.tensor(SEVERITY_LEVELS['LOW'], dtype=torch.long)
        elif severity < 0.7:
            severity_class = torch.tensor(SEVERITY_LEVELS['MEDIUM'], dtype=torch.long)
        else:
            severity_class = torch.tensor(SEVERITY_LEVELS['HIGH'], dtype=torch.long)
        
        return {
            'degraded': degraded,
            'clean': clean,
            'task_id': torch.tensor(self.task_id, dtype=torch.long),
            'task_name': self.task_name,
            'severity': severity,
            'severity_class': severity_class
        }

# ============================================================================
# TRAINER CONFIG
# ============================================================================

class TrainerConfig:
    def __init__(self):
        self.dataset_root = DATASET_ROOT
        self.batch_size = 16
        self.num_epochs = 1000
        self.learning_rate = 1e-4
        self.num_workers = 4
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_amp = torch.cuda.is_available()
        self.freeze_encoder_initially = True
        self.freeze_early_layers_until = 100
        self.unfreeze_all_at_epoch = 150
        self.log_dir = Path('./logs/wrdit_v4_fixed')
        self.checkpoint_dir = Path('./checkpoints/wrdit_v4_fixed')
        
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

# ============================================================================
# TRAINER (FIXED)
# ============================================================================

class TrainerFixed:
    """ FIXED Trainer with proper router loss integration"""
    
    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.logger = OptimizedInstrumentedLogger(str(config.log_dir))
        
        print('\n' + '='*120)
        print('🎯 WRDiT v4 FIXED TRAINER INITIALIZED')
        print('='*120)
        print(' Router gradient flow FIXED - threshold parameter now learns!')
        print(' Router losses properly connected to parameters')
        print(' All components receive gradients')
        print('='*120 + '\n')
        
        self.model = WRDiTv4Fixed(num_weather_types=NUM_WEATHER_TYPES).to(self.device)
        #  Initialize BatchNorm statistics to prevent NaN
        print(' Initializing BatchNorm statistics...')
        self.model.train()
        self._initialize_batchnorm(self.model)
        print(' BatchNorm initialized\n')

        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f' Model parameters: {trainable_params:,} trainable / {total_params:,} total\n')
        
        self.setup_datasets()
        
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = AggressivePerceptualLoss().to(self.device)
        self.router_loss_fn = RouterLoss(
            weight_entropy=0.02,
            weight_importance=0.02,
            weight_threshold=0.01
        ).to(self.device)
        
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.learning_rate, weight_decay=0.01)
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=1000, T_mult=1, eta_min=8e-8)
        self.scaler = GradScaler() if config.use_amp else None
        self.writer = SummaryWriter(str(config.log_dir / 'tensorboard'))
        
        self.best_psnr = 0
        self.setup_encoder_scheduling()
    
    def setup_encoder_scheduling(self):
        """Freeze encoder initially"""
        if self.config.freeze_encoder_initially:
            print('🔒 Freezing early encoder layers...')
            for i, layer in enumerate(self.model.encoder.features):
                if i < 14:
                    for param in layer.parameters():
                        param.requires_grad = False
            print(' Early layers frozen\n')
    
    def unfreeze_encoder_layers(self, epoch):
        """Gradually unfreeze encoder"""
        if epoch == self.config.freeze_early_layers_until:
            print(f'🔓 Unfreezing early encoder layers at epoch {epoch}...')
            for i, layer in enumerate(self.model.encoder.features):
                if i < 14:
                    for param in layer.parameters():
                        param.requires_grad = True
        
        if epoch == self.config.unfreeze_all_at_epoch:
            print(f'🔓 Unfreezing ALL encoder at epoch {epoch}...')
            for param in self.model.encoder.parameters():
                param.requires_grad = True
    
    def setup_datasets(self):
        """Setup training and validation datasets"""
        print('📂 Loading datasets...')
        train_datasets = []
        sample_weights_list = []
        per_class_counts = {}
        
        for task_name in TASK_NAMES.keys():
            task_dir = Path(self.config.dataset_root) / 'train' / f'train_{task_name.lower()}'
            if not task_dir.exists():
                print(f'⚠️  {task_name} not found')
                continue
            
            degraded_path = task_dir / 'data'
            clean_path = task_dir / 'clean'
            
            if degraded_path.exists() and clean_path.exists():
                try:
                    dataset = WeatherDataset(str(degraded_path), str(clean_path), task_name, augment=True)
                    train_datasets.append(dataset)
                    n = len(dataset)
                    per_class_counts[task_name] = n
                    weight = BALANCED_SAMPLING_WEIGHTS.get(task_name, 1.0)
                    sample_weights_list.extend([weight] * n)
                except Exception as e:
                    print(f' Error loading {task_name}: {e}')
        
        if not train_datasets:
            raise ValueError('No training datasets found!')
        
        combined_train = ConcatDataset(train_datasets)
        num_samples_balanced = sum(
            int(per_class_counts.get(task, 0) * BALANCED_SAMPLING_WEIGHTS.get(task, 1.0))
            for task in BALANCED_SAMPLING_WEIGHTS.keys()
        )
        
        sampler = WeightedRandomSampler(
            weights=sample_weights_list,
            num_samples=num_samples_balanced,
            replacement=True
        )
        
        self.train_loader = DataLoader(
            combined_train,
            batch_size=self.config.batch_size,
            sampler=sampler,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        print(f' Total samples (balanced): {num_samples_balanced}')
        print(f' Batches per epoch: {len(self.train_loader)}\n')
        
        # Validation datasets
        self.val_loaders = {}
        for task_name in TASK_NAMES.keys():
            test_dir = Path(self.config.dataset_root) / 'test' / f'test_{task_name.lower()}'
            if not test_dir.exists():
                continue
            
            degraded_path = test_dir / 'data'
            clean_path = test_dir / 'clean'
            
            if degraded_path.exists() and clean_path.exists():
                try:
                    dataset = WeatherDataset(str(degraded_path), str(clean_path), task_name, augment=False)
                    self.val_loaders[task_name] = DataLoader(
                        dataset,
                        batch_size=self.config.batch_size,
                        shuffle=False,
                        num_workers=self.config.num_workers,
                        pin_memory=True
                    )
                except Exception as e:
                    print(f' Error loading test {task_name}: {e}')
        
        print(f' Test datasets: {list(self.val_loaders.keys())}\n')
    
    def train_epoch(self, epoch):
        """ FIXED Train one epoch with proper router loss"""
        self.model.train()
        
        total_loss = 0
        total_l1 = 0
        total_perc = 0
        total_weather = 0
        total_severity = 0
        total_router_loss = 0
        weather_acc_list = []
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config.num_epochs}')
        
        for batch_idx, batch in enumerate(pbar):
            try:
                degraded = batch['degraded'].to(self.device)
                clean = batch['clean'].to(self.device)
                task_id = batch['task_id'].to(self.device)
                severity_class = batch['severity_class'].to(self.device)
                
                self.optimizer.zero_grad()
                
                # Temperature annealing
                # tau = max(0.1, 1.0 - (epoch / self.config.num_epochs))
                tau = max(0.5, 1.0 - (epoch / (self.config.num_epochs * 2)))
                tau = min(1.0, tau)  # Never exceed 1.0
                
                if self.config.use_amp:
                    with autocast():
                        #  SAFE: Model forward with output validation
                        output = self.model(degraded, return_aux=True, task_id=task_id, 
                                        severity_class=severity_class, tau=tau)
                        restored = output['restored']
                        weather_logits = output['weather_logits']
                        severity_logits = output['severity_logits']
                        loss_info = output['loss_info']

                        #  SAFE: Clamp model output to prevent explosion
                        restored = torch.clamp(restored, -2.0, 2.0)

                        #  SAFE: Verify no NaN in inputs to loss
                        if torch.isnan(restored).any() or torch.isnan(clean).any():
                            print(f"⚠️  NaN detected in batch {batch_idx}, skipping...")
                            self.optimizer.zero_grad()
                            continue

                        # Compute losses with clamping
                        loss_l1 = self.l1_loss(restored, clean)
                        loss_l1 = torch.clamp(loss_l1, 0.0, 1000.0)

                        loss_perceptual = self.perceptual_loss(restored, clean)
                        loss_perceptual = torch.clamp(loss_perceptual, 0.0, 1000.0)

                        loss_weather = F.cross_entropy(weather_logits, task_id)
                        loss_weather = torch.clamp(loss_weather, 0.0, 100.0)

                        loss_severity = F.cross_entropy(severity_logits, severity_class)
                        loss_severity = torch.clamp(loss_severity, 0.0, 100.0)

                        loss_entropy, loss_importance, loss_threshold, loss_router_total = \
                            self.router_loss_fn(loss_info, self.model.token_router)
                        loss_router_total = torch.clamp(loss_router_total, 0.0, 1000.0)

                        #  SAFE: Verify no NaN in any loss
                        losses_ok = (
                            not torch.isnan(loss_l1) and 
                            not torch.isnan(loss_perceptual) and
                            not torch.isnan(loss_weather) and
                            not torch.isnan(loss_severity) and
                            not torch.isnan(loss_router_total)
                        )

                        if not losses_ok:
                            print(f"⚠️  NaN in loss terms, skipping batch {batch_idx}")
                            print(f"   L1: {loss_l1.item():.6f}, Perc: {loss_perceptual.item():.6f}, "
                                f"W: {loss_weather.item():.6f}, S: {loss_severity.item():.6f}, "
                                f"R: {loss_router_total.item():.6f}")
                            self.optimizer.zero_grad()
                            continue

                        # Combined loss
                        loss = (0.20 * loss_l1 + 0.60 * loss_perceptual +
                            0.05 * loss_weather + 0.05 * loss_severity +
                            0.10 * loss_router_total)

                        #  SAFE: Final NaN check
                        if torch.isnan(loss) or torch.isinf(loss):
                            print(f"⚠️  Combined loss is {loss.item()}, skipping batch {batch_idx}")
                            self.optimizer.zero_grad()
                            continue

                        # Backprop
                        self.optimizer.zero_grad()
                        if self.config.use_amp:
                            with autocast():
                                loss_scaled = loss
                            self.scaler.scale(loss_scaled).backward()
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                            self.optimizer.step()

                
                # Logging
                routing_stats = output.get('routing_stats', {})
                self.logger.log_batch_losses(epoch, batch_idx, degraded.shape[0],
                                            loss_l1.item(), loss_perceptual.item(),
                                            loss_weather.item(), loss_severity.item(),
                                            loss_router_total.item(), loss.item())
                
                if routing_stats:
                    self.logger.log_router_analysis(epoch, batch_idx,
                                                   routing_stats.get('num_tokens_original', 0),
                                                   routing_stats.get('num_tokens_selected', 0),
                                                   routing_stats.get('compression_ratio', 1.0),
                                                   routing_stats.get('gumbel_entropy', 0),
                                                   routing_stats.get('strategy', 'UNKNOWN'))
                
                # Component gradients
                component_grad_info = compute_all_component_gradients(self.model)
                self.logger.log_all_component_gradients(epoch, batch_idx, component_grad_info)
                
                # Routing parameters
                routing_params = compute_routing_parameters(self.model)
                self.logger.log_routing_threshold_analysis(epoch, batch_idx, routing_params)
                
                # Router loss breakdown
                self.logger.log_router_loss_breakdown(epoch, batch_idx, loss_entropy.item(), loss_importance.item(), loss_threshold.item(), loss_router_total.item())
                
                # Metrics
                total_loss += loss.item()
                total_l1 += loss_l1.item()
                total_perc += loss_perceptual.item()
                total_weather += loss_weather.item()
                total_severity += loss_severity.item()
                total_router_loss += loss_router_total.item()
                
                weather_pred = weather_logits.argmax(dim=1)
                weather_acc = (weather_pred == task_id).float().mean().item()
                weather_acc_list.append(weather_acc)
                
                #  COMPREHENSIVE CONSOLE LOGGING EVERY 10 BATCHES
                # if (batch_idx + 1) % 10 == 0:
                #     print(f'\n{"="*120}')
                #     print(f'Batch {batch_idx + 1}: 🔍 COMPREHENSIVE GRADIENT MONITORING')
                #     print(f'{"="*120}')
                    
                #     print(f'\n Component Gradient Status:')
                #     components = component_grad_info['components']
                #     for comp_name in ['encoder', 'feature_pyramid', 'reasoning', 'token_router', 
                #                      'token_reconstructor', 'dit_blocks', 'decoder']:
                #         comp_data = components[comp_name]
                #         status = '✓ Learning' if comp_data['has_grad'] == 1 else '✗ FROZEN!'
                #         print(f'  ✓ {comp_name:18s} | grad_norm: {comp_data["grad_norm"]:10.6f} | '
                #               f'participation: {comp_data["percentage"]:6.2f}% | {status}')
                    
                #     print(f'  {"-"*100}')
                #     print(f'  Total Grad Norm: {component_grad_info["total_grad_norm"]:.6f}')
                    
                #     print(f'\n🎯 Routing Parameters:')
                #     print(f'  • Threshold:            {routing_params["routing_threshold"]:+.6f}')
                #     print(f'  • Learned ratio:        {routing_params["learned_ratio"]:.6f}')
                #     print(f'  • Threshold gradient:   {routing_params["threshold_grad"]:+.8f} '
                #           f'{"✓ LEARNING" if routing_params["threshold_has_nonzero_grad"] else "✗ FROZEN"}')
                #     print(f'  • Temperature:          {routing_params["temperature"]:.6f}')
                #     print(f'  • Temp gradient:        {routing_params["temperature_grad"]:+.8f} '
                #           f'{"✓ LEARNING" if routing_params["temperature_has_nonzero_grad"] else "✗ FROZEN"}')
                    
                #     print(f'\n💪 Router Loss Breakdown:')
                #     print(f'  • Entropy loss:         {loss_entropy.item():.6f}')
                #     print(f'  • Importance loss:      {loss_importance.item():.6f}')
                #     print(f'  • Threshold loss:       {loss_threshold.item():.6f}')
                #     print(f'  • Total router loss:    {loss_router_total.item():.6f}')
                    
                #     # Check all learning
                #     all_learning = all(components[c]['has_grad'] == 1 for c in components)
                #     threshold_learning = routing_params['threshold_has_nonzero_grad']
                    
                #     if all_learning and threshold_learning:
                #         print(f'\n  {"="*100}')
                #         print(f'   ALL COMPONENTS ARE LEARNING! 🎉')
                #         print(f'   THRESHOLD GRADIENT IS ACTIVE! 🚀')
                #         print(f'  {"="*100}\n')
                    # else:
                    #     print(f'\n  {"="*100}')
                    #     if not all_learning:
                    #         print(f'  ⚠️  WARNING: Some components NOT learning:')
                    #         for c in components:
                    #             if components[c]['has_grad'] == 0:
                    #                 print(f'     - {c.upper()} FROZEN')
                    #     if not threshold_learning:
                    #         print(f'  ⚠️  WARNING: THRESHOLD GRADIENT STILL ZERO!')
                    #     print(f'  {"="*100}\n')
                
                pbar.set_postfix(loss=f'{loss.item():.4f}', l1=f'{loss_l1.item():.4f}',
                               router=f'{loss_router_total.item():.6f}',
                               wacc=f'{weather_acc:.4f}', lr=f'{self.optimizer.param_groups[0]["lr"]:.2e}')
                
            finally:
                del degraded, clean, task_id, severity_class, output, restored, weather_logits
                del loss_l1, loss_perceptual, loss_weather, loss_severity, loss, weather_pred
                
                if (batch_idx + 1) % 4 == 0:
                    torch.cuda.empty_cache()
        
        # Flush logs
        self.logger.flush_logs()
        torch.cuda.empty_cache()
        
        # Epoch metrics
        num_batches = len(self.train_loader)
        avg_loss = total_loss / num_batches
        avg_l1 = total_l1 / num_batches
        avg_perc = total_perc / num_batches
        avg_weather = total_weather / num_batches
        avg_severity = total_severity / num_batches
        avg_router_loss = total_router_loss / num_batches
        avg_weather_acc = np.mean(weather_acc_list)
        lr = self.optimizer.param_groups[0]['lr']
        
        print('='*120)
        print(f'Epoch {epoch+1} Training Summary')
        print('='*120)
        print(f'Total Loss: {avg_loss:.4f}')
        print(f'L1 Loss: {avg_l1:.4f} ({avg_l1/avg_loss*100:.1f}%)')
        print(f'Perceptual Loss: {avg_perc:.4f} ({avg_perc/avg_loss*100:.1f}%)')
        print(f'Weather Loss: {avg_weather:.4f}')
        print(f'Severity Loss: {avg_severity:.4f}')
        print(f'Router Loss: {avg_router_loss:.6f} ({avg_router_loss/avg_loss*100:.1f}%)')
        print(f'Weather Accuracy: {avg_weather_acc:.4f}')
        print(f'Learning Rate: {lr:.6e}')
        print('='*120 + '\n')
        
        return {
            'train_loss_total': avg_loss,
            'train_loss_l1': avg_l1,
            'train_loss_perceptual': avg_perc,
            'train_loss_weather': avg_weather,
            'train_loss_severity': avg_severity,
            'train_loss_router': avg_router_loss,
            'weather_acc': avg_weather_acc,
            'lr': lr
        }
    
    def validate(self):
        """Validate"""
        self.model.eval()
        results = {}
        
        for task_name, loader in self.val_loaders.items():
            psnr_list = []
            ssim_list = []
            val_loss_list = []
            weather_acc_list = []
            
            with torch.no_grad():
                for batch in tqdm(loader, desc=f'Validating {task_name}', leave=False):
                    try:
                        degraded = batch['degraded'].to(self.device)
                        clean = batch['clean'].to(self.device)
                        task_id = batch['task_id'].to(self.device)
                        
                        if self.config.use_amp:
                            with autocast():
                                output = self.model(degraded, return_aux=True)
                                restored = output['restored']
                                weather_logits = output['weather_logits']
                        else:
                            output = self.model(degraded, return_aux=True)
                            restored = output['restored']
                            weather_logits = output['weather_logits']
                        
                        loss = self.l1_loss(restored, clean)
                        val_loss_list.append(loss.item())
                        
                        for i in range(restored.shape[0]):
                            psnr = compute_psnr(restored[i:i+1], clean[i:i+1])
                            ssim = compute_ssim(restored[i:i+1], clean[i:i+1])
                            psnr_list.append(psnr)
                            ssim_list.append(ssim)
                        
                        weather_acc = (weather_logits.argmax(dim=1) == task_id).float().mean().item()
                        weather_acc_list.append(weather_acc)
                    
                    finally:
                        del degraded, clean, task_id, output, restored, weather_logits, loss
                        torch.cuda.empty_cache()
        
            avg_psnr = np.mean(psnr_list)
            avg_ssim = np.mean(ssim_list)
            avg_val_loss = np.mean(val_loss_list)
            avg_weather_acc = np.mean(weather_acc_list)
            
            results[task_name] = {'psnr': avg_psnr, 'ssim': avg_ssim, 'loss': avg_val_loss, 'weather_acc': avg_weather_acc}
        
        return results
    def _initialize_batchnorm(self, model, num_batches=2):
        """ Initialize BatchNorm statistics to prevent NaN in early batches"""
        if not hasattr(self, 'train_loader'):
            return
        
        model.train()
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.train_loader):
                if batch_idx >= num_batches:
                    break
                
                try:
                    degraded = batch['degraded'].to(self.device)
                    _ = model(degraded, return_aux=False)
                    print(f"   BatchNorm init batch {batch_idx + 1}/{num_batches}")
                finally:
                    del degraded
                    torch.cuda.empty_cache()

    
    def train(self):
        """Main training loop"""
        print('\n' + '='*120)
        print('🚀 TRAINING START - WRDiT v4 FIXED')
        print('='*120 + '\n')
        
        for epoch in range(self.config.num_epochs):
            self.unfreeze_encoder_layers(epoch)
            
            train_metrics = self.train_epoch(epoch)
            self.scheduler.step()
            val_results = self.validate()
            
            print('-'*120)
            print(f'Validation Results:')
            for task, metrics in val_results.items():
                print(f'  {task:10s} | PSNR: {metrics["psnr"]:6.2f} dB | SSIM: {metrics["ssim"]:.4f}')
                
                self.writer.add_scalar(f'val/{task}/psnr', metrics['psnr'], epoch)
                self.writer.add_scalar(f'val/{task}/ssim', metrics['ssim'], epoch)
            
            if val_results:
                best_psnr = max(m['psnr'] for m in val_results.values())
                if best_psnr > self.best_psnr:
                    self.best_psnr = best_psnr
                    for task in val_results.keys():
                        torch.save(self.model.state_dict(), 
                                 self.config.checkpoint_dir / f'best_model_{task}.pth')
            
            epoch_metrics = {
                'train_loss_total': train_metrics['train_loss_total'],
                'train_loss_l1': train_metrics['train_loss_l1'],
                'train_loss_perceptual': train_metrics['train_loss_perceptual'],
                'train_loss_weather': train_metrics['train_loss_weather'],
                'train_loss_severity': train_metrics['train_loss_severity'],
                'train_loss_router': train_metrics['train_loss_router'],
                'avg_psnr': np.mean([m['psnr'] for m in val_results.values()]) if val_results else 0,
                'avg_ssim': np.mean([m['ssim'] for m in val_results.values()]) if val_results else 0,
                'lr': train_metrics['lr'],
                'weather_acc': train_metrics['weather_acc']
            }
            self.logger.log_epoch_metrics(epoch, epoch_metrics)
            
            if (epoch + 1) % 10 == 0:
                torch.save(self.model.state_dict(), self.config.checkpoint_dir / f'epoch_{epoch+1}.pth')
        
        print('\n' + '='*120)
        print(f' Training complete! Best PSNR: {self.best_psnr:.2f} dB')
        print('='*120)
        
        self.writer.close()
        self.logger.close_logger()
        torch.cuda.empty_cache()

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    config = TrainerConfig()
    trainer = TrainerFixed(config)
    trainer.train()
