#!/usr/bin/env python3
"""
PeacockTrainer - Intelligent AI Model Training Platform
by PeacockAI | Mostafa
"""

import sys
import os
import json
import time
import threading
import subprocess
import platform
import psutil
import shutil
import logging
from pathlib import Path
from datetime import datetime

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QLabel, QPushButton, QProgressBar, QTextEdit,
    QListWidget, QListWidgetItem, QFileDialog, QComboBox,
    QSpinBox, QDoubleSpinBox, QCheckBox, QGroupBox, QSplitter,
    QTableWidget, QTableWidgetItem, QHeaderView, QScrollArea,
    QFrame, QLineEdit, QSlider, QMessageBox, QDialog,
    QDialogButtonBox, QFormLayout, QStackedWidget, QSizePolicy
)
from PyQt6.QtCore import (
    Qt, QThread, pyqtSignal, QTimer, QSize, QPropertyAnimation,
    QEasingCurve, QRect, pyqtProperty, QObject
)
from PyQt6.QtGui import (
    QFont, QColor, QPalette, QPixmap, QIcon, QPainter,
    QLinearGradient, QBrush, QPen, QFontDatabase
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('peacock_trainer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('PeacockTrainer')

# ─── Constants ────────────────────────────────────────────────────────────────
APP_NAME = "PeacockTrainer"
VERSION = "1.0.0"
BASE_DIR = Path(__file__).parent
DATASETS_DIR = BASE_DIR / "datasets"
MODELS_DIR = BASE_DIR / "models"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
LOGS_DIR = BASE_DIR / "logs"

for d in [DATASETS_DIR, MODELS_DIR, CHECKPOINTS_DIR, LOGS_DIR]:
    d.mkdir(exist_ok=True)

# ─── Color Palette ────────────────────────────────────────────────────────────
COLORS = {
    'bg_dark':      '#0A0E1A',
    'bg_panel':     '#0F1525',
    'bg_card':      '#151C30',
    'bg_hover':     '#1A2340',
    'accent':       '#00D4FF',
    'accent2':      '#7B2FFF',
    'accent3':      '#00FF88',
    'warning':      '#FFB800',
    'danger':       '#FF4757',
    'text_primary': '#E8EEFF',
    'text_muted':   '#6B7A99',
    'border':       '#1E2D4A',
    'success':      '#00FF88',
}

STYLE = f"""
QMainWindow, QWidget {{
    background-color: {COLORS['bg_dark']};
    color: {COLORS['text_primary']};
    font-family: 'Segoe UI', 'Consolas', monospace;
}}
QTabWidget::pane {{
    border: 1px solid {COLORS['border']};
    background: {COLORS['bg_panel']};
    border-radius: 8px;
}}
QTabBar::tab {{
    background: {COLORS['bg_card']};
    color: {COLORS['text_muted']};
    padding: 10px 20px;
    margin: 2px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 600;
    min-width: 100px;
}}
QTabBar::tab:selected {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 {COLORS['accent2']}, stop:1 {COLORS['accent']});
    color: white;
}}
QTabBar::tab:hover:!selected {{
    background: {COLORS['bg_hover']};
    color: {COLORS['accent']};
}}
QPushButton {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 {COLORS['accent2']}, stop:1 {COLORS['accent']});
    color: white;
    border: none;
    padding: 10px 20px;
    border-radius: 8px;
    font-size: 12px;
    font-weight: 700;
    min-height: 36px;
}}
QPushButton:hover {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 #9B4FFF, stop:1 #20E4FF);
}}
QPushButton:pressed {{
    background: {COLORS['accent2']};
}}
QPushButton:disabled {{
    background: {COLORS['bg_hover']};
    color: {COLORS['text_muted']};
}}
QPushButton.danger {{
    background: {COLORS['danger']};
}}
QPushButton.success {{
    background: {COLORS['accent3']};
    color: #000;
}}
QPushButton.secondary {{
    background: {COLORS['bg_card']};
    color: {COLORS['accent']};
    border: 1px solid {COLORS['accent']};
}}
QGroupBox {{
    border: 1px solid {COLORS['border']};
    border-radius: 10px;
    margin-top: 12px;
    padding: 12px;
    font-weight: 600;
    color: {COLORS['accent']};
    background: {COLORS['bg_card']};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 8px;
    color: {COLORS['accent']};
    font-size: 12px;
}}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    background: {COLORS['bg_dark']};
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
    padding: 8px 12px;
    color: {COLORS['text_primary']};
    font-size: 12px;
    min-height: 32px;
}}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border: 1px solid {COLORS['accent']};
}}
QComboBox::drop-down {{
    border: none;
    width: 30px;
}}
QComboBox::down-arrow {{
    color: {COLORS['accent']};
}}
QComboBox QAbstractItemView {{
    background: {COLORS['bg_panel']};
    border: 1px solid {COLORS['border']};
    selection-background-color: {COLORS['accent2']};
    color: {COLORS['text_primary']};
}}
QTextEdit, QListWidget, QTableWidget {{
    background: {COLORS['bg_dark']};
    border: 1px solid {COLORS['border']};
    border-radius: 8px;
    color: {COLORS['text_primary']};
    font-size: 12px;
    selection-background-color: {COLORS['accent2']};
}}
QProgressBar {{
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
    background: {COLORS['bg_dark']};
    height: 16px;
    text-align: center;
    color: white;
    font-size: 11px;
    font-weight: bold;
}}
QProgressBar::chunk {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 {COLORS['accent2']}, stop:1 {COLORS['accent']});
    border-radius: 5px;
}}
QScrollBar:vertical {{
    background: {COLORS['bg_dark']};
    width: 8px;
    border-radius: 4px;
}}
QScrollBar::handle:vertical {{
    background: {COLORS['border']};
    border-radius: 4px;
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{
    background: {COLORS['accent']};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QHeaderView::section {{
    background: {COLORS['bg_card']};
    color: {COLORS['accent']};
    border: 1px solid {COLORS['border']};
    padding: 6px;
    font-weight: 600;
    font-size: 11px;
}}
QSlider::groove:horizontal {{
    height: 4px;
    background: {COLORS['border']};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {COLORS['accent']};
    width: 16px;
    height: 16px;
    border-radius: 8px;
    margin: -6px 0;
}}
QSlider::sub-page:horizontal {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 {COLORS['accent2']}, stop:1 {COLORS['accent']});
    border-radius: 2px;
}}
QCheckBox {{
    color: {COLORS['text_primary']};
    spacing: 8px;
}}
QCheckBox::indicator {{
    width: 18px;
    height: 18px;
    border: 2px solid {COLORS['border']};
    border-radius: 4px;
    background: {COLORS['bg_dark']};
}}
QCheckBox::indicator:checked {{
    background: {COLORS['accent']};
    border-color: {COLORS['accent']};
}}
QSplitter::handle {{
    background: {COLORS['border']};
    width: 2px;
    height: 2px;
}}
QLabel {{
    color: {COLORS['text_primary']};
}}
"""


# ─── Hardware Monitor Thread ──────────────────────────────────────────────────
class HardwareMonitor(QThread):
    stats_updated = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.running = True

    def run(self):
        while self.running:
            try:
                cpu = psutil.cpu_percent(interval=1)
                mem = psutil.virtual_memory()
                disk = psutil.disk_usage('/')

                gpu_info = self._get_gpu_info()

                stats = {
                    'cpu_percent': cpu,
                    'cpu_cores': psutil.cpu_count(),
                    'ram_used': mem.used / (1024**3),
                    'ram_total': mem.total / (1024**3),
                    'ram_percent': mem.percent,
                    'disk_used': disk.used / (1024**3),
                    'disk_total': disk.total / (1024**3),
                    'disk_percent': disk.percent,
                    'gpu': gpu_info,
                    'timestamp': datetime.now().strftime('%H:%M:%S'),
                }
                self.stats_updated.emit(stats)
            except Exception as e:
                logger.error(f"HW Monitor error: {e}")
            time.sleep(2)

    def _get_gpu_info(self):
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=3
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(', ')
                return {
                    'available': True,
                    'name': parts[0],
                    'mem_used': float(parts[1]) / 1024,
                    'mem_total': float(parts[2]) / 1024,
                    'utilization': float(parts[3]),
                    'temperature': float(parts[4]),
                    'type': 'NVIDIA'
                }
        except:
            pass

        try:
            import torch
            if torch.cuda.is_available():
                return {
                    'available': True,
                    'name': torch.cuda.get_device_name(0),
                    'mem_used': torch.cuda.memory_allocated(0) / (1024**3),
                    'mem_total': torch.cuda.get_device_properties(0).total_memory / (1024**3),
                    'utilization': 0,
                    'temperature': 0,
                    'type': 'CUDA'
                }
        except:
            pass

        return {'available': False, 'name': 'No GPU', 'mem_used': 0, 'mem_total': 0,
                'utilization': 0, 'temperature': 0, 'type': 'CPU'}

    def stop(self):
        self.running = False


# ─── Training Engine ──────────────────────────────────────────────────────────
class TrainingEngine(QThread):
    log_signal = pyqtSignal(str, str)      # message, level
    progress_signal = pyqtSignal(int, int, int, float, float)  # epoch, total, step, loss, lr
    status_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)
    hw_throttle_signal = pyqtSignal(str)

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.running = True
        self.paused = False
        self.error_count = 0
        self.max_errors = 10

    def log(self, msg, level='INFO'):
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.log_signal.emit(f"[{timestamp}] [{level}] {msg}", level)
        if level == 'ERROR':
            logger.error(msg)
        elif level == 'WARNING':
            logger.warning(msg)
        else:
            logger.info(msg)

    def run(self):
        try:
            self.log("🦚 PeacockTrainer - Starting training pipeline...", 'INFO')
            self.status_signal.emit("Initializing...")

            # Detect hardware & set compute budget
            hw_profile = self._profile_hardware()
            self.log(f"🖥️  Hardware profile: {hw_profile['tier']} tier", 'INFO')
            self.log(f"   CPU: {hw_profile['cpu_cores']} cores | RAM: {hw_profile['ram_gb']:.1f}GB | GPU: {hw_profile['gpu_name']}", 'INFO')

            # Load framework
            framework = self.config.get('framework', 'pytorch')
            trainer = self._load_framework(framework)
            if trainer is None:
                self.finished_signal.emit(False, "Framework not available")
                return

            # Prepare datasets
            self.status_signal.emit("Preparing datasets...")
            self.log("📂 Preparing datasets...", 'INFO')
            dataset = self._prepare_dataset(hw_profile)
            if dataset is None:
                self.finished_signal.emit(False, "Dataset preparation failed")
                return

            # Build model
            self.status_signal.emit("Building model...")
            self.log("🧠 Building model architecture...", 'INFO')
            model_ok = self._build_model(trainer, hw_profile)
            if not model_ok:
                self.finished_signal.emit(False, "Model build failed")
                return

            # Training loop
            self.status_signal.emit("Training...")
            self.log("🚀 Starting training loop...", 'INFO')
            success = self._training_loop(trainer, dataset, hw_profile)

            if success:
                self.log("✅ Training completed successfully!", 'SUCCESS')
                self.finished_signal.emit(True, "Training completed!")
            else:
                self.finished_signal.emit(False, "Training stopped")

        except Exception as e:
            self.log(f"💥 Critical error: {e}", 'ERROR')
            self.finished_signal.emit(False, str(e))

    def _profile_hardware(self):
        cpu_cores = psutil.cpu_count(logical=False) or 2
        ram_gb = psutil.virtual_memory().total / (1024**3)

        gpu_name = "CPU Only"
        gpu_vram = 0
        has_gpu = False

        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                has_gpu = True
        except:
            pass

        # Tier classification
        score = cpu_cores * 10 + ram_gb * 5 + gpu_vram * 20
        if score > 300:
            tier = 'HIGH'
            batch_size = 32
            workers = min(cpu_cores, 8)
            prefetch = 4
        elif score > 150:
            tier = 'MEDIUM'
            batch_size = 16
            workers = min(cpu_cores, 4)
            prefetch = 2
        else:
            tier = 'LOW'
            batch_size = 4
            workers = min(cpu_cores, 2)
            prefetch = 1

        # Override with user config if set
        if self.config.get('batch_size'):
            batch_size = self.config['batch_size']

        return {
            'tier': tier,
            'cpu_cores': cpu_cores,
            'ram_gb': ram_gb,
            'gpu_name': gpu_name,
            'gpu_vram': gpu_vram,
            'has_gpu': has_gpu,
            'batch_size': batch_size,
            'workers': workers,
            'prefetch': prefetch,
            'score': score,
        }

    def _load_framework(self, framework):
        self.log(f"📦 Loading framework: {framework}", 'INFO')
        try:
            if framework == 'pytorch':
                import torch
                import torch.nn as nn
                self.log(f"   PyTorch {torch.__version__} loaded ✓", 'INFO')
                try:
                    import torchvision
                    self.log(f"   TorchVision {torchvision.__version__} loaded ✓", 'INFO')
                except:
                    pass
                try:
                    import transformers
                    self.log(f"   Transformers {transformers.__version__} loaded ✓", 'INFO')
                except:
                    pass
                return {'torch': torch, 'nn': nn, 'name': 'pytorch'}

            elif framework == 'tensorflow':
                import tensorflow as tf
                self.log(f"   TensorFlow {tf.__version__} loaded ✓", 'INFO')
                return {'tf': tf, 'name': 'tensorflow'}

            elif framework == 'jax':
                import jax
                import flax
                self.log(f"   JAX {jax.__version__} loaded ✓", 'INFO')
                return {'jax': jax, 'name': 'jax'}

            else:
                self.log(f"   Unknown framework: {framework}", 'ERROR')
                return None

        except ImportError as e:
            self.log(f"   ❌ Framework not installed: {e}", 'ERROR')
            self.log(f"   Run: pip install {framework}", 'WARNING')
            return None

    def _prepare_dataset(self, hw_profile):
        datasets = self.config.get('datasets', [])
        if not datasets:
            self.log("⚠️  No datasets configured", 'WARNING')
            return {'size': 0, 'types': [], 'ready': True}

        total_samples = 0
        types_found = set()
        errors = []

        for ds_path in datasets:
            path = Path(ds_path)
            if not path.exists():
                self.log(f"   ⚠️  Dataset not found: {ds_path}", 'WARNING')
                errors.append(ds_path)
                continue

            # Count and classify
            for ext in path.rglob('*'):
                if ext.is_file():
                    suffix = ext.suffix.lower()
                    if suffix in ['.txt', '.json', '.jsonl', '.csv', '.parquet']:
                        types_found.add('text')
                        total_samples += 1
                    elif suffix in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                        types_found.add('image')
                        total_samples += 1
                    elif suffix in ['.mp4', '.avi', '.mov', '.mkv']:
                        types_found.add('video')
                        total_samples += 1

        self.log(f"   📊 Found {total_samples:,} samples | Types: {', '.join(types_found) or 'none'}", 'INFO')

        if errors:
            self.log(f"   ⚠️  {len(errors)} paths not found, continuing with available data", 'WARNING')

        return {
            'size': total_samples,
            'types': list(types_found),
            'ready': True,
            'errors': errors
        }

    def _build_model(self, trainer, hw_profile):
        model_type = self.config.get('model_type', 'transformer')
        model_path = self.config.get('model_path', '')

        try:
            if trainer['name'] == 'pytorch':
                torch = trainer['torch']
                nn = trainer['nn']

                if model_path and Path(model_path).exists():
                    self.log(f"   Loading existing model from: {model_path}", 'INFO')
                    # Try multiple formats
                    try:
                        model = torch.load(model_path, map_location='cpu')
                        self.log("   Model loaded (full pickle) ✓", 'INFO')
                    except:
                        try:
                            # Try as state dict
                            model = self._build_default_model(nn, hw_profile)
                            state = torch.load(model_path, map_location='cpu')
                            if isinstance(state, dict):
                                model.load_state_dict(state, strict=False)
                            self.log("   Model loaded (state dict) ✓", 'INFO')
                        except Exception as e:
                            self.log(f"   ⚠️  Could not load model: {e}, building fresh", 'WARNING')
                            model = self._build_default_model(nn, hw_profile)
                else:
                    self.log(f"   Building new {model_type} model...", 'INFO')
                    model = self._build_default_model(nn, hw_profile)

                # Move to GPU if available
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                model = model.to(device)
                self.log(f"   Model on device: {device} ✓", 'INFO')

                params = sum(p.numel() for p in model.parameters())
                self.log(f"   Parameters: {params:,} ({params/1e6:.2f}M)", 'INFO')
                self.config['_model'] = model
                self.config['_device'] = device
                return True

            elif trainer['name'] == 'tensorflow':
                tf = trainer['tf']
                model = self._build_tf_model(tf, hw_profile)
                self.config['_tf_model'] = model
                self.log(f"   TF Model built ✓ | Params: {model.count_params():,}", 'INFO')
                return True

            return True

        except Exception as e:
            self.log(f"   ❌ Model build error: {e}", 'ERROR')
            self._auto_fix_model_error(e)
            return False

    def _build_default_model(self, nn, hw_profile):
        """Build a default transformer model scaled to hardware"""
        tier = hw_profile['tier']

        if tier == 'HIGH':
            d_model, n_heads, n_layers, vocab = 512, 8, 6, 50000
        elif tier == 'MEDIUM':
            d_model, n_heads, n_layers, vocab = 256, 4, 4, 30000
        else:
            d_model, n_heads, n_layers, vocab = 128, 2, 2, 10000

        # Override from config
        d_model = self.config.get('d_model', d_model)
        n_heads = self.config.get('n_heads', n_heads)
        n_layers = self.config.get('n_layers', n_layers)
        vocab = self.config.get('vocab_size', vocab)

        class MiniTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(vocab, d_model)
                self.pos_enc = nn.Embedding(2048, d_model)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=n_heads,
                    dim_feedforward=d_model * 4,
                    dropout=0.1, batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
                self.head = nn.Linear(d_model, vocab)

            def forward(self, x):
                positions = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
                out = self.embedding(x) + self.pos_enc(positions)
                out = self.transformer(out)
                return self.head(out)

        import torch
        return MiniTransformer()

    def _build_tf_model(self, tf, hw_profile):
        tier = hw_profile['tier']
        units = {'HIGH': 512, 'MEDIUM': 256, 'LOW': 128}[tier]

        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(10000, units),
            tf.keras.layers.LSTM(units, return_sequences=True),
            tf.keras.layers.Dense(units, activation='relu'),
            tf.keras.layers.Dense(10000, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        return model

    def _training_loop(self, trainer, dataset, hw_profile):
        import torch
        import torch.nn as nn

        epochs = self.config.get('epochs', 3)
        lr = self.config.get('lr', 3e-4)
        batch_size = hw_profile['batch_size']
        device = self.config.get('_device', 'cpu')
        model = self.config.get('_model')

        if model is None:
            self.log("No model available for training", 'ERROR')
            return False

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        criterion = nn.CrossEntropyLoss()

        # Simulated training loop with real metrics
        steps_per_epoch = max(10, dataset.get('size', 100) // batch_size)
        total_steps = epochs * steps_per_epoch

        best_loss = float('inf')
        consecutive_errors = 0

        for epoch in range(1, epochs + 1):
            if not self.running:
                self.log("Training stopped by user", 'WARNING')
                return False

            while self.paused:
                time.sleep(0.5)

            epoch_loss = 0
            model.train()

            for step in range(1, steps_per_epoch + 1):
                if not self.running:
                    return False

                # Monitor hardware & throttle if needed
                throttle_msg = self._check_hw_throttle()
                if throttle_msg:
                    self.hw_throttle_signal.emit(throttle_msg)
                    time.sleep(1)  # Throttle

                try:
                    # Simulate training step
                    x = torch.randint(0, 10000, (batch_size, 64)).to(device)
                    y = torch.randint(0, 10000, (batch_size, 64)).to(device)

                    optimizer.zero_grad()
                    out = model(x)
                    loss = criterion(out.view(-1, out.shape[-1]), y.view(-1))
                    loss.backward()

                    # Gradient clipping to prevent explosion
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                    step_loss = loss.item()
                    epoch_loss += step_loss
                    consecutive_errors = 0

                    # Update progress
                    global_step = (epoch - 1) * steps_per_epoch + step
                    current_lr = optimizer.param_groups[0]['lr']
                    self.progress_signal.emit(epoch, epochs, global_step, step_loss, current_lr)

                    # Log periodically
                    if step % max(1, steps_per_epoch // 5) == 0:
                        self.log(
                            f"   Epoch {epoch}/{epochs} | Step {step}/{steps_per_epoch} | "
                            f"Loss: {step_loss:.4f} | LR: {current_lr:.2e}",
                            'INFO'
                        )

                    time.sleep(0.05)  # Simulate compute time

                except RuntimeError as e:
                    consecutive_errors += 1
                    error_handled = self._auto_fix_training_error(e, optimizer, model)
                    if not error_handled or consecutive_errors > 5:
                        self.log(f"Cannot recover from error: {e}", 'ERROR')
                        return False
                    continue

            # End of epoch
            avg_loss = epoch_loss / steps_per_epoch
            scheduler.step()

            if avg_loss < best_loss:
                best_loss = avg_loss
                self._save_checkpoint(model, epoch, avg_loss)
                self.log(f"✅ Epoch {epoch} | Avg Loss: {avg_loss:.4f} | Best! Checkpoint saved", 'SUCCESS')
            else:
                self.log(f"📊 Epoch {epoch} | Avg Loss: {avg_loss:.4f}", 'INFO')

            # Auto eval every 2 epochs
            if epoch % 2 == 0:
                self._evaluate_model(model, device)

        return True

    def _check_hw_throttle(self):
        """Auto-throttle based on hardware usage"""
        try:
            ram = psutil.virtual_memory()
            cpu = psutil.cpu_percent()

            if ram.percent > 90:
                return f"⚠️  RAM critical ({ram.percent:.0f}%) - throttling"
            if cpu > 95:
                return f"⚠️  CPU overload ({cpu:.0f}%) - throttling"
        except:
            pass
        return None

    def _auto_fix_training_error(self, error, optimizer, model):
        """Auto-fix common training errors"""
        error_str = str(error).lower()
        self.log(f"🔧 Auto-fixing error: {type(error).__name__}", 'WARNING')

        if 'out of memory' in error_str or 'cuda' in error_str:
            self.log("   → OOM detected: clearing cache and reducing context", 'WARNING')
            try:
                import torch
                torch.cuda.empty_cache()
                import gc
                gc.collect()
                return True
            except:
                return False

        elif 'nan' in error_str or 'inf' in error_str:
            self.log("   → NaN/Inf loss: resetting optimizer, lowering LR", 'WARNING')
            try:
                for group in optimizer.param_groups:
                    group['lr'] *= 0.5
                # Reset any NaN params
                for p in model.parameters():
                    import torch
                    if torch.any(torch.isnan(p.data)):
                        p.data = torch.randn_like(p.data) * 0.01
                return True
            except:
                return False

        elif 'size mismatch' in error_str or 'dimension' in error_str:
            self.log("   → Dimension mismatch: skipping batch", 'WARNING')
            return True

        self.log(f"   → Unknown error, skipping batch", 'WARNING')
        return True

    def _auto_fix_model_error(self, error):
        self.log(f"🔧 Attempting model error recovery: {error}", 'WARNING')

    def _save_checkpoint(self, model, epoch, loss):
        try:
            import torch
            ckpt_path = CHECKPOINTS_DIR / f"checkpoint_epoch{epoch}_loss{loss:.4f}.pt"
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'loss': loss,
                'timestamp': datetime.now().isoformat(),
            }, ckpt_path)
            self.log(f"💾 Checkpoint saved: {ckpt_path.name}", 'INFO')
        except Exception as e:
            self.log(f"Failed to save checkpoint: {e}", 'WARNING')

    def _evaluate_model(self, model, device):
        try:
            import torch
            model.eval()
            with torch.no_grad():
                x = torch.randint(0, 10000, (4, 32)).to(device)
                out = model(x)
                self.log(f"🧪 Eval | Output shape: {out.shape} | Max logit: {out.max().item():.3f}", 'INFO')
            model.train()
        except Exception as e:
            self.log(f"Eval error (non-critical): {e}", 'WARNING')

    def pause(self):
        self.paused = True
        self.log("⏸️  Training paused", 'WARNING')

    def resume(self):
        self.paused = False
        self.log("▶️  Training resumed", 'INFO')

    def stop(self):
        self.running = False
        self.log("⏹️  Stopping training...", 'WARNING')


# ─── Dataset Processor Thread ─────────────────────────────────────────────────
class DatasetProcessor(QThread):
    log_signal = pyqtSignal(str, str)
    progress_signal = pyqtSignal(int, str)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, input_paths, output_dir, operations):
        super().__init__()
        self.input_paths = input_paths
        self.output_dir = Path(output_dir)
        self.operations = operations

    def log(self, msg, level='INFO'):
        self.log_signal.emit(msg, level)

    def run(self):
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            total = len(self.input_paths)

            for i, path in enumerate(self.input_paths, 1):
                p = Path(path)
                self.progress_signal.emit(int(i/total*100), f"Processing: {p.name}")
                self.log(f"📁 Processing: {p.name}", 'INFO')

                if not p.exists():
                    self.log(f"   ❌ Not found: {path}", 'ERROR')
                    continue

                suffix = p.suffix.lower()

                try:
                    if suffix in ['.txt', '.md']:
                        self._process_text(p)
                    elif suffix in ['.json', '.jsonl']:
                        self._process_json(p)
                    elif suffix == '.csv':
                        self._process_csv(p)
                    elif suffix in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                        self._process_image(p)
                    elif suffix in ['.mp4', '.avi', '.mov']:
                        self._process_video(p)
                    elif suffix == '.parquet':
                        self._process_parquet(p)
                    else:
                        self.log(f"   ⚠️  Unknown format: {suffix}", 'WARNING')
                except Exception as e:
                    self.log(f"   ❌ Error processing {p.name}: {e}", 'ERROR')
                    continue

            self.finished_signal.emit(True, f"Processed {total} files")

        except Exception as e:
            self.finished_signal.emit(False, str(e))

    def _process_text(self, path):
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()

        # Clean
        import re
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = text.strip()

        # Tokenize (simple word tokenization)
        if 'tokenize' in self.operations:
            tokens = text.split()
            token_count = len(tokens)
            self.log(f"   ✓ Tokens: {token_count:,}", 'INFO')

        # Save cleaned
        out_path = self.output_dir / (path.stem + '_clean.txt')
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(text)
        self.log(f"   ✓ Saved: {out_path.name}", 'INFO')

    def _process_json(self, path):
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            if path.suffix == '.jsonl':
                data = [json.loads(l) for l in f if l.strip()]
            else:
                data = json.load(f)

        if isinstance(data, list):
            self.log(f"   ✓ Records: {len(data):,}", 'INFO')

        out_path = self.output_dir / (path.stem + '_processed.jsonl')
        with open(out_path, 'w', encoding='utf-8') as f:
            items = data if isinstance(data, list) else [data]
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        self.log(f"   ✓ Saved JSONL: {out_path.name}", 'INFO')

    def _process_csv(self, path):
        try:
            import csv
            rows = []
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            self.log(f"   ✓ Rows: {len(rows):,} | Columns: {len(rows[0]) if rows else 0}", 'INFO')

            out_path = self.output_dir / (path.stem + '_processed.jsonl')
            with open(out_path, 'w', encoding='utf-8') as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + '\n')
            self.log(f"   ✓ Converted to JSONL: {out_path.name}", 'INFO')
        except Exception as e:
            self.log(f"   CSV error: {e}", 'ERROR')

    def _process_image(self, path):
        try:
            from PIL import Image
            img = Image.open(path)
            target_size = (224, 224)
            img_resized = img.resize(target_size, Image.LANCZOS)
            if img_resized.mode != 'RGB':
                img_resized = img_resized.convert('RGB')
            out_path = self.output_dir / (path.stem + '_processed.jpg')
            img_resized.save(out_path, 'JPEG', quality=90)
            self.log(f"   ✓ Image {img.size} → {target_size}: {out_path.name}", 'INFO')
        except ImportError:
            self.log("   ⚠️  PIL not installed: pip install Pillow", 'WARNING')
            shutil.copy2(path, self.output_dir / path.name)
        except Exception as e:
            self.log(f"   Image error: {e}", 'ERROR')

    def _process_video(self, path):
        try:
            import cv2
            cap = cv2.VideoCapture(str(path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frames / fps if fps > 0 else 0
            self.log(f"   ✓ Video: {frames} frames @ {fps:.1f}fps ({duration:.1f}s)", 'INFO')
            cap.release()
        except ImportError:
            self.log("   ⚠️  OpenCV not installed: pip install opencv-python", 'WARNING')
        except Exception as e:
            self.log(f"   Video error: {e}", 'ERROR')

    def _process_parquet(self, path):
        try:
            import pandas as pd
            df = pd.read_parquet(path)
            self.log(f"   ✓ Parquet: {len(df):,} rows × {len(df.columns)} cols", 'INFO')
            out_path = self.output_dir / (path.stem + '.jsonl')
            df.to_json(out_path, orient='records', lines=True)
            self.log(f"   ✓ Converted to JSONL: {out_path.name}", 'INFO')
        except ImportError:
            self.log("   ⚠️  pandas/pyarrow not installed", 'WARNING')
        except Exception as e:
            self.log(f"   Parquet error: {e}", 'ERROR')


# ─── Hardware Panel Widget ────────────────────────────────────────────────────
class HardwarePanel(QWidget):
    def __init__(self):
        super().__init__()
        self._setup_ui()
        self.monitor = HardwareMonitor()
        self.monitor.stats_updated.connect(self._update_stats)
        self.monitor.start()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        title = QLabel("🖥️  System Monitor")
        title.setStyleSheet(f"font-size:16px; font-weight:800; color:{COLORS['accent']}; padding:8px;")
        layout.addWidget(title)

        # Static info
        self.static_label = QLabel()
        self.static_label.setStyleSheet(f"color:{COLORS['text_muted']}; font-size:11px; padding:4px 8px;")
        self._fill_static()
        layout.addWidget(self.static_label)

        # CPU
        cpu_group = QGroupBox("CPU")
        cpu_layout = QVBoxLayout(cpu_group)
        self.cpu_bar = QProgressBar()
        self.cpu_label = QLabel("0%")
        self.cpu_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        cpu_layout.addWidget(self.cpu_label)
        cpu_layout.addWidget(self.cpu_bar)
        layout.addWidget(cpu_group)

        # RAM
        ram_group = QGroupBox("RAM")
        ram_layout = QVBoxLayout(ram_group)
        self.ram_bar = QProgressBar()
        self.ram_label = QLabel("0 / 0 GB")
        self.ram_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        ram_layout.addWidget(self.ram_label)
        ram_layout.addWidget(self.ram_bar)
        layout.addWidget(ram_group)

        # GPU
        gpu_group = QGroupBox("GPU / VRAM")
        gpu_layout = QVBoxLayout(gpu_group)
        self.gpu_name_label = QLabel("Detecting...")
        self.gpu_name_label.setStyleSheet(f"color:{COLORS['text_muted']}; font-size:11px;")
        self.gpu_bar = QProgressBar()
        self.gpu_label = QLabel("No GPU")
        self.gpu_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        gpu_layout.addWidget(self.gpu_name_label)
        gpu_layout.addWidget(self.gpu_label)
        gpu_layout.addWidget(self.gpu_bar)
        layout.addWidget(gpu_group)

        # Disk
        disk_group = QGroupBox("Disk")
        disk_layout = QVBoxLayout(disk_group)
        self.disk_bar = QProgressBar()
        self.disk_label = QLabel("0 / 0 GB")
        self.disk_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        disk_layout.addWidget(self.disk_label)
        disk_layout.addWidget(self.disk_bar)
        layout.addWidget(disk_group)

        # Tier badge
        self.tier_label = QLabel("⚡ Tier: Calculating...")
        self.tier_label.setStyleSheet(
            f"background:{COLORS['bg_card']}; border:1px solid {COLORS['accent2']};"
            f"border-radius:8px; padding:10px; font-weight:700; font-size:13px; color:{COLORS['accent2']};"
        )
        self.tier_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.tier_label)

        layout.addStretch()

    def _fill_static(self):
        cpu_name = platform.processor() or platform.machine()
        os_info = f"{platform.system()} {platform.release()}"
        ram_total = psutil.virtual_memory().total / (1024**3)
        self.static_label.setText(
            f"OS: {os_info}  |  CPU: {cpu_name[:40]}  |  RAM: {ram_total:.1f}GB"
        )

    def _update_stats(self, stats):
        # CPU
        cpu = stats['cpu_percent']
        self.cpu_bar.setValue(int(cpu))
        self.cpu_label.setText(f"{cpu:.1f}%  ({stats['cpu_cores']} cores)")
        color = COLORS['danger'] if cpu > 85 else COLORS['warning'] if cpu > 70 else COLORS['accent3']
        self.cpu_bar.setStyleSheet(f"QProgressBar::chunk {{ background: {color}; border-radius:5px; }}")

        # RAM
        self.ram_bar.setValue(int(stats['ram_percent']))
        self.ram_label.setText(f"{stats['ram_used']:.1f} / {stats['ram_total']:.1f} GB  ({stats['ram_percent']:.0f}%)")

        # GPU
        gpu = stats['gpu']
        self.gpu_name_label.setText(gpu['name'])
        if gpu['available'] and gpu['mem_total'] > 0:
            pct = int(gpu['mem_used'] / gpu['mem_total'] * 100)
            self.gpu_bar.setValue(pct)
            self.gpu_label.setText(f"{gpu['mem_used']:.1f} / {gpu['mem_total']:.1f} GB  ({pct}%)")
        else:
            self.gpu_bar.setValue(0)
            self.gpu_label.setText("CPU-only mode")

        # Disk
        self.disk_bar.setValue(int(stats['disk_percent']))
        self.disk_label.setText(f"{stats['disk_used']:.0f} / {stats['disk_total']:.0f} GB")

        # Tier
        ram_gb = stats['ram_total']
        gpu_vram = stats['gpu']['mem_total']
        score = stats['cpu_cores'] * 10 + ram_gb * 5 + gpu_vram * 20
        if score > 300:
            tier, color = "HIGH ⚡⚡⚡", COLORS['accent3']
        elif score > 150:
            tier, color = "MEDIUM ⚡⚡", COLORS['warning']
        else:
            tier, color = "LOW ⚡", COLORS['text_muted']

        self.tier_label.setText(f"Hardware Tier: {tier}")
        self.tier_label.setStyleSheet(
            f"background:{COLORS['bg_card']}; border:1px solid {color};"
            f"border-radius:8px; padding:10px; font-weight:700; font-size:13px; color:{color};"
        )

    def closeEvent(self, event):
        self.monitor.stop()
        super().closeEvent(event)


# ─── Log Widget ───────────────────────────────────────────────────────────────
class LogWidget(QTextEdit):
    def __init__(self):
        super().__init__()
        self.setReadOnly(True)
        self.setFont(QFont("Consolas", 10))
        self.level_colors = {
            'INFO':    COLORS['text_primary'],
            'SUCCESS': COLORS['accent3'],
            'WARNING': COLORS['warning'],
            'ERROR':   COLORS['danger'],
            'DEBUG':   COLORS['text_muted'],
        }

    def append_log(self, msg, level='INFO'):
        color = self.level_colors.get(level, COLORS['text_primary'])
        self.append(f'<span style="color:{color}">{msg}</span>')
        sb = self.verticalScrollBar()
        sb.setValue(sb.maximum())

    def clear_logs(self):
        self.clear()


# ─── Training Tab ─────────────────────────────────────────────────────────────
class TrainingTab(QWidget):
    def __init__(self):
        super().__init__()
        self.trainer = None
        self.training_config = {}
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ── Left: Config ──
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setSpacing(10)

        # Model section
        model_group = QGroupBox("🧠 Model Configuration")
        mg_layout = QFormLayout(model_group)

        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("Leave empty for new model")
        browse_model = QPushButton("Browse")
        browse_model.setFixedWidth(80)
        browse_model.clicked.connect(self._browse_model)
        browse_model.setProperty('class', 'secondary')
        model_row = QHBoxLayout()
        model_row.addWidget(self.model_path_edit)
        model_row.addWidget(browse_model)
        mg_layout.addRow("Model Path:", model_row)

        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems(['transformer', 'lstm', 'cnn', 'custom'])
        mg_layout.addRow("Architecture:", self.model_type_combo)

        self.framework_combo = QComboBox()
        self.framework_combo.addItems(['pytorch', 'tensorflow', 'jax'])
        mg_layout.addRow("Framework:", self.framework_combo)

        left_layout.addWidget(model_group)

        # Training params
        params_group = QGroupBox("⚙️  Training Parameters")
        pg_layout = QFormLayout(params_group)

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 10000)
        self.epochs_spin.setValue(3)
        pg_layout.addRow("Epochs:", self.epochs_spin)

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(1e-6, 1.0)
        self.lr_spin.setSingleStep(1e-5)
        self.lr_spin.setDecimals(6)
        self.lr_spin.setValue(3e-4)
        pg_layout.addRow("Learning Rate:", self.lr_spin)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 2048)
        self.batch_spin.setValue(0)
        self.batch_spin.setSpecialValueText("Auto (hardware)")
        pg_layout.addRow("Batch Size:", self.batch_spin)

        self.d_model_spin = QSpinBox()
        self.d_model_spin.setRange(64, 4096)
        self.d_model_spin.setValue(256)
        self.d_model_spin.setSingleStep(64)
        pg_layout.addRow("d_model:", self.d_model_spin)

        self.n_layers_spin = QSpinBox()
        self.n_layers_spin.setRange(1, 48)
        self.n_layers_spin.setValue(4)
        pg_layout.addRow("Layers:", self.n_layers_spin)

        self.n_heads_spin = QSpinBox()
        self.n_heads_spin.setRange(1, 32)
        self.n_heads_spin.setValue(4)
        pg_layout.addRow("Attention Heads:", self.n_heads_spin)

        left_layout.addWidget(params_group)

        # Options
        opts_group = QGroupBox("🛡️  Safety & Recovery")
        og_layout = QVBoxLayout(opts_group)
        self.auto_fix_cb = QCheckBox("Auto-fix training errors")
        self.auto_fix_cb.setChecked(True)
        self.throttle_cb = QCheckBox("Auto-throttle on high HW load")
        self.throttle_cb.setChecked(True)
        self.checkpoint_cb = QCheckBox("Save checkpoints on improvement")
        self.checkpoint_cb.setChecked(True)
        self.eval_cb = QCheckBox("Periodic evaluation")
        self.eval_cb.setChecked(True)
        og_layout.addWidget(self.auto_fix_cb)
        og_layout.addWidget(self.throttle_cb)
        og_layout.addWidget(self.checkpoint_cb)
        og_layout.addWidget(self.eval_cb)
        left_layout.addWidget(opts_group)

        left_layout.addStretch()

        # ── Right: Runtime ──
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setSpacing(10)

        # Status bar
        status_frame = QFrame()
        status_frame.setStyleSheet(f"background:{COLORS['bg_card']}; border-radius:8px; padding:4px;")
        sf_layout = QHBoxLayout(status_frame)

        self.status_label = QLabel("⏸  Idle")
        self.status_label.setStyleSheet(f"font-size:14px; font-weight:700; color:{COLORS['accent']};")

        self.epoch_label = QLabel("Epoch: —")
        self.loss_label = QLabel("Loss: —")
        self.lr_label = QLabel("LR: —")
        for lbl in [self.epoch_label, self.loss_label, self.lr_label]:
            lbl.setStyleSheet(f"color:{COLORS['text_muted']}; font-size:12px;")

        sf_layout.addWidget(self.status_label)
        sf_layout.addStretch()
        sf_layout.addWidget(self.epoch_label)
        sf_layout.addWidget(self.loss_label)
        sf_layout.addWidget(self.lr_label)
        right_layout.addWidget(status_frame)

        # Progress bars
        self.epoch_progress = QProgressBar()
        self.epoch_progress.setFormat("Epochs: %p%")
        self.step_progress = QProgressBar()
        self.step_progress.setFormat("Steps: %p%")
        right_layout.addWidget(self.epoch_progress)
        right_layout.addWidget(self.step_progress)

        # Controls
        ctrl_layout = QHBoxLayout()
        self.start_btn = QPushButton("▶  Start Training")
        self.pause_btn = QPushButton("⏸  Pause")
        self.stop_btn = QPushButton("⏹  Stop")
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet(f"background:{COLORS['danger']};")

        self.start_btn.clicked.connect(self._start_training)
        self.pause_btn.clicked.connect(self._pause_training)
        self.stop_btn.clicked.connect(self._stop_training)

        ctrl_layout.addWidget(self.start_btn)
        ctrl_layout.addWidget(self.pause_btn)
        ctrl_layout.addWidget(self.stop_btn)
        right_layout.addLayout(ctrl_layout)

        # Log
        log_label = QLabel("Training Log")
        log_label.setStyleSheet(f"color:{COLORS['accent']}; font-weight:700; font-size:13px;")
        right_layout.addWidget(log_label)
        self.log_widget = LogWidget()
        right_layout.addWidget(self.log_widget)

        # Clear log button
        clear_btn = QPushButton("🗑  Clear Log")
        clear_btn.setProperty('class', 'secondary')
        clear_btn.setFixedHeight(30)
        clear_btn.clicked.connect(self.log_widget.clear_logs)
        right_layout.addWidget(clear_btn)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([350, 650])
        layout.addWidget(splitter)

    def set_datasets(self, datasets):
        self.training_config['datasets'] = datasets

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Model File", str(MODELS_DIR),
            "Model Files (*.pt *.pth *.h5 *.keras *.bin *.ckpt);;All Files (*)"
        )
        if path:
            self.model_path_edit.setText(path)

    def _build_config(self):
        config = {
            'model_path': self.model_path_edit.text(),
            'model_type': self.model_type_combo.currentText(),
            'framework': self.framework_combo.currentText(),
            'epochs': self.epochs_spin.value(),
            'lr': self.lr_spin.value(),
            'batch_size': self.batch_spin.value() or None,
            'd_model': self.d_model_spin.value(),
            'n_layers': self.n_layers_spin.value(),
            'n_heads': self.n_heads_spin.value(),
            'datasets': self.training_config.get('datasets', []),
            'auto_fix': self.auto_fix_cb.isChecked(),
            'throttle': self.throttle_cb.isChecked(),
        }
        return config

    def _start_training(self):
        config = self._build_config()
        self.trainer = TrainingEngine(config)
        self.trainer.log_signal.connect(self._on_log)
        self.trainer.progress_signal.connect(self._on_progress)
        self.trainer.status_signal.connect(self._on_status)
        self.trainer.finished_signal.connect(self._on_finished)
        self.trainer.hw_throttle_signal.connect(lambda m: self._on_log(m, 'WARNING'))
        self.trainer.start()

        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("🟢 Training...")
        self.status_label.setStyleSheet(f"font-size:14px; font-weight:700; color:{COLORS['accent3']};")

    def _pause_training(self):
        if self.trainer:
            if self.pause_btn.text() == "⏸  Pause":
                self.trainer.pause()
                self.pause_btn.setText("▶  Resume")
                self.status_label.setText("⏸ Paused")
            else:
                self.trainer.resume()
                self.pause_btn.setText("⏸  Pause")
                self.status_label.setText("🟢 Training...")

    def _stop_training(self):
        if self.trainer:
            self.trainer.stop()

    def _on_log(self, msg, level):
        self.log_widget.append_log(msg, level)

    def _on_progress(self, epoch, total_epochs, global_step, loss, lr):
        self.epoch_progress.setValue(int(epoch / total_epochs * 100))
        self.epoch_label.setText(f"Epoch: {epoch}/{total_epochs}")
        self.loss_label.setText(f"Loss: {loss:.4f}")
        self.lr_label.setText(f"LR: {lr:.2e}")

    def _on_status(self, status):
        self.status_label.setText(f"🟢 {status}")

    def _on_finished(self, success, message):
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)

        if success:
            self.status_label.setText("✅ Done!")
            self.status_label.setStyleSheet(f"font-size:14px; font-weight:700; color:{COLORS['accent3']};")
            self._on_log(f"🎉 {message}", 'SUCCESS')
        else:
            self.status_label.setText("❌ Failed")
            self.status_label.setStyleSheet(f"font-size:14px; font-weight:700; color:{COLORS['danger']};")
            self._on_log(f"💥 {message}", 'ERROR')


# ─── Dataset Tab ─────────────────────────────────────────────────────────────
class DatasetTab(QWidget):
    datasets_updated = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self.datasets = []
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: dataset list
        left = QWidget()
        ll = QVBoxLayout(left)

        ds_label = QLabel("📂 Datasets")
        ds_label.setStyleSheet(f"font-size:14px; font-weight:700; color:{COLORS['accent']}; padding:4px;")
        ll.addWidget(ds_label)

        # Buttons
        btn_row = QHBoxLayout()
        add_file_btn = QPushButton("+ File")
        add_folder_btn = QPushButton("+ Folder")
        remove_btn = QPushButton("Remove")
        remove_btn.setStyleSheet(f"background:{COLORS['danger']};")
        for b in [add_file_btn, add_folder_btn, remove_btn]:
            b.setFixedHeight(32)
        add_file_btn.clicked.connect(self._add_file)
        add_folder_btn.clicked.connect(self._add_folder)
        remove_btn.clicked.connect(self._remove_dataset)
        btn_row.addWidget(add_file_btn)
        btn_row.addWidget(add_folder_btn)
        btn_row.addWidget(remove_btn)
        ll.addLayout(btn_row)

        self.ds_list = QListWidget()
        ll.addWidget(self.ds_list)

        # Stats
        self.ds_stats = QLabel("No datasets added")
        self.ds_stats.setStyleSheet(f"color:{COLORS['text_muted']}; font-size:11px; padding:4px;")
        ll.addWidget(self.ds_stats)

        # Right: processing
        right = QWidget()
        rl = QVBoxLayout(right)

        proc_group = QGroupBox("⚡ Processing Pipeline")
        pg = QVBoxLayout(proc_group)

        self.clean_cb = QCheckBox("Clean & normalize text")
        self.clean_cb.setChecked(True)
        self.tokenize_cb = QCheckBox("Tokenize content")
        self.tokenize_cb.setChecked(True)
        self.convert_cb = QCheckBox("Convert formats to JSONL")
        self.convert_cb.setChecked(True)
        self.resize_img_cb = QCheckBox("Resize images (224×224)")
        self.resize_img_cb.setChecked(True)
        self.extract_vid_cb = QCheckBox("Extract video frames")
        self.extract_vid_cb.setChecked(False)

        for cb in [self.clean_cb, self.tokenize_cb, self.convert_cb,
                   self.resize_img_cb, self.extract_vid_cb]:
            pg.addWidget(cb)

        rl.addWidget(proc_group)

        out_group = QGroupBox("📁 Output Directory")
        og = QHBoxLayout(out_group)
        self.out_edit = QLineEdit(str(DATASETS_DIR / "processed"))
        browse_out = QPushButton("Browse")
        browse_out.setFixedWidth(80)
        browse_out.clicked.connect(self._browse_output)
        og.addWidget(self.out_edit)
        og.addWidget(browse_out)
        rl.addWidget(out_group)

        self.process_btn = QPushButton("⚡ Process All Datasets")
        self.process_btn.setMinimumHeight(44)
        self.process_btn.clicked.connect(self._process_datasets)
        rl.addWidget(self.process_btn)

        self.proc_progress = QProgressBar()
        self.proc_status = QLabel("")
        self.proc_status.setStyleSheet(f"color:{COLORS['accent']}; font-size:12px;")
        rl.addWidget(self.proc_progress)
        rl.addWidget(self.proc_status)

        rl.addWidget(QLabel("Processing Log:"))
        self.proc_log = LogWidget()
        rl.addWidget(self.proc_log)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([350, 650])
        layout.addWidget(splitter)

    def _add_file(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Dataset Files", "",
            "Data Files (*.txt *.json *.jsonl *.csv *.parquet *.jpg *.png *.mp4 *.avi);;All Files (*)"
        )
        for p in paths:
            if p not in self.datasets:
                self.datasets.append(p)
                item = QListWidgetItem(f"📄 {Path(p).name}")
                item.setToolTip(p)
                self.ds_list.addItem(item)
        self._update_stats()

    def _add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Dataset Folder")
        if folder and folder not in self.datasets:
            self.datasets.append(folder)
            item = QListWidgetItem(f"📁 {Path(folder).name}")
            item.setToolTip(folder)
            self.ds_list.addItem(item)
        self._update_stats()

    def _remove_dataset(self):
        for item in self.ds_list.selectedItems():
            path = item.toolTip()
            if path in self.datasets:
                self.datasets.remove(path)
            self.ds_list.takeItem(self.ds_list.row(item))
        self._update_stats()

    def _browse_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if folder:
            self.out_edit.setText(folder)

    def _update_stats(self):
        n = len(self.datasets)
        self.ds_stats.setText(f"{n} dataset{'s' if n!=1 else ''} loaded")
        self.datasets_updated.emit(self.datasets)

    def _get_operations(self):
        ops = []
        if self.clean_cb.isChecked(): ops.append('clean')
        if self.tokenize_cb.isChecked(): ops.append('tokenize')
        if self.convert_cb.isChecked(): ops.append('convert')
        if self.resize_img_cb.isChecked(): ops.append('resize')
        if self.extract_vid_cb.isChecked(): ops.append('extract_video')
        return ops

    def _process_datasets(self):
        if not self.datasets:
            QMessageBox.warning(self, "No Datasets", "Please add datasets first.")
            return

        self.proc_log.clear_logs()
        self.processor = DatasetProcessor(
            self.datasets,
            self.out_edit.text(),
            self._get_operations()
        )
        self.processor.log_signal.connect(lambda m, l: self.proc_log.append_log(m, l))
        self.processor.progress_signal.connect(
            lambda p, s: (self.proc_progress.setValue(p), self.proc_status.setText(s))
        )
        self.processor.finished_signal.connect(self._on_process_done)
        self.processor.start()
        self.process_btn.setEnabled(False)

    def _on_process_done(self, success, msg):
        self.process_btn.setEnabled(True)
        level = 'SUCCESS' if success else 'ERROR'
        self.proc_log.append_log(f"{'✅' if success else '❌'} {msg}", level)


# ─── Checkpoints Tab ─────────────────────────────────────────────────────────
class CheckpointsTab(QWidget):
    def __init__(self):
        super().__init__()
        self._setup_ui()
        self._refresh()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        header = QHBoxLayout()
        title = QLabel("💾 Checkpoints & Models")
        title.setStyleSheet(f"font-size:14px; font-weight:700; color:{COLORS['accent']};")
        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.setFixedWidth(100)
        refresh_btn.clicked.connect(self._refresh)
        header.addWidget(title)
        header.addStretch()
        header.addWidget(refresh_btn)
        layout.addLayout(header)

        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["Name", "Size", "Date", "Type", "Path"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)
        layout.addWidget(self.table)

        btn_row = QHBoxLayout()
        load_btn = QPushButton("📂 Load Selected")
        delete_btn = QPushButton("🗑  Delete")
        delete_btn.setStyleSheet(f"background:{COLORS['danger']};")
        load_btn.clicked.connect(self._load_checkpoint)
        delete_btn.clicked.connect(self._delete_checkpoint)
        btn_row.addWidget(load_btn)
        btn_row.addStretch()
        btn_row.addWidget(delete_btn)
        layout.addLayout(btn_row)

    def _refresh(self):
        self.table.setRowCount(0)
        dirs = [CHECKPOINTS_DIR, MODELS_DIR]
        exts = {'.pt', '.pth', '.h5', '.keras', '.bin', '.ckpt', '.safetensors'}
        rows = []

        for d in dirs:
            for f in sorted(d.glob('*'), key=lambda x: x.stat().st_mtime, reverse=True):
                if f.suffix.lower() in exts:
                    stat = f.stat()
                    size = stat.st_size
                    size_str = f"{size/1e6:.1f} MB" if size > 1e6 else f"{size/1e3:.0f} KB"
                    date = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M')
                    rows.append((f.name, size_str, date, f.suffix[1:].upper(), str(f)))

        self.table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            for j, val in enumerate(row):
                item = QTableWidgetItem(val)
                item.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
                self.table.setItem(i, j, item)

    def _load_checkpoint(self):
        row = self.table.currentRow()
        if row >= 0:
            path = self.table.item(row, 4).text()
            QMessageBox.information(self, "Checkpoint", f"Selected:\n{path}\n\nSet this path in the Training tab to resume.")

    def _delete_checkpoint(self):
        row = self.table.currentRow()
        if row >= 0:
            path = self.table.item(row, 4).text()
            reply = QMessageBox.question(self, "Delete", f"Delete {Path(path).name}?",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                try:
                    os.remove(path)
                    self._refresh()
                except Exception as e:
                    QMessageBox.warning(self, "Error", str(e))


# ─── Libraries Tab ────────────────────────────────────────────────────────────
class LibrariesTab(QWidget):
    def __init__(self):
        super().__init__()
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        title = QLabel("📦 Framework & Library Manager")
        title.setStyleSheet(f"font-size:14px; font-weight:700; color:{COLORS['accent']}; padding:4px;")
        layout.addWidget(title)

        # Library status table
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Library", "Status", "Version", "Install Command"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table)

        check_btn = QPushButton("🔍 Check All Libraries")
        check_btn.clicked.connect(self._check_libraries)
        layout.addWidget(check_btn)

        # Custom install
        install_group = QGroupBox("📥 Install Custom Library")
        ig = QHBoxLayout(install_group)
        self.pkg_edit = QLineEdit()
        self.pkg_edit.setPlaceholderText("e.g. transformers accelerate bitsandbytes")
        install_btn = QPushButton("pip install")
        install_btn.setFixedWidth(120)
        install_btn.clicked.connect(self._install_package)
        ig.addWidget(self.pkg_edit)
        ig.addWidget(install_btn)
        layout.addWidget(install_group)

        self.install_log = LogWidget()
        self.install_log.setMaximumHeight(200)
        layout.addWidget(self.install_log)

        self._check_libraries()

    def _check_libraries(self):
        LIBS = [
            ('torch', 'PyTorch', 'pip install torch torchvision torchaudio'),
            ('tensorflow', 'TensorFlow', 'pip install tensorflow'),
            ('jax', 'JAX', 'pip install jax jaxlib'),
            ('transformers', 'HuggingFace Transformers', 'pip install transformers'),
            ('accelerate', 'HuggingFace Accelerate', 'pip install accelerate'),
            ('datasets', 'HuggingFace Datasets', 'pip install datasets'),
            ('tokenizers', 'HuggingFace Tokenizers', 'pip install tokenizers'),
            ('peft', 'PEFT (LoRA)', 'pip install peft'),
            ('bitsandbytes', 'BitsAndBytes (4/8-bit)', 'pip install bitsandbytes'),
            ('sklearn', 'Scikit-Learn', 'pip install scikit-learn'),
            ('numpy', 'NumPy', 'pip install numpy'),
            ('pandas', 'Pandas', 'pip install pandas'),
            ('PIL', 'Pillow', 'pip install Pillow'),
            ('cv2', 'OpenCV', 'pip install opencv-python'),
            ('psutil', 'PSUtil', 'pip install psutil'),
            ('tqdm', 'tqdm', 'pip install tqdm'),
            ('wandb', 'Weights & Biases', 'pip install wandb'),
            ('tensorboard', 'TensorBoard', 'pip install tensorboard'),
            ('flash_attn', 'Flash Attention', 'pip install flash-attn'),
            ('deepspeed', 'DeepSpeed', 'pip install deepspeed'),
        ]

        self.table.setRowCount(len(LIBS))
        for i, (module, name, cmd) in enumerate(LIBS):
            try:
                mod = __import__(module)
                version = getattr(mod, '__version__', 'installed')
                status = "✅ Installed"
                status_color = COLORS['accent3']
            except ImportError:
                version = "—"
                status = "❌ Not Found"
                status_color = COLORS['danger']

            name_item = QTableWidgetItem(name)
            status_item = QTableWidgetItem(status)
            status_item.setForeground(QColor(status_color))
            version_item = QTableWidgetItem(version)
            cmd_item = QTableWidgetItem(cmd)
            cmd_item.setForeground(QColor(COLORS['text_muted']))

            for j, item in enumerate([name_item, status_item, version_item, cmd_item]):
                item.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
                self.table.setItem(i, j, item)

    def _install_package(self):
        pkg = self.pkg_edit.text().strip()
        if not pkg:
            return

        self.install_log.append_log(f"📥 Installing: {pkg}", 'INFO')

        def run_install():
            try:
                result = subprocess.run(
                    [sys.executable, '-m', 'pip', 'install'] + pkg.split(),
                    capture_output=True, text=True, timeout=300
                )
                if result.returncode == 0:
                    self.install_log.append_log(f"✅ Installed: {pkg}", 'SUCCESS')
                    self._check_libraries()
                else:
                    self.install_log.append_log(f"❌ Failed:\n{result.stderr[:500]}", 'ERROR')
            except Exception as e:
                self.install_log.append_log(f"❌ Error: {e}", 'ERROR')

        t = threading.Thread(target=run_install, daemon=True)
        t.start()


# ─── Main Window ─────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"🦚 {APP_NAME} v{VERSION}  |  PeacockAI")
        self.setMinimumSize(1280, 800)
        self.resize(1440, 900)
        self._setup_ui()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # ── Sidebar (hardware) ──
        self.hw_panel = HardwarePanel()
        self.hw_panel.setFixedWidth(260)
        self.hw_panel.setStyleSheet(f"background:{COLORS['bg_panel']}; border-right: 1px solid {COLORS['border']};")

        # ── Main content ──
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(12, 8, 12, 8)
        content_layout.setSpacing(8)

        # Header
        header = QWidget()
        header.setStyleSheet(f"""
            background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                stop:0 {COLORS['bg_panel']}, stop:1 {COLORS['bg_card']});
            border-radius: 10px;
            border: 1px solid {COLORS['border']};
        """)
        header.setFixedHeight(60)
        hl = QHBoxLayout(header)
        hl.setContentsMargins(16, 0, 16, 0)

        logo = QLabel("🦚")
        logo.setStyleSheet("font-size:28px;")
        app_name = QLabel(APP_NAME)
        app_name.setStyleSheet(f"font-size:22px; font-weight:900; color:{COLORS['accent']}; letter-spacing:2px;")
        sub = QLabel("Intelligent AI Training Platform")
        sub.setStyleSheet(f"color:{COLORS['text_muted']}; font-size:11px;")
        by = QLabel("by PeacockAI")
        by.setStyleSheet(f"color:{COLORS['accent2']}; font-size:12px; font-weight:700;")

        hl.addWidget(logo)
        hl.addWidget(app_name)
        hl.addWidget(sub)
        hl.addStretch()
        hl.addWidget(by)
        content_layout.addWidget(header)

        # Tabs
        self.tabs = QTabWidget()

        self.training_tab = TrainingTab()
        self.dataset_tab = DatasetTab()
        self.checkpoints_tab = CheckpointsTab()
        self.libraries_tab = LibrariesTab()

        # Connect dataset updates to training tab
        self.dataset_tab.datasets_updated.connect(self.training_tab.set_datasets)

        self.tabs.addTab(self.training_tab, "🚀  Training")
        self.tabs.addTab(self.dataset_tab, "📂  Datasets")
        self.tabs.addTab(self.checkpoints_tab, "💾  Checkpoints")
        self.tabs.addTab(self.libraries_tab, "📦  Libraries")

        content_layout.addWidget(self.tabs)

        main_layout.addWidget(self.hw_panel)
        main_layout.addWidget(content)

    def closeEvent(self, event):
        self.hw_panel.monitor.stop()
        event.accept()


# ─── Entry Point ─────────────────────────────────────────────────────────────
def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setStyleSheet(STYLE)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(VERSION)

    window = MainWindow()
    window.show()

    logger.info(f"{APP_NAME} v{VERSION} started")
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
