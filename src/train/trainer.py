"""Training script for video dialogue systems."""

import argparse
import os
import time
from typing import Dict, Optional
import yaml

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from src.models.video_dialogue import create_video_dialogue_model
from src.data.dataset import VideoDialogueDataModule, create_sample_dataset
from src.eval.metrics import VideoDialogueEvaluator, VideoDialogueMetrics
from src.utils.device import get_device, set_seed, print_device_info


class VideoDialogueTrainer:
    """Trainer for video dialogue systems."""
    
    def __init__(self, config: Dict):
        """Initialize trainer.
        
        Args:
            config: Training configuration.
        """
        self.config = config
        self.device = get_device()
        self.set_seed()
        
        # Initialize model
        self.model = create_video_dialogue_model(
            config["model"]["type"],
            config["model"]["params"]
        ).to(self.device)
        
        # Initialize data module
        self.data_module = VideoDialogueDataModule(config["data"])
        self.data_module.setup()
        
        # Initialize optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Initialize metrics
        self.evaluator = VideoDialogueEvaluator()
        self.metrics_tracker = VideoDialogueMetrics()
        
        # Training state
        self.current_epoch = 0
        self.best_metric = 0.0
        
        # Initialize wandb if enabled
        if config.get("wandb", {}).get("enabled", False):
            wandb.init(
                project=config["wandb"]["project"],
                name=config["wandb"]["name"],
                config=config
            )
    
    def set_seed(self):
        """Set random seed."""
        seed = self.config.get("seed", 42)
        set_seed(seed)
    
    def _create_optimizer(self):
        """Create optimizer."""
        optimizer_config = self.config["training"]["optimizer"]
        
        if optimizer_config["type"] == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=optimizer_config["lr"],
                weight_decay=optimizer_config.get("weight_decay", 0.0)
            )
        elif optimizer_config["type"] == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=optimizer_config["lr"],
                weight_decay=optimizer_config.get("weight_decay", 0.01)
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_config['type']}")
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        scheduler_config = self.config["training"].get("scheduler", {})
        
        if scheduler_config.get("type") == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_config.get("T_max", 100)
            )
        elif scheduler_config.get("type") == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.get("step_size", 30),
                gamma=scheduler_config.get("gamma", 0.1)
            )
        else:
            return None
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        train_loader = self.data_module.train_dataloader()
        
        epoch_metrics = {
            "train_loss": 0.0,
            "train_accuracy": 0.0
        }
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            videos = batch["videos"].to(self.device)
            questions = batch["questions"]
            answers = batch["answers"]
            
            # Forward pass
            self.optimizer.zero_grad()
            
            try:
                outputs = self.model(videos, questions[0])  # Simplified for demo
                
                # Compute loss (simplified)
                loss = self._compute_loss(outputs, answers)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.config["training"].get("grad_clip", 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config["training"]["grad_clip"]
                    )
                
                self.optimizer.step()
                
                # Update metrics
                epoch_metrics["train_loss"] += loss.item()
                
                # Update progress bar
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{self.optimizer.param_groups[0]['lr']:.6f}"
                })
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
        
        # Average metrics
        epoch_metrics["train_loss"] /= len(train_loader)
        
        return epoch_metrics
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        val_loader = self.data_module.val_dataloader()
        
        predictions = []
        references = []
        questions = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                videos = batch["videos"].to(self.device)
                batch_questions = batch["questions"]
                batch_answers = batch["answers"]
                
                try:
                    # Generate predictions
                    for i, question in enumerate(batch_questions):
                        video = videos[i:i+1]
                        response = self.model.generate_response(video, question)
                        predictions.append(response)
                        references.append(batch_answers[i])
                        questions.append(question)
                
                except Exception as e:
                    print(f"Error in validation batch: {e}")
                    continue
        
        # Compute metrics
        metrics = self.evaluator.evaluate_batch(predictions, references, questions)
        
        return metrics
    
    def _compute_loss(self, outputs: Dict, answers: list) -> torch.Tensor:
        """Compute loss (simplified for demo)."""
        # This is a simplified loss computation
        # In practice, you would use proper language modeling loss
        
        if "response_logits" in outputs:
            # Cross-entropy loss for response generation
            target_tokens = torch.randint(0, 1000, (outputs["response_logits"].size(0), 10)).to(self.device)
            loss = nn.CrossEntropyLoss()(outputs["response_logits"], target_tokens)
        else:
            # Simple MSE loss for feature matching
            loss = nn.MSELoss()(outputs["fused_features"], torch.randn_like(outputs["fused_features"]))
        
        return loss
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_metric": self.best_metric,
            "config": self.config
        }
        
        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        # Save checkpoint
        checkpoint_path = os.path.join(self.config["output_dir"], f"checkpoint_epoch_{epoch}.pt")
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.config["output_dir"], "best_model.pt")
            torch.save(checkpoint, best_path)
            print(f"Saved best model at epoch {epoch}")
    
    def train(self):
        """Main training loop."""
        print("Starting training...")
        print_device_info()
        
        num_epochs = self.config["training"]["num_epochs"]
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            start_time = time.time()
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate_epoch()
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Log metrics
            epoch_time = time.time() - start_time
            all_metrics = {
                **train_metrics,
                **val_metrics,
                "epoch_time": epoch_time
            }
            
            self.metrics_tracker.update(all_metrics)
            self.metrics_tracker.log_epoch(epoch)
            
            # Print metrics
            print(f"\nEpoch {epoch}/{num_epochs-1}")
            print(f"Train Loss: {train_metrics['train_loss']:.4f}")
            print(f"Val F1: {val_metrics.get('f1_score', 0):.4f}")
            print(f"Val BLEU-1: {val_metrics.get('bleu_1', 0):.4f}")
            print(f"Epoch Time: {epoch_time:.2f}s")
            
            # Save checkpoint
            current_metric = val_metrics.get("f1_score", 0)
            is_best = current_metric > self.best_metric
            if is_best:
                self.best_metric = current_metric
            
            if epoch % self.config["training"].get("save_every", 10) == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
            
            # Log to wandb
            if self.config.get("wandb", {}).get("enabled", False):
                wandb.log(all_metrics)
        
        print("Training completed!")
        print(f"Best F1 Score: {self.best_metric:.4f}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train video dialogue system")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--create_sample_data", action="store_true", help="Create sample dataset")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update output directory
    config["output_dir"] = args.output_dir
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create sample data if requested
    if args.create_sample_data:
        data_dir = os.path.join(args.output_dir, "data")
        create_sample_dataset(data_dir, num_samples=100)
        print(f"Created sample dataset in {data_dir}")
        return
    
    # Initialize trainer
    trainer = VideoDialogueTrainer(config)
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
