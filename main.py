import yaml
import torch
import os
import argparse

from dense_unet_3d.dataset.prepare_dataset import prepare_dataloader
from dense_unet_3d.model.DenseUNet3d import DenseUNet3d
from dense_unet_3d.evaluation.evaluate import evaluate
from dense_unet_3d.training.train import train as train_model


def train(config, device, num_classes):
    """
    Train the model on training data.
    """
    print("\n" + "="*50)
    print("TRAINING MODE")
    print("="*50)

    # Load training data
    print("\nLoading training data...")
    trainloader = prepare_dataloader(config, train=True)
    print(f"Training dataset size: {len(trainloader.dataset)} samples")
    print(f"Training batches: {len(trainloader)}")

    # Initialize model
    model = DenseUNet3d(num_classes=num_classes)

    # Check if resuming from checkpoint
    model_save_dir = config["pathing"]["model_save_dir"]
    run_name = config["pathing"]["run_name"]
    model_path = os.path.join(model_save_dir, run_name, f"latest.pt")

    if os.path.exists(model_path):
        print(f"\nLoading checkpoint from: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Checkpoint loaded! Resuming training...")
    else:
        print("\nNo checkpoint found. Starting training from scratch...")

    # Train the model
    print("\nStarting training...\n")
    losses = train_model(config, model, device, trainloader)

    print("\n" + "="*50)
    print("Training complete!")
    print(f"Final loss: {losses[-1]:.4f}")
    print("="*50)

    # Optionally evaluate on training data
    print("\nEvaluating on training data...")
    liver_dice = evaluate(model, device, trainloader, dim=1)
    print(f"Training Liver Dice Score: {liver_dice:.4f}")

    return {"losses": losses, "liver_dice": liver_dice}


def test(config, device, num_classes):
    """
    Run inference and evaluation on test data.
    """
    print("\n" + "="*50)
    print("TEST/INFERENCE MODE")
    print("="*50)

    # Load test data
    print("\nLoading test data...")
    testloader = prepare_dataloader(config, train=False)
    print(f"Test dataset size: {len(testloader.dataset)} samples")
    print(f"Test batches: {len(testloader)}")

    # Initialize model
    model = DenseUNet3d(num_classes=num_classes)

    # Check if a saved model exists and load it
    model_save_dir = config["pathing"]["model_save_dir"]
    run_name = config["pathing"]["run_name"]
    model_path = os.path.join(model_save_dir, run_name, f"latest.pt")

    if os.path.exists(model_path):
        print(f"\nLoading pre-trained model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded successfully!")
    else:
        print(f"\nNo pre-trained model found at: {model_path}")
        print("Using untrained model (random weights) for testing pipeline...")
        print("Note: Results will not be meaningful without a trained model.")

    model = model.to(device)

    # Run evaluation
    print("\nRunning evaluation on test data...")
    model.eval()

    with torch.no_grad():
        # Evaluate on liver segmentation (dim=1)
        print("Evaluating liver segmentation...")
        liver_dice_score = evaluate(model, device, testloader, dim=1)
        print(f"Liver Dice Score: {liver_dice_score:.4f}")

        # Evaluate on tumor segmentation (dim=2)
        print("Evaluating tumor segmentation...")
        tumor_dice_score = evaluate(model, device, testloader, dim=2)
        print(f"Tumor Dice Score: {tumor_dice_score:.4f}")

    print("\n" + "="*50)
    print("Test evaluation complete!")
    print(f"Liver Dice Score: {liver_dice_score:.4f}")
    print(f"Tumor Dice Score: {tumor_dice_score:.4f}")
    print("="*50)

    return {
        "liver_dice": liver_dice_score,
        "tumor_dice": tumor_dice_score
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train or test the Dense UNet 3D model')
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'],
                        help='Operation mode: train or test (default: test)')
    # add argument for number of classes
    parser.add_argument('--num_classes', type=int, default=16,
                        help='Number of segmentation classes (default: 16)')
    args = parser.parse_args()

    # Load configuration
    with open("./dense_unet_3d/config.yaml", "r") as infile:
        config = yaml.load(infile, Loader=yaml.FullLoader)

    # Set device
    if torch.cuda.is_available() and config["gpu"]["use_gpu"]:
        device = torch.device(config["gpu"]["gpu_name"])
        print(f"Using device: {device}")
    else:
        device = torch.device("cpu")
        print("Using device: cpu")

    # Run selected mode
    if args.mode == 'train':
        train(config, device, args.num_classes)
    elif args.mode == 'test':
        test(config, device, args.num_classes)
