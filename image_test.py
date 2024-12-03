import torch
import os
from model.config import load_config
from model.genconvit_ed import GenConViTED
from model.genconvit_vae import GenConViTVAE
from dataset.loader import load_data
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from datetime import datetime

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure and axes
    plt.figure(figsize=(10, 8))
    
    # Plot confusion matrix using seaborn
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save or show the plot
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()
    plt.close()

def test_model(weight_path, model_type='ed', batch_size=1):
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load config
    config = load_config()

    # Load test data
    test_path = "dataset/combined_data"
    dataloaders, dataset_sizes = load_data(test_path, batch_size)
    test_loader = dataloaders['test']

    # Initialize model based on type
    if model_type == 'ed':
        model = GenConViTED(config)
    else:
        model = GenConViTVAE(config)

    # Load weights
    print(f"Loading weights from {weight_path}")
    checkpoint = torch.load(weight_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    model.eval()

    # Test loop
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    print("\nRunning inference on test set...")
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            if model_type == 'ed':
                outputs = model(inputs)
            else:
                outputs = model(inputs)[0]

            # Get predictions
            _, predicted = torch.max(outputs, 1)
            
            # Calculate accuracy
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Store predictions and labels for confusion matrix
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Print progress
            if batch_idx % 10 == 0:
                print(f'Processed {batch_idx * batch_size}/{len(test_loader.dataset)} images')

    accuracy = 100 * correct / total
    print(f'\nTest Results:')
    print(f'Accuracy on test set: {accuracy:.2f}% ({correct}/{total})')

    # Generate and save confusion matrix
    print("\nGenerating confusion matrix...")
    plot_confusion_matrix(
        all_labels, 
        all_predictions, 
        save_path=f'result/figures/confusion_matrix_{model_type}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    )

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions))

def main():
    parser = argparse.ArgumentParser(description='Test GenConViT model on test dataset')
    parser.add_argument('--weight_path', type=str, required=True,
                      help='Path to the pretrained model weights (.pth file)')
    parser.add_argument('--model_type', type=str, choices=['ed', 'vae'],
                      default='ed', help='Model type (ed or vae)')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for testing')

    args = parser.parse_args()

    test_model(args.weight_path, args.model_type, args.batch_size)

if __name__ == '__main__':
    main()
