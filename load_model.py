import torch
from torch.cuda.amp import autocast
from model import genconvit_ed, genconvit,config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_state(weight_path):
    # Load the saved state dictionary
    model_config = config.load_config()
    model = genconvit.GenConViTED(model_config)  # or GenConViTVAE(config)
    checkpoint = torch.load(weight_path, map_location=device)
    
    # Load state dict into model
    model.load_state_dict(checkpoint['state_dict'])
    
    # Optionally, if you need the optimizer state
    # optimizer.load_state_dict(checkpoint['optimizer'])
    
    # Get other saved information if needed
    epoch = checkpoint['epoch']
    min_loss = checkpoint['min_loss']
    model.to(device)  # Move to GPU if available
    model = model.half()  # Convert model parameters to half precision
    model.eval()  # Set to evaluation mode

    return model, epoch, min_loss

def initialize_model():
    # Load the default configuration
    model_config = config.load_config()
    
    # Initialize model without weights
    model = genconvit.GenConViTED(model_config)
    
    # Move to device and set to evaluation mode
    model.to(device)
    model = model.half()  # Convert to half precision
    model.eval()
    
    return model


model, epoch, min_loss = load_model_state( "weight/genconvit_ed_Nov_28_2024_01_05_29.pth")
model = initialize_model()
# Calculate and print trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Number of trainable parameters: {trainable_params:,}')

print(epoch)
# Convert model to FP16
