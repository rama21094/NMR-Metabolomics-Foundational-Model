# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 11:13:55 2025

@author: shank
"""

import torch
from torchviz import make_dot

# Assuming your model definition code is available
# For example, from your tf_aug_1.py file
from transformer import NMRMaskedAutoencoder

# Load your model from the .pth file
# This assumes you have the model's configuration saved
model_path = 'nmr_ssl_final_model.pth' # Replace with your filename
checkpoint = torch.load(model_path, map_location='cpu')

# Re-initialize the model using the saved config
model = NMRMaskedAutoencoder(
    spectrum_length=checkpoint['spectrum_length'],
    patch_size=checkpoint['patch_size'],
    **checkpoint['model_config']
)
# Load the trained weights
model.load_state_dict(checkpoint['model_state_dict'])

# Create a dummy input tensor with the correct shape
# Use your spectrum length and patch size
dummy_input = torch.randn(1, checkpoint['spectrum_length'])

# Pass the dummy input through the model
# You might need to adjust this depending on your model's forward() method
output, _ = model(dummy_input)

# Create the visualization
dot = make_dot(output, params=dict(model.named_parameters()))

# Save the plot to a file (e.g., PDF or PNG)
dot.render("nmr_model_architecture", format="png")

print("Model architecture plot saved as 'nmr_model_architecture.png'")
