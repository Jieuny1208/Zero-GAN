
import os
import json
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import timm
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import cv2

# 모델 관련 클래스 import (ZeroGanGenerator, Discriminator, ImageToInput)
from models.zero_gan import ZeroGanGenerator, Discriminator, ImageToInput
from utils.visualizer import Visualizer


def visualize_difference(original, reconstructed, difference, save_path=None):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(original)
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    axs[1].imshow(reconstructed)
    axs[1].set_title('Reconstructed Image')
    axs[1].axis('off')

    difference = (difference - np.min(difference)) / (np.max(difference) - np.min(difference))  # Normalize to 0-1
    norm_difference = np.clip(difference * 2, 0, 1)  # Amplify differences
    axs[2].imshow(norm_difference, cmap='hot')
    axs[2].set_title('Difference Map')
    axs[2].axis('off')

    if save_path:
        plt.savefig(save_path)
    plt.show()

def calculate_anomaly_score(original, reconstructed):
    return np.mean(np.abs(original - reconstructed))

def visualize_anomaly_score_distribution(scores):
    plt.figure(figsize=(10, 5))
    plt.hist(scores, bins=50, alpha=0.75, color='blue', edgecolor='black')
    plt.title('Anomaly Score Distribution')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()



def enhance_visuality(image):
    threshold = 100/255
    max_val = 255/255

    _, thresholded_image = cv2.threshold(image, threshold, max_val, cv2.THRESH_BINARY)

    return thresholded_image


def test_single_image(image, model, image_to_input, discriminator, device, net, opt, patch_size=14):
    original_image = image.convert('RGB')
    original_size = original_image.size  # 원본 이미지 크기 저장

    # 패치 크기에 맞게 이미지 크기 조정
    new_width = (original_size[0] // patch_size) * patch_size
    new_height = (original_size[1] // patch_size) * patch_size
    original_image = original_image.resize((new_width, new_height))
    original_np = np.array(original_image) / 255.0

    transform = transforms.Compose([
        transforms.Resize((opt['image_size'], opt['image_size'])),  # 모델의 입력 크기에 맞게 조정
        transforms.ToTensor()
    ])
    img_tensor = transform(original_image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        gen_img_accum = torch.zeros_like(img_tensor).to(device)
        mask_iter = 5
        for _ in range(mask_iter):
            embeddings, _ = image_to_input(img_tensor)
            latent_i, gen_img, latent_o = model(embeddings)
            gen_img_accum += gen_img

    generated_img_tensor = gen_img_accum / mask_iter
    generated_img_tensor = nn.functional.interpolate(generated_img_tensor, size=(new_height, new_width), mode='bilinear', align_corners=False)
    generated_img_np = generated_img_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)

    difference_np = np.abs(original_np - generated_img_np)
    difference_np = enhance_visuality(difference_np)
    anomaly_score = calculate_anomaly_score(original_np, generated_img_np)

    return original_image, Image.fromarray((generated_img_np * 255).astype(np.uint8)), Image.fromarray((difference_np * 255).astype(np.uint8)), anomaly_score

def run_anomaly_detection(image):
    try:
        model_name = "ZeroGan_Adam_state_epoch_90.pt"
        model_path = 'result/train_2024_05_30_00_22/'+model_name
        with open(model_path.replace(model_name, 'config.json'), 'r', encoding='utf-8') as file:
            opt = json.load(file)

        model = ZeroGanGenerator(opt).to(device)
        discriminator = Discriminator(opt).to(device)
        image_to_input = ImageToInput(opt).to(device)

        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        # discriminator.load_state_dict(checkpoint['d_model_state'])
        image_to_input.load_state_dict(checkpoint['image_to_input_state'])

        net = timm.create_model('vit_huge_patch14_224.orig_in21k', pretrained=True, num_classes=10).to(device)

        original_image, reconstructed_image, difference_map, anomaly_score = test_single_image(image, model, image_to_input, discriminator, device, net, opt)
        
        # Anomaly Score Distribution (Sample Data for Demo)
        scores = np.random.normal(loc=0.5, scale=0.1, size=1000)  # Sample anomaly scores
        scores = np.append(scores, anomaly_score)
        visualize_anomaly_score_distribution(scores)

        return original_image, reconstructed_image, difference_map, f"Anomaly Score: {anomaly_score:.4f}"
    except Exception as e:
        print(f"Error: {e}")
        return None, None, None, f"Error occurred: {e}"


if __name__=="__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    st.title("Anomaly Detection with ZeroGan")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")
        st.write("Running anomaly detection...")
        original_image, reconstructed_image, difference_map, anomaly_score_text = run_anomaly_detection(image)
        if original_image and reconstructed_image and difference_map:
            st.image([original_image, reconstructed_image, difference_map], 
                    caption=['Original Image', 'Reconstructed Image', 'Difference Map'], 
                    use_column_width=True)
            st.write(anomaly_score_text)

