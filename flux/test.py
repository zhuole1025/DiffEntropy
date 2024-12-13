import torch
from PIL import Image
from torchvision import transforms
from basicsr.utils import USMSharp
import matplotlib.pyplot as plt

def main():
    # Load and convert image to tensor
    img_path = "/data/zl/DiffEntropy/flux/demos/fighter.webp"  # Replace with your image path
    image = Image.open(img_path).convert('RGB')
    to_tensor = transforms.ToTensor()
    image_tensor = to_tensor(image).unsqueeze(0)  # Add batch dimension

    # Apply USM sharpening
    usm_sharpener = USMSharp()
    sharpened = usm_sharpener(image_tensor)

    # Convert tensors back to images for visualization
    to_pil = transforms.ToPILImage()
    original_img = to_pil(image_tensor.squeeze())
    sharpened_img = to_pil(sharpened.squeeze())

    # Display images side by side
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(original_img)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(sharpened_img)
    plt.title('USM Sharpened')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

    # Optionally save the sharpened image
    sharpened_img.save('sharpened_output.png')

if __name__ == "__main__":
    main()