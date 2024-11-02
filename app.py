import cv2
import numpy as np
import requests
from urllib.parse import urlparse
from pdf2image import convert_from_path, convert_from_bytes
import os

def is_url(path):
    """Check if the given path is a URL."""
    try:
        result = urlparse(path)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

def is_pdf(path):
    """Check if the file is a PDF."""
    return path.lower().endswith('.pdf')

def download_file(url):
    """Download file from URL."""
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Failed to download file from URL: {url}")
    return response.content

def process_pdf(pdf_path):
    """Convert PDF to images."""
    try:
        if is_url(pdf_path):
            # If PDF is from URL, download it first
            pdf_content = download_file(pdf_path)
            images = convert_from_bytes(pdf_content)
        else:
            # If PDF is local file
            images = convert_from_path(pdf_path)
        return images
    except Exception as e:
        raise ValueError(f"Error processing PDF: {str(e)}")

def download_image(url):
    """Download image from URL and convert to OpenCV format."""
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Failed to download image from URL: {url}")
    
    image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    
    if image is None:
        raise ValueError(f"Failed to decode image from URL: {url}")
    
    return image

def detect_blur(image, threshold=100):
    """
    Detect if an image is blurry using Laplacian variance method.
    
    Args:
        image: OpenCV image or PIL Image
        threshold (float): Threshold value for blur detection
    
    Returns:
        tuple: (is_blurry, blur_score)
    """
    # Convert PIL Image to OpenCV format if necessary
    if not isinstance(image, np.ndarray):
        # Convert PIL Image to OpenCV format
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate Laplacian variance
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    blur_score = np.var(laplacian)
    
    # Determine if image is blurry
    is_blurry = blur_score < threshold
    
    return is_blurry, blur_score

def process_file(file_path, threshold=100):
    """Process either PDF or image file."""
    if is_pdf(file_path):
        print(f"Processing PDF: {file_path}")
        images = process_pdf(file_path)
        for i, image in enumerate(images):
            is_blurry, score = detect_blur(image, threshold)
            status = "BLURRY" if is_blurry else "CLEAR"
            print(f"Page {i+1}:")
            print(f"Status: {status}")
            print(f"Blur Score: {score:.2f}")
            print("-" * 50)
    else:
        # Handle as regular image
        if is_url(file_path):
            image = download_image(file_path)
        else:
            image = cv2.imread(file_path)
            
        if image is None:
            raise ValueError(f"Could not read image at {file_path}")
            
        is_blurry, score = detect_blur(image, threshold)
        status = "BLURRY" if is_blurry else "CLEAR"
        print(f"Image: {file_path}")
        print(f"Status: {status}")
        print(f"Blur Score: {score:.2f}")

if __name__ == "__main__":
    # Example usage - works with PDFs, images, and URLs
    file_path = "random.pdf"  # Can be PDF, image, or URL
    threshold = 100  # Adjust this value based on your needs
    
    try:
        process_file(file_path, threshold)
    except Exception as e:
        print(f"Error: {str(e)}")