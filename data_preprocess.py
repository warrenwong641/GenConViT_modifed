import os
from PIL import Image
import glob

def preprocess_images(input_dir='dataset/combined_data', output_dir='dataset/preprocessed_data'):
    # Create base output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each split (train, valid, test)
    for split in ['train', 'valid', 'test']:
        # Process each category (fake, real)
        for category in ['fake', 'real']:
            # Create corresponding output directories
            input_path = os.path.join(input_dir, split, category)
            output_path = os.path.join(output_dir, split, category)
            os.makedirs(output_path, exist_ok=True)
            
            # Get all image files in current directory
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                image_files.extend(glob.glob(os.path.join(input_path, ext)))
                image_files.extend(glob.glob(os.path.join(input_path, ext.upper())))
            
            total_images = len(image_files)
            print(f"\nProcessing {split}/{category}: Found {total_images} images")
            
            # Process each image
            for idx, img_path in enumerate(image_files, 1):
                try:
                    # Open image
                    with Image.open(img_path) as img:
                        # Convert to RGB if necessary
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        
                        # Resize image to 256x256
                        resized_img = img.resize((256, 256), Image.Resampling.LANCZOS)
                        
                        # Prepare output filename
                        filename = os.path.basename(img_path)
                        output_file_path = os.path.join(output_path, filename)
                        
                        # Save processed image
                        resized_img.save(output_file_path, quality=95)
                        
                    print(f"Processed {split}/{category} - {idx}/{total_images}: {filename}")
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {str(e)}")
    
    print("\nPreprocessing completed!")

if __name__ == "__main__":
    preprocess_images()
