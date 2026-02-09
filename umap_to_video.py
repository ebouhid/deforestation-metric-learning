import cv2
import os
import argparse

def images_to_video(image_folder, output_video, fps=30, resolution=None):
    # Get image files sorted alphabetically
    images = sorted([img for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))], key=lambda x: int(os.path.splitext(x)[0].split('_')[-1].replace('vep','')))
    
    if not images:
        print("No images found in the specified folder.")
        return
    
    # Read the first image to determine resolution
    first_image = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, _ = first_image.shape
    
    if resolution:
        width, height = resolution
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)
        
        if resolution:
            frame = cv2.resize(frame, (width, height))
        
        video_writer.write(frame)
    
    video_writer.release()
    print(f"Video saved as {output_video}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--image_dir", type=str, help='Path to input dir')
    parser.add_argument('-o', "--output_file", type=str, help='Output filename')
    parser.add_argument('-f', "--fps", type=int, help="Output video FPS", default=30)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    images_to_video(image_folder=args.image_dir, output_video=args.output_file, fps=args.fps, resolution=None)
