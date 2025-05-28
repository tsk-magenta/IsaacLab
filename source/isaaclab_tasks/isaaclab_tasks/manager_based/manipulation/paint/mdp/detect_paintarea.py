import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import os
import shutil
from sklearn.cluster import KMeans
from scipy.spatial import distance

mask_generator = None

def initialize_sam():
    global mask_generator
    
    # sam_checkpoint = "./sam_vit_b_01ec64.pth"  
    # sam_checkpoint = "C:/Users/tsk/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/paint/mdp/sam_vit_b_01ec64.pth"  
    sam_checkpoint = "paint/sam_vit_b_01ec64.pth"  
    model_type = "vit_b" 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading SAM model from {sam_checkpoint} on {device}...")
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    
    # SAM automatic mask generator 
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100
    )

def calculate_painting_completion_rate(image_path, output_folder="paintarea_results"):
   
    # if os.path.exists(output_folder):
    #     shutil.rmtree(output_folder) 
    # os.makedirs(output_folder, exist_ok=True)
    
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return None
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print("Generating SAM masks...")
    # Generate all masks 
    masks = mask_generator.generate(image_rgb)
    print(f"Generated {len(masks)} masks")
    
    """
    # Save the visualization of all SAM masks
    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)
    
    def show_anns(anns):
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)
        
        img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
        img[:,:,3] = 0
        
        for ann in sorted_anns:
            m = ann['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.35]])
            img[m] = color_mask
        
        ax.imshow(img)
    
    show_anns(masks)
    plt.title(f'SAM Detection: {len(masks)} masks')
    plt.axis('off')
    plt.savefig(os.path.join(output_folder, "01_all_sam_masks.png"), dpi=300)
    """
    
  
    wall_mask = None
    wall_score = 0
    wall_index = -1
    
  
    mask_scores = []
    
    for i, mask_data in enumerate(masks):
        mask = mask_data['segmentation']
        mask_img = np.zeros_like(image[:, :, 0], dtype=np.uint8)
        mask_img[mask] = 255
        
       
        x, y, w, h = cv2.boundingRect(mask_img)
        
        # Scoring metrics:
        # 1. Size 
        size_score = mask_data['area'] / (image.shape[0] * image.shape[1])
        # 2. Position  
        center_score = 1 - np.sqrt((x + w/2 - image.shape[1]/2)**2 + (y + h/2 - image.shape[0]/2)**2) / np.sqrt((image.shape[1]/2)**2 + (image.shape[0]/2)**2)
        # 3. Shape 
        contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rectangle_score = min(1, 4 * cv2.contourArea(contours[0]) / (w * h)) if len(contours) > 0 else 0
        
        # Combined score
        combined_score = 0.5 * size_score + 0.3 * center_score + 0.2 * rectangle_score
        mask_scores.append((i, combined_score, mask_data))
        
        if combined_score > wall_score:
            wall_score = combined_score
            wall_mask = mask
            wall_index = i
    
    if wall_mask is None:
        print("Error: Wall not detected")
        return None
    
    # Create a wall mask image
    wall_mask_img = np.zeros_like(image[:, :, 0], dtype=np.uint8)
    wall_mask_img[wall_mask] = 255
    
    """
    # Save the wall mask visualization
    plt.figure(figsize=(10, 10))
    plt.imshow(wall_mask_img, cmap='gray')
    plt.title('Detected Wall Mask')
    plt.axis('off')
    plt.savefig(os.path.join(output_folder, "02_wall_mask.png"), dpi=300)
    """
    
    # Create a visualization of the wall overlay on the original image
    wall_overlay = image_rgb.copy()
    wall_overlay_mask = np.zeros_like(image_rgb)
    wall_overlay_mask[wall_mask] = [0, 255, 0]  # Green for the wall
    alpha = 0.5
    cv2.addWeighted(wall_overlay_mask, alpha, wall_overlay, 1-alpha, 0, wall_overlay)
    
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(wall_overlay)
    plt.title(f'Selected Wall (Score: {wall_score:.3f})')
    plt.axis('off')
    plt.savefig(os.path.join(output_folder, "03_wall_overlay.png"), dpi=300)
    """
    
    wall_only = image_rgb.copy()
    wall_only[~wall_mask] = [0, 0, 0] 
    
    """
    # Save the wall-only image
    plt.figure(figsize=(10, 10))
    plt.imshow(wall_only)
    plt.title('Wall Area Only')
    plt.axis('off')
    plt.savefig(os.path.join(output_folder, "04_wall_only.png"), dpi=300)
    """
   
    wall_pixels = wall_only[wall_mask]
    
    # Skip if there are no wall pixels
    if len(wall_pixels) == 0:
        print("Error: No wall pixels detected")
        return None
    
    #  K-means with k=2 (painted and unpainted)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(wall_pixels)
    
    
    labels = kmeans.predict(wall_pixels)
    
    # Create masks for each cluster
    cluster_masks = []
    for i in range(2):
        cluster_mask = np.zeros_like(wall_mask_img)
        cluster_mask[wall_mask] = (labels == i) * 255
        cluster_masks.append(cluster_mask)
    
    
    cluster0_pixels = wall_pixels[labels == 0]
    cluster1_pixels = wall_pixels[labels == 1]
    
    # Calculate color statistics for each cluster
    if len(cluster0_pixels) > 0 and len(cluster1_pixels) > 0:
        cluster0_mean = np.mean(cluster0_pixels, axis=0)
        cluster1_mean = np.mean(cluster1_pixels, axis=0)
        
        cluster0_var = np.var(cluster0_pixels, axis=0).sum()
        cluster1_var = np.var(cluster1_pixels, axis=0).sum()
        
        # Calculate the color distance from gray (unpainted walls are usually grayish)
        gray_ref = np.array([128, 128, 128])  # Mid-gray reference
        cluster0_gray_dist = distance.euclidean(cluster0_mean, gray_ref)
        cluster1_gray_dist = distance.euclidean(cluster1_mean, gray_ref)
        
        
        if (cluster0_var > cluster1_var and cluster0_gray_dist > cluster1_gray_dist) or \
           (cluster0_var < cluster1_var and cluster0_gray_dist > cluster1_gray_dist * 1.5):
            paint_mask = cluster_masks[0]
            unpaint_mask = cluster_masks[1]
        else:
            paint_mask = cluster_masks[1]
            unpaint_mask = cluster_masks[0]
    else:
        # Fallback if one cluster is empty
        paint_mask = cluster_masks[0] if len(cluster0_pixels) > 0 else cluster_masks[1]
        unpaint_mask = cluster_masks[1] if len(cluster0_pixels) > 0 else cluster_masks[0]
    
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(paint_mask, cmap='gray')
    plt.title('Paint Mask (K-means)')
    plt.axis('off')
    plt.savefig(os.path.join(output_folder, "05_paint_mask_kmeans.png"), dpi=300)  
   
    wall_gray = cv2.cvtColor(wall_only, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(wall_gray, 50, 150)
    
    # Dilate edges to connect nearby paint edges
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Save the edge detection result
    plt.figure(figsize=(10, 10))
    plt.imshow(dilated_edges, cmap='gray')
    plt.title('Paint Edge Detection')
    plt.axis('off')
    plt.savefig(os.path.join(output_folder, "06_paint_edges.png"), dpi=300)
    """
       
    final_paint_mask = paint_mask
        
    kernel = np.ones((5, 5), np.uint8)
    final_paint_mask = cv2.morphologyEx(final_paint_mask, cv2.MORPH_OPEN, kernel)
    final_paint_mask = cv2.morphologyEx(final_paint_mask, cv2.MORPH_CLOSE, kernel)
    
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(final_paint_mask, cmap='gray')
    plt.title('Final Paint Mask')
    plt.axis('off')
    plt.savefig(os.path.join(output_folder, "07_final_paint_mask.png"), dpi=300)
   
    paint_viz = image_rgb.copy()
    paint_overlay = np.zeros_like(image_rgb)
    paint_overlay[final_paint_mask > 0] = [255, 0, 0]  
    alpha = 0.6
    cv2.addWeighted(paint_overlay, alpha, paint_viz, 1-alpha, 0, paint_viz)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(paint_viz)
    plt.title('Painted Areas Visualization')
    plt.axis('off')
    plt.savefig(os.path.join(output_folder, "08_paint_visualization.png"), dpi=300)
    """
   
    total_wall_area = np.sum(wall_mask)
    painted_area = np.sum(final_paint_mask > 0)
    
    completion_rate = (painted_area / total_wall_area) * 100

    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    
    # Original Image
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Wall Detection
    axes[0, 1].imshow(wall_overlay)
    axes[0, 1].set_title('Detected Wall')
    axes[0, 1].axis('off')
    
    # Paint Detection
    axes[1, 0].imshow(paint_viz)
    axes[1, 0].set_title('Detected Paint')
    axes[1, 0].axis('off')
   
    completion_viz = image_rgb.copy()
    painted_area_mask = np.zeros_like(image_rgb)
    unpainted_area_mask = np.zeros_like(image_rgb)
    
    painted_area_mask[final_paint_mask > 0] = [0, 255, 0] 
    unpainted_area_mask[(wall_mask) & (final_paint_mask == 0)] = [255, 0, 0] 
    
    alpha_painted = 0.7
    alpha_unpainted = 0.5
    
 
    cv2.addWeighted(painted_area_mask, alpha_painted, completion_viz, 1-alpha_painted, 0, completion_viz)
    cv2.addWeighted(unpainted_area_mask, alpha_unpainted, completion_viz, 1, 0, completion_viz)
    
    axes[1, 1].imshow(completion_viz)
    axes[1, 1].set_title(f'Completion Rate: {completion_rate:.2f}%')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "09_final_summary.png"), dpi=300)
    plt.close()
    """
   
    # with open(os.path.join(output_folder, "completion_rate.txt"), "w") as f:
    #     f.write(f"Wall Painting Completion Rate: {completion_rate:.2f}%\n")
    #     f.write(f"Total Wall Area: {total_wall_area} pixels\n")
    #     f.write(f"Painted Area: {painted_area} pixels\n")
    
    # print(f"Analysis complete. Results saved to {output_folder}/")
    # print(f"Wall Painting Completion Rate: {completion_rate:.2f}%")
    
    return completion_rate

def main():
  
    image_path = "/media/mgt/76ACB30FACB2C8BF/depthanything/Inline-image-2025-05-20 13.43.54.006.png"
    completion_rate = calculate_painting_completion_rate(image_path)
    
    if completion_rate is not None:
        print(f"Painting completion rate: {completion_rate:.2f}%")
    else:
        print("Failed to calculate completion rate")

if __name__ == "__main__":
    main()
 
 

    
    