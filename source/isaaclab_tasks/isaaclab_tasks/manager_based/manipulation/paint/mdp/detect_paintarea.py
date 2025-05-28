import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import os
import shutil
from sklearn.cluster import KMeans
from scipy.spatial import distance

# Global variable for the SAM mask generator
mask_generator = None

def initialize_sam():
    """Initializes the global SAM mask generator if it hasn't been initialized yet."""
    global mask_generator
    if mask_generator is not None:
        print("SAM mask generator already initialized.")
        return mask_generator
    
    # sam_checkpoint = "./sam_vit_b_01ec64.pth"  
    # Using the path provided in the last manual edit
    sam_checkpoint = "paint/sam_vit_b_01ec64.pth"  
    model_type = "vit_b" 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading SAM model from {sam_checkpoint} on {device}...")
    try:
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
        print("SAM mask generator initialized successfully.")
        return mask_generator
    except FileNotFoundError:
        print(f"Error: SAM checkpoint file not found at {sam_checkpoint}. Please ensure the file exists.")
        mask_generator = None # Ensure it's None if loading fails
        return None
    except Exception as e:
        print(f"An error occurred during SAM initialization: {e}")
        mask_generator = None # Ensure it's None if initialization fails
        return None

def calculate_painting_completion_rate(image_path):
    """Calculates the wall painting completion rate using a provided SAM mask generator.

    Args:
        image_path (str): The path to the input image.
        mask_generator (SamAutomaticMaskGenerator): The initialized SAM mask generator.

    Returns:
        float: The painting completion rate as a percentage (0.0 to 100.0),
               or None if calculation fails.
    """
   
    # if mask_generator is None:
    #     print("Error: SAM mask generator is not initialized. Cannot calculate completion rate.")
    #     return None

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return None
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
    # Generate all masks 
    print("Generating SAM masks...")
    masks = mask_generator.generate(image_rgb)
    print(f"Generated {len(masks)} masks")
    
    wall_mask = None
    wall_score = 0
    
    # Find the most likely wall mask based on heuristics (size, position, shape)
    for mask_data in masks:
        mask = mask_data['segmentation']
        
        # Calculate bounding box for scoring
        # Note: cv2.boundingRect expects a single channel mask, convert boolean mask to uint8
        mask_uint8 = mask.astype(np.uint8) * 255
        x, y, w, h = cv2.boundingRect(mask_uint8)
        
        # Scoring metrics:
        # 1. Size - larger masks in the center are preferred
        image_area = image.shape[0] * image.shape[1]
        size_score = mask_data['area'] / image_area
        
        # 2. Position - masks closer to the center are preferred
        image_center_x = image.shape[1] / 2
        image_center_y = image.shape[0] / 2
        mask_center_x = x + w / 2
        mask_center_y = y + h / 2
        dist_to_center = np.sqrt((mask_center_x - image_center_x)**2 + (mask_center_y - image_center_y)**2)
        max_dist = np.sqrt(image_center_x**2 + image_center_y**2)
        center_score = 1 - (dist_to_center / max_dist) if max_dist > 0 else 0
        
        # 3. Shape - more rectangular masks are preferred (closer to a wall shape)
        # Find contours to calculate area for shape score
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_area = cv2.contourArea(contours[0]) if len(contours) > 0 else 0
        bbox_area = w * h
        rectangle_score = min(1, 4 * contour_area / bbox_area) if bbox_area > 0 else 0 # Factor 4 is arbitrary, adjust as needed
        
        # Combined score (weights can be tuned)
        combined_score = 0.5 * size_score + 0.3 * center_score + 0.2 * rectangle_score
        
        if combined_score > wall_score:
            wall_score = combined_score
            wall_mask = mask
    
    if wall_mask is None or wall_score < 0.1: # Add a minimum score threshold
        # print("Warning: Wall not detected or score too low.")
        return 0.0 # Return 0% completion if wall not confidently detected
    
    # Apply the wall mask to isolate the wall area in the image
    wall_only_rgb = image_rgb.copy()
    wall_only_rgb[~wall_mask] = [0, 0, 0] 
   
    # Extract pixels within the wall mask
    wall_pixels = wall_only_rgb[wall_mask]
    
    # Skip if there are no wall pixels (shouldn't happen if wall_mask is found, but good check)
    if len(wall_pixels) == 0:
        return 0.0
    
    # Use K-means to cluster wall pixels into painted and unpainted
    # Reshape pixels for KMeans: (N, 3) where N is number of pixels, 3 for RGB channels
    wall_pixels_reshaped = wall_pixels.reshape(-1, 3)
    
    # Ensure enough pixels for clustering
    if wall_pixels_reshaped.shape[0] < 2:
        # print("Warning: Not enough wall pixels for clustering.")
        return 0.0

    try:
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10) # n_init='auto' in newer sklearn versions
        kmeans.fit(wall_pixels_reshaped)
        labels = kmeans.predict(wall_pixels_reshaped)
    except Exception as e:
        print(f"Error during KMeans clustering: {e}")
        return None # Return None if clustering fails
    
    # Separate pixels into clusters
    cluster0_pixels = wall_pixels_reshaped[labels == 0]
    cluster1_pixels = wall_pixels_reshaped[labels == 1]
    
    # Calculate color statistics for each cluster to identify which is 'painted'
    # This logic assumes painted color is distinct from unpainted wall color (e.g., gray)
    # You might need to tune the reference_color and comparison logic
    
    # Handle cases where one cluster is empty after prediction (unlikely with successful fit, but robust)
    if len(cluster0_pixels) == 0:
        # print("Warning: Cluster 0 is empty.")
        paint_cluster_label = 1
    elif len(cluster1_pixels) == 0:
        # print("Warning: Cluster 1 is empty.")
        paint_cluster_label = 0
    else:
        cluster0_mean = np.mean(cluster0_pixels, axis=0) # type: ignore
        cluster1_mean = np.mean(cluster1_pixels, axis=0) # type: ignore

        # Simple heuristic: Assume the painted cluster's mean color is further from gray
        gray_ref = np.array([128, 128, 128]) # Mid-gray reference
        dist_to_gray_0 = distance.euclidean(cluster0_mean, gray_ref)
        dist_to_gray_1 = distance.euclidean(cluster1_mean, gray_ref)

        # The cluster with the color furthest from gray is more likely to be the painted area
        paint_cluster_label = 0 if dist_to_gray_0 > dist_to_gray_1 else 1

    # Create the paint mask based on the identified painted cluster
    paint_mask_indices = np.where(labels == paint_cluster_label)[0]
    
    # Create a mask over the original image shape
    final_paint_mask = np.zeros_like(wall_mask, dtype=np.uint8)
    # Map the clustered pixels back to their original positions in the wall_mask area
    # Need to find the indices in the original wall_mask that correspond to wall_pixels_reshaped
    # This part is tricky as KMeans flattens the array. A more robust way is to apply labels back to the original wall_mask shape.
    
    # A potentially simpler approach: create a result image of the same size as wall_only_rgb,
    # fill in the wall_mask area with clustered colors, then threshold.
    clustered_image = np.zeros_like(wall_only_rgb)
    clustered_image[wall_mask] = [kmeans.cluster_centers_[label] for label in labels]
    
    # Now create the paint mask from the clustered image. Identify pixels belonging to the paint_cluster_label.
    # We need to compare pixel colors to the mean of the paint cluster, accounting for floating point inaccuracies.
    paint_cluster_mean_color = kmeans.cluster_centers_[paint_cluster_label]
    # Define a tolerance for color comparison
    color_tolerance = 5 # Adjust tolerance as needed

    # Create boolean mask for pixels close to the paint cluster mean within the wall area
    is_painted = np.all(np.abs(clustered_image - paint_cluster_mean_color) < color_tolerance, axis=-1)
    final_paint_mask = ((wall_mask) & is_painted).astype(np.uint8) * 255

    # Optional: Apply morphological operations to refine the paint mask (re-added as they can be useful)
    # kernel = np.ones((5, 5), np.uint8)
    # final_paint_mask = cv2.morphologyEx(final_paint_mask, cv2.MORPH_OPEN, kernel)
    # final_paint_mask = cv2.morphologyEx(final_paint_mask, cv2.MORPH_CLOSE, kernel)

    total_wall_area = np.sum(wall_mask)
    painted_area = np.sum(final_paint_mask > 0)
    
    # Avoid division by zero if no wall area is detected (already checked wall_mask is not None)
    if total_wall_area == 0:
        # print("Warning: Total wall area is zero.")
        return 0.0

    completion_rate = (painted_area / total_wall_area) * 100.0
    
    return completion_rate

def main():
    """Example usage of the painting completion rate calculation."""
    # This main function is for testing the detect_paintarea.py script directly.
    # When used from another script (like record_demos.py), initialize_sam() should be called once
    # and the resulting mask_generator passed to calculate_painting_completion_rate.

    image_path = "/media/mgt/76ACB30FACB2C8BF/depthanything/Inline-image-2025-05-20 13.43.54.006.png"
    
    # Initialize SAM mask generator (only once in the calling script)
    sam_gen = initialize_sam()
    
    if sam_gen is not None:
        # Calculate completion rate using the initialized generator
        completion_rate = calculate_painting_completion_rate(image_path)
        
        if completion_rate is not None:
            print(f"Painting completion rate for {image_path}: {completion_rate:.2f}%")
        else:
            print(f"Failed to calculate completion rate for {image_path}")
    else:
        print("SAM mask generator failed to initialize.")

if __name__ == "__main__":
    main()
 
 

    
    