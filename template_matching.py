# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt

def find_matching_boxes(image, template, detector_method, params):

    # Parameters and their default values
    MAX_MATCHING_OBJECTS = params.get('max_matching_objects', 10)
    SIFT_DISTANCE_THRESHOLD = params.get('SIFT_distance_threshold', 0.5)
    BEST_MATCHES_POINTS = params.get('best_matches_points', 20)

    # Initialize the detector and matcher
    if detector_method == "SIFT":
        detector = cv2.SIFT_create()
        bf = cv2.BFMatcher()
    elif detector_method == "ORB":
        detector = cv2.ORB_create(fastThreshold=5, edgeThreshold=10)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    else:
        raise ValueError("Unsupported detector method")

    # Find keypoints and descriptors for the template
    keypoints2, descriptors2 = detector.detectAndCompute(template, None)

    matched_boxes = []
    matching_img = image.copy()

    for i in range(MAX_MATCHING_OBJECTS):
        # Match descriptors
        keypoints1, descriptors1 = detector.detectAndCompute(matching_img, None)

        if detector_method == "SIFT":
            # Matching strategy for SIFT
            matches = bf.knnMatch(descriptors1, descriptors2, k=2)
            good_matches = [m for m, n in matches if m.distance < SIFT_DISTANCE_THRESHOLD * n.distance]
            good_matches = sorted(good_matches, key=lambda x: x.distance)[:BEST_MATCHES_POINTS]

        elif detector_method == "ORB":
            # Matching strategy for ORB
            matches = bf.match(descriptors1, descriptors2)
            matches = sorted(matches, key=lambda x: x.distance)
            good_matches = matches[:BEST_MATCHES_POINTS]
        
        else:
            raise ValueError("Unsupported detector method")

        # Extract location of good matches
        points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
        points2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])

        # Find homography for drawing the bounding box
        try:
            H, _ = cv2.findHomography(points2, points1, cv2.RANSAC, 2)
        except cv2.error:
            print("No more matching box")
            break

        # Transform the corners of the template to the matching points in the image
        h, w = template.shape[:2]
        corners = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        transformed_corners = cv2.perspectiveTransform(corners, H)
        matched_boxes.append(transformed_corners)

        # You can uncomment the following lines to see the matching process
        # Draw the bounding box
        img1_with_box = matching_img.copy()
        matching_result = cv2.drawMatches(img1_with_box, keypoints1, template, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.polylines(matching_result, [np.int32(transformed_corners)], True, (255, 0, 0), 3, cv2.LINE_AA)
        plt.imshow(matching_result, cmap='gray')
        plt.show()

        # Create a mask and fill the matched area with near neighbors
        matching_img2 = cv2.cvtColor(matching_img, cv2.COLOR_BGR2GRAY) 
        mask = np.ones_like(matching_img2) * 255
        cv2.fillPoly(mask, [np.int32(transformed_corners)], 0)
        mask = cv2.bitwise_not(mask)
        matching_img = cv2.inpaint(matching_img, mask, 3, cv2.INPAINT_TELEA)

    return matched_boxes

# Example usage:
img1 = cv2.imread('i_remoter.png' ) # Image
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

template = cv2.imread('t_remoter.png' ) # Template
template = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)

params = {
    'max_matching_objects': 10,
    'SIFT_distance_threshold': 0.85,
    'best_matches_points': 500
}

# Change to "SIFT" or "ORB" depending on your requirement
matched_boxes = find_matching_boxes(img1, template, "SIFT", params) 

# Draw the bounding boxes on the original image
for box in matched_boxes:
    cv2.polylines(img1, [np.int32(box)], True, (0, 255, 0), 3, cv2.LINE_AA)

plt.imshow(img1)
plt.show()

# %%
