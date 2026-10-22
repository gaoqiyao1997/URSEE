import os
import numpy as np
import cv2


def find_homography_matrix(pts_src, pts_dst):
    # Convert points to numpy arrays
    pts_src = np.array(pts_src)
    pts_dst = np.array(pts_dst)

    # Calculate homography matrix
    H, _ = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC)
    return H


def warp_image(img_src, H, img_dst_shape):
    # Warp the source image to the destination based on the homography matrix
    img_warp = cv2.warpPerspective(img_src, H, img_dst_shape)
    return img_warp


def process_images(input_folder, output_folder, H):
    # Create output folder if not exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each image in the input folder
    for filename in os.listdir(input_folder):
        # Load image
        img_path = os.path.join(input_folder, filename)
        img_src = cv2.imread(img_path)

        # Warp image
        img_warp = warp_image(img_src, H, (1280, 720))

        # Save the warped image
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, img_warp)
        print(f"Saved {output_path}")


def main():
    # # Load the images
    # img1 = cv2.imread('checkboardRGB.png')  # Replace 'image1.jpg' with your first image path
    # img2 = cv2.imread('checkboardEvent.jpg')  # Replace 'image2.jpg' with your second image path

    # Feature points coordinates
    pts1 = [[391, 884], [341, 127], [476, 246], [612, 366], [748, 485], [884, 605], [1020, 725], [1156, 844],
            [1099, 93], [1234, 212], [1369, 331], [1054, 449]]
    pts2 = [[243, 568], [209, 26], [307, 114], [402, 199], [498, 284], [594, 368], [689, 455], [788, 539], [746, 5],
            [842, 90], [941, 173], [1036, 259]]

    # Find homography matrix
    H = find_homography_matrix(pts1, pts2)

    # Process images in the input folder and save the results in the output folder
    input_folder = 'zoom_original'  # Replace 'input_images_folder' with your input folder path
    output_folder = 'zoom_after'  # Replace 'output_images_folder' with your output folder path
    process_images(input_folder, output_folder, H)


if __name__ == "__main__":
    main()
