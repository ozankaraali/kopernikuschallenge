import os
import cv2
from imaging_interview import preprocess_image_change_detection, compare_frames_change_detection


def remove_duplicates(path, min_contour_area=500, score_threshold=10000):
    # Read all images in the directory
    filenames = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    images = [cv2.imread(os.path.join(path, f)) for f in filenames]

    # Preprocess all images if image is readable
    images_preprocessed = [preprocess_image_change_detection(img) for img in images if img is not None]

    # Compare each pair of images
    for i in range(len(images_preprocessed)):
        for j in range(i + 1, len(images_preprocessed)):
            try:
                # Calculate the score
                score, _, _ = compare_frames_change_detection(images_preprocessed[i], images_preprocessed[j],
                                                              min_contour_area)

                # If the score is less than a certain threshold, consider the images to be duplicates
                if score < score_threshold:
                    # Delete the second image
                    os.remove(os.path.join(path, filenames[j]))
                    print(f'Removed {filenames[j]} as it was similar to {filenames[i]}')
                    # Ensure we don't attempt to remove it again
                    images_preprocessed[j] = None
            except cv2.error as e:
                # size error, it is expected when some images are in different sizes, so we just ignore it
                # we could also resize the images to the same size, but it wasn't specified in the task
                continue
            except Exception as e:
                # print the exception if there is any other error for debugging purposes
                print(e)
                continue

    print('Finished removing duplicates.')


if __name__ == '__main__':
    # Call the function with the path to your directory
    remove_duplicates('./dataset')