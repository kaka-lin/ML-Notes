"""MediaPipe Object Detector Example."""
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red


def visualize(image, detection_result) -> np.ndarray:
    """Draws bounding boxes on the input image and return it.
    Args:
        image: The input RGB image.
        detection_result: The list of all "Detection" entities to be visualize.
    Returns:
        Image with bounding boxes.
    """
    for detection in detection_result.detections:
        # Draw bounding_box
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)

        # Draw label and score
        category = detection.categories[0]
        category_name = category.category_name
        probability = round(category.score, 2)
        result_text = category_name + ' (' + str(probability) + ')'
        text_location = (MARGIN + bbox.origin_x,
                         MARGIN + ROW_SIZE + bbox.origin_y)
        cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

    return image


if __name__ == "__main__":
    IMAGE_FILE = 'images/image.jpg'
    img = cv2.imread(IMAGE_FILE)

    # STEP 1: Create an ObjectDetector object.
    base_options = python.BaseOptions(model_asset_path='model/efficientdet.tflite')
    options = vision.ObjectDetectorOptions(base_options=base_options,
                                           score_threshold=0.5)

    # Use the `create_from_options()` function to create the task
    #   including: running mode, display names, max number of results,
    #              confidence threshold, category allow list, and deny list.
    # https://developers.google.com/mediapipe/solutions/vision/object_detector/python#configuration
    detector = vision.ObjectDetector.create_from_options(options)

    # STEP 2: Load the input image.
    image = mp.Image.create_from_file(IMAGE_FILE)

    # STEP 3: Detect objects in the input image.
    detection_result = detector.detect(image)

    # STEP 4: Process the detection result. In this case, visualize it.
    image_copy = np.copy(image.numpy_view())
    annotated_image = visualize(image_copy, detection_result)
    rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

    cv2.imshow('image', rgb_annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
