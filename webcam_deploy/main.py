import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}


def prediction(frame):
    """
    Parameters:
    frame: an image frame from the webcam

    Output:
    keypoints_with_scores: a list of keypoints with their corresponding scores have 4 dimensions (1, 1, 17, 3)

    Example with right eye keypoints:
    [0.4741213, 0.42508835, 0.75313896] :
    x = 0.4741213: The x-coordinate is approximately 47.4% of the image's width from the left.
    y = 0.42508835: The y-coordinate is approximately 42.5% of the image's height from the top.
    score = 0.75313896: Confidence score of 0.753, indicating high confidence in this keypoint.
    """

    #kich thuoc cam hien co (480, 640, 3)

    img = frame.copy()
    img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192, 192) #(1,480,640,3) -> (1,192,192,3)
    input_image = tf.cast(img, dtype=tf.float32)

    # load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=r"single_pose.tflite")
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()



    interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
    interpreter.invoke()
    keypoints_with_scores =interpreter.get_tensor(output_details[0]['index']) #17 keypoints start with index 0 (in the order of: [nose, left eye, right eye, left ear, right ear, left shoulder, right shoulder, left elbow, right elbow, left wrist, right wrist, left hip, right hip, left knee, right knee, left ankle, right ankle])
    
    return keypoints_with_scores

def draw_keypoints(frame, keypoints, confidence_threshold):
    x, y , c = frame.shape


    shaped = np.squeeze(np.multiply(keypoints, [x, y, 1]))

    for keypoints in shaped:
        ky, kx, kp_conf = keypoints

        if (kp_conf > confidence_threshold):
            cv2.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)

def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape

    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for edge, color in edges.items():
        p1, p2 = edge

        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if (c1 > confidence_threshold) and (c2 > confidence_threshold):
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255,0,0), 2)



def main():
    # Open the webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error")
        return
    
    while True:
        ret, frame = cap.read()

        keypoints_with_scores = prediction(frame)
        
        
        draw_connections(frame, keypoints_with_scores,KEYPOINT_EDGE_INDS_TO_COLOR, confidence_threshold=0.4)
        draw_keypoints(frame, keypoints_with_scores, confidence_threshold=0.4)

        cv2.imshow('MoveNet Lightning', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and destroy all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()