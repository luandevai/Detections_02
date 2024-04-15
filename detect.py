import cv2
import numpy as np

def load_model():
    # Load the pre-trained MobileNet SSD model from OpenCV's model zoo
    config_path = 'MobileNetSSD_deploy.prototxt'
    model_path = 'MobileNetSSD_deploy.caffemodel'
    net = cv2.dnn.readNetFromCaffe(config_path, model_path)
    return net

def detect_objects(image_path, net):
    # Read the image
    image = cv2.imread(image_path)
    (h, w) = image.shape[:2]

    # Create a blob from the image
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

    # Set the blob as input to the model
    net.setInput(blob)

    # Forward pass through the network to get the detections
    detections = net.forward()

    # Loop over the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        # Filter out weak detections
        if confidence > 0.2:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            # Draw the prediction on the image
            label = f"{confidence:.2f}"
            cv2.rectangle(image, (startX, startY), (endX, endY), (23, 230, 210), 2)
            cv2.putText(image, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (23, 230, 210), 2)

    # Display the output image
    cv2.imshow("Output", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Load the pre-trained model
net = load_model()

# Path to your image
image_path = 'OIF.jpg'

# Detect objects in the image
detect_objects(image_path, net)
