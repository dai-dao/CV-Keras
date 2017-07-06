import cv2
import time
import numpy as np

# Define the prediction model

def get_model():
    from keras.models import Sequential
    from keras.layers import Convolution2D, MaxPooling2D, Dropout
    from keras.layers import Flatten, Dense
    
    model = Sequential()
    model.add(Convolution2D(64, 3, 3, input_shape=((96, 96, 1))))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(128, activation='tanh'))
    model.add(Dropout(0.3))
    model.add(Dense(30, activation='tanh'))

    # Load trained weights
    model.load_weights('model.h5')
    return model


def warp(image, dest_points):
    w, h = image.shape[1], image.shape[0]

    src_points = np.float32([[0, 0],
                             [0, h],
                             [w, h],
                             [w, 0]])
    
    M = cv2.getPerspectiveTransform(src_points, dest_points)
    image_size = (image.shape[1], image.shape[0])
    warped = cv2.warpPerspective(image, M, image_size, flags=cv2.INTER_LINEAR)
    return warped


def detect_keypoints(image, model):
    # image: raw image read by Opencv (BGR)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Extract the pre-trained face detector from an xml file
    face_cascade = cv2.CascadeClassifier('detector_architectures/haarcascade_frontalface_default.xml')
    # Detect the faces in image
    faces = face_cascade.detectMultiScale(gray, 2, 6)
    # Make a copy of the orginal image to draw face detections on
    image_with_detections = np.copy(image)
    num_face_keypoints = []
    
    # Get the bounding box for each detected face
    for (x,y,w,h) in faces:
        face_img = image_with_detections[y:y+h, x:x+w, :]
        # Pre-process
        face_reshaped = cv2.resize(face_img, (96, 96))
        gray = cv2.cvtColor(face_reshaped, cv2.COLOR_RGB2GRAY)
        gray_normalized = gray / 255.
        gray_normalized = gray_normalized[np.newaxis, :, :, np.newaxis]
        # Predict
        key_points = model.predict(gray_normalized)
        key_points = key_points * 48 + 48
        # Re-normalize
        x_coords = key_points[0][0::2]
        y_coords = key_points[0][1::2]
        x_coords = x_coords * w / 96 + x
        y_coords = y_coords * h / 96 + y
        # Add a red bounding box to the detections image
        cv2.rectangle(image_with_detections, (x,y), (x+w,y+h), (255,0,0), 3)
        num_face_keypoints.append((x_coords, y_coords))
        
    for face in num_face_keypoints:
        for x, y in zip(face[0], face[1]):
            cv2.circle(image_with_detections, (x, y), 3, (0,0,255), -1)
        
    return num_face_keypoints, image_with_detections


def sunglasses_overlay(image, model, reshaped_sunglasses):
    keypoints, _ = detect_keypoints(image, model)

    image_with_sunglasses = np.copy(image)
    output = np.copy(image)

    for person_points in keypoints:
        important_points_x, important_points_y = person_points
        h = (important_points_y[5] - important_points_y[9]) * 3

        dest_pts = np.float32([[important_points_x[9], important_points_y[9]],
                               [important_points_x[9], important_points_y[9]+h],
                               [important_points_x[7], important_points_y[9]+h],
                               [important_points_x[7], important_points_y[9]]])

        warped_sunglasses = warp(reshaped_sunglasses, dest_pts)
        mask = warped_sunglasses[:,:,:3]
        mask[mask == 0] = 255
        output[mask != 255] = mask[mask != 255]

    return output 


def laptop_camera_go(model):
    sunglasses = cv2.imread("images/sunglasses_4.png", cv2.IMREAD_UNCHANGED)
    reshaped_sunglasses = None
    
    # Create instance of video capturer
    cap = cv2.VideoCapture(0)
    cap.set(3, 200)
    cap.set(4, 200)
    
    cv2.namedWindow('face detection activated',cv2.WINDOW_NORMAL)
    cv2.namedWindow('sunglasses overlay',cv2.WINDOW_NORMAL)

    #fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # 'x264' doesn't work
    #overlay_out = cv2.VideoWriter('overlay.avi', fourcc, 20.0, (288, 352), False)
    #keypoints_out = cv2.VideoWriter('keypoints.avi', fourcc, 20.0, (288, 352), False)

    t = 0

    while(t < 500):
        t += 1
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if reshaped_sunglasses is None:
            reshaped_sunglasses = cv2.resize(sunglasses, (frame.shape[1], frame.shape[0]))
        
        _, detected_frame = detect_keypoints(frame, model)
        overlay = sunglasses_overlay(frame, model, reshaped_sunglasses)
        
        # Display the resulting frame
        cv2.imshow("face detection activated", detected_frame)
        cv2.imshow("sunglasses overlay", overlay)

        # Write image to folder
        cv2.imwrite('overlay/im_{}.png'.format(t), overlay)
        cv2.imwrite('keypoints/im_{}.png'.format(t), detected_frame)

        # Write to file --------- DOESN't WORK
        #overlay_out.write(overlay)
        #keypoints_out.write(detected_frame)

        # Quit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(0.05)

    # When everything done, release the capture
    #overlay_out.release()
    #keypoints_out.release()
    cap.release()
    cv2.destroyAllWindows()
    

if __name__ == '__main__':
    
    
    model = get_model()
    laptop_camera_go(model)

    '''
    image = cv2.imread('images/obamas4.jpg')
    sunglasses = cv2.imread("images/sunglasses_4.png", cv2.IMREAD_UNCHANGED)
    reshaped_sunglasses = cv2.resize(sunglasses, (image.shape[1], image.shape[0]))

    _, detected = detect_keypoints(image, model)
    overlay = sunglasses_overlay(image, model)
    
    cv2.namedWindow('face detection activated',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('face detection activated', 100, 100)
    cv2.imshow('face detection activated', detected)
    
    cv2.namedWindow('overlay',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('overlay', 100, 100)
    cv2.imshow('overlay', overlay)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''