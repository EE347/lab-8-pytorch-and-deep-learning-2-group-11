from picamera2 import Picamera2

import cv2

import os

 

# Initialize the camera and face detector

picam2 = Picamera2()

picam2.configure(picam2.create_preview_configuration())

picam2.start()

 

# Haarcascade for face detection

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

 

# Folder paths

data_folder = '/home/pi/ee347/lab-8-pytorch-and-deep-learning-2-group-11/data'

train_folder_0 = os.path.join(data_folder, 'train', '0')

train_folder_1 = os.path.join(data_folder, 'train', '1')

test_folder_0 = os.path.join(data_folder, 'test', '0')

test_folder_1 = os.path.join(data_folder, 'test', '1')

 

# Create folders if they don't exist

os.makedirs(train_folder_0, exist_ok=True)

os.makedirs(train_folder_1, exist_ok=True)

os.makedirs(test_folder_0, exist_ok=True)

os.makedirs(test_folder_1, exist_ok=True)

 

# Counters for saving images

image_count_0 = 0

image_count_1 = 0

train_images_per_person = 50

total_images_per_person = 60

 

# Loop to capture images

while True:

    frame = picam2.capture_array()

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

 

    # Detect faces

    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

 

    if len(faces) == 0:

        print("No faces detected.")

    else:

        print(f"{len(faces)} face(s) detected.")

 

    # Process each detected face (only take the first face if multiple detected)

    for (x, y, w, h) in faces[:1]:

        # Ensure the crop stays within frame boundaries

        x = max(0, x)

        y = max(0, y)

        w = min(w, frame.shape[1] - x)

        h = min(h, frame.shape[0] - y)

       

        # Crop and resize the detected face

        face_crop = frame[y:y + h, x:x + w]

        if face_crop.size > 0:  # Ensure the crop is valid

            face_crop_resized = cv2.resize(face_crop, (64, 64))

 

            # Determine folder and image counter based on the person

            if image_count_0 < total_images_per_person:

                if image_count_0 < train_images_per_person:

                    folder_type = 'train'

                else:

                    folder_type = 'test'

                person_folder = '0'

                image_path = os.path.join(data_folder, folder_type, person_folder, f'face_{image_count_0}.jpg')

                image_count_0 += 1

            elif image_count_1 < total_images_per_person:

                if image_count_1 < train_images_per_person:

                    folder_type = 'train'

                else:

                    folder_type = 'test'

                person_folder = '1'

                image_path = os.path.join(data_folder, folder_type, person_folder, f'face_{image_count_1}.jpg')

                image_count_1 += 1

            else:

                print("Captured all required images.")

                picam2.stop()

                cv2.destroyAllWindows()

                exit()

 

            # Save the cropped face

            cv2.imwrite(image_path, face_crop_resized)

            print(f"Saved cropped face at {image_path}")

 

            # Display the cropped face and full frame for reference

            cv2.imshow('Cropped face', face_crop_resized)

            cv2.imshow('face detection', frame)

 

    # Quit on 'q' key press

    if cv2.waitKey(1) & 0xFF == ord('q'):

        break

 

# Cleanup

picam2.stop()

cv2.destroyAllWindows()