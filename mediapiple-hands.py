import pickle
import sys
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
from sklearn.preprocessing import StandardScaler


# For static images:
# IMAGE_FILES = []
# with mp_hands.Hands(
#     static_image_mode=True,
#     max_num_hands=2,
#     min_detection_confidence=0.5) as hands:
#   for idx, file in enumerate(IMAGE_FILES):
#     # Read an image, flip it around y-axis for correct handedness output (see
#     # above).
#     image = cv2.flip(cv2.imread(file), 1)
#     # Convert the BGR image to RGB before processing.
#     results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

#     # Print handedness and draw hand landmarks on the image.
#     print('Handedness:', results.multi_handedness)
#     if not results.multi_hand_landmarks:
#       continue
#     image_height, image_width, _ = image.shape
#     annotated_image = image.copy()
#     for hand_landmarks in results.multi_hand_landmarks:
#       print('hand_landmarks:', hand_landmarks)
#       print(
#           f'Index finger tip coordinates: (',
#           f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
#           f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
#       )
#       mp_drawing.draw_landmarks(
#           annotated_image,
#           hand_landmarks,
#           mp_hands.HAND_CONNECTIONS,
#           mp_drawing_styles.get_default_hand_landmarks_style(),
#           mp_drawing_styles.get_default_hand_connections_style())
#     cv2.imwrite(
#         '/tmp/annotated_image' + str(idx) + '.png', cv2.flip(annotated_image, 1))
#     # Draw hand world landmarks.
#     if not results.multi_hand_world_landmarks:
#       continue
#     for hand_world_landmarks in results.multi_hand_world_landmarks:
#       mp_drawing.plot_landmarks(
#         hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # if results.multi_hand_world_landmarks:
      # print(results.multi_hand_world_landmarks)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
        # print(type(hand_landmarks))
        # print(hand_landmarks)
        # print(type(hand_landmarks.landmark[0]))
        # print(hand_landmarks.landmark[0].x)
        # print(mp_hands.HandLandmark.WRIST) 
        # print(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x)

        hand = hand_landmarks.landmark
        # print(hand)
        hand_row = np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand]).flatten()
        # hand_row.reshape(1,63)
        # row = hand_row
        # print(row)

        # print(hand_landmarks)
        # print(hand_landmarks.landmark)

        

        feature = pd.DataFrame(
          data=[hand_row],
          columns=[
            'x00',	'y00',	'z00',	'x01',	'y01',	'z01',	'x02',	'y02',	'z02',	'x03',	'y03',	'z03',	'x04',	'y04',	'z04',	'x05',	'y05',	'z05',	'x06',	'y06',	'z06',	'x07',	'y07',	'z07',	'x08',	'y08',	'z08',	'x09',	'y09',	'z09',	'x10',	'y10',	'z10',	'x11',	'y11',	'z11',	'x12',	'y12',	'z12',	'x13',	'y13',	'z13',	'x14',	'y14',	'z14',	'x15',	'y15',	'z15',	'x16',	'y16',	'z16',	'x17',	'y17',	'z17',	'x18',	'y18',	'z18',	'x19',	'y19',	'z19',	'x20',	'y20',	'z20'
          ]
        )
        # print(feature)

        features_np = np.array(feature)
        print(features_np)

        # np.set_printoptions(threshold=sys.maxsize)
        sc_X = StandardScaler()
        x = sc_X.fit_transform(features_np)
        x = sc_X.transform(features_np.reshape(1,63))
        # print(x)

        # with open('sign-langugage/KNN_Model.pkl', 'rb') as f:
        with open('sign-langugage/SVML_Model.pkl', 'rb') as f:
          model = pickle.load(f)

        y = model.predict(features_np)
        print(y)
        # cv2.putText(image,str(y),(95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image,str(y[0]), (100,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
        # cv2.imshow("Frame",image)
        # input()


        
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
