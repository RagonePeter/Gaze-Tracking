# This file follows set up information from the below tutorials
# https://www.youtube.com/watch?v=rAS17tDYeA0
# https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python
#
# TODO: Test w/ camera capturing multiple faces——Follow largest one? that would
#       work best for the demo scenario, rather than only working when one face
#       is visible.

import cv2
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python          # THIS IS MESSY CODE -- FIGURE OUT HOW
from mediapipe.tasks.python import vision   # TO SHORTEN NAME 'as' WITHOUT REIMPORTING
import numpy as np
import matplotlib.pyplot as plt
import wx

outp= wx.App(False)
width, height= wx.GetDisplaySize()
center = (int(width/2), int(height/2))

model_path = '/face_landmarker_v2_with_blendshapes.task'

mp_drawing = solutions.drawing_utils
mp_drawing_styles = solutions.drawing_styles

cap = cv2.VideoCapture(0)   # 0 for built in webcam

mp_face_mesh = solutions.face_mesh

lm_left_iris = tuple[0, 0, 0]
lm_right_iris = tuple[0, 0, 0]
lx = 0
ly = 0
lz = 0
rx = 0
ry = 0
rz = 0

left_iris_center = [0.4593483, 0.3455325, 0.007853137]
right_iris_center = [0.56650186, 0.3441249, 0.009465696]

def gaze_estimate(image):
    global width, height
    global center
    global lm_left_iris
    global lm_right_iris
    global left_iris_center
    global right_iris_center
    global lx
    global ly
    global lz
    global rx
    global ry
    global rz
    cx, cy = center
    radius = 5
    color = (255,0,0)
    thickness = 2

    print(left_iris_center[0] - lx)
    print(left_iris_center[1] - ly)
    #TODO: take into account movement of the head (to cancel out)
    #TODO: take into account the distance

    # Calculate differences using left eye
    x_diff = left_iris_center[0] - lx
    y_diff = left_iris_center[1] - ly

    # Calculate x/y_coords for eye_tracking
    x_coords = int(cx - x_diff*50000) # diff subtracted b/c image shown is mirrored?
    y_coords = int(cy - y_diff*50000) # is this the same as what we want?

    # Account for point going beyond possible coords on screen
    if(x_coords > width):
        x_coords = width-1
    elif(x_coords < 0):
        x_coords = 0
    #else:
    #    x_coords = x_coords

    if(y_coords > width):
        y_coords = width-1
    elif(y_coords < 0):
        y_coords = 0
    #else:
    #    y_coords = y_coords

    # determine left/right 
    if(x_coords > 3*width/4):
        print("looking left")
    elif(x_coords < width/4):
        print("looking right")
    else:
        print("center width")
    # calculate up/down
    if(y_coords > 3*height/4):
        print("looking down")
    elif(y_coords < height/4):
        print("looking up")
    else:
        print("center height")


    cv2.circle(image, (x_coords, y_coords), radius, color, thickness)

# Draws landmarks onto image
def draw_landmarks_on_image(image_rgb, detection_result):
    global lm_left_iris
    global lm_right_iris
    global lx
    global ly
    global lz
    global rx
    global ry
    global rz
    face_landmarks_list = detection_result.multi_face_landmarks
    annotated_image = np.copy(image_rgb)
    if face_landmarks_list:
        for face_landmarks in face_landmarks_list:
            # Draw the face mesh landmarks
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_tesselation_style()
            )
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_contours_style()
            )
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_iris_connections_style()
            )
            # from https://stackoverflow.com/questions/72891538/mediapipe-facemesh-irises-coordinates
            lm_left_iris = face_landmarks.landmark[468]
            lm_right_iris = face_landmarks.landmark[473]
            lx = lm_left_iris.x
            ly = lm_left_iris.y
            lz = lm_left_iris.z
            rx = lm_right_iris.x
            ry = lm_right_iris.y
            rz = lm_right_iris.z


            #print(lm_left_iris)     # (lm_left_iris.x, lm_left_iris.y, lm_left_iris.z) = face_landmarks.landmark[468]
            #print(lm_right_iris)     # (lm_right_iris.x, lm_right_iris.y, lm_right_iris.z) = face_landmarks.landmark[473]
    return annotated_image


with mp_face_mesh.FaceMesh(
    max_num_faces = 1,
    refine_landmarks = True,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5
) as face_mesh:

    while cap.isOpened():
        success, image = cap.read()  # Capture a frame
        if not success:
            print("Failed to capture frame. Exiting...")
            break   # TODO: Maybe this should be 'continue'?

        # Convert the BGR image to RGB as Mediapipe expects RGB input
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Pass the image to Mediapipe FaceMesh
        results = face_mesh.process(image_rgb)

        # annoted image, not image_rgb, so colors are correct
        annotated_image = draw_landmarks_on_image(image, results)

        gaze_estimate(annotated_image)

        # Display the frame with a mirrored effect
        cv2.imshow("My video capture", cv2.flip(annotated_image, 1))

        # Quit command. If you press 'q' for 100ms, then view breaks.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # center eyesight
        if cv2.waitKey(1) == ord('c'):
            left_iris_center = [lx, ly, lz]
            right_iris_center = [rx, ry, rz]

cap.release()
cv2.destroyAllWindows()