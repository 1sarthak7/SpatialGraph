import cv2
import mediapipe as mp
import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

# ------------------ MediaPipe Setup ------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ------------------ OpenCV ------------------
cap = cv2.VideoCapture(0)

# ------------------ Pygame + OpenGL ------------------
pygame.init()
display = (1000, 700)
pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

glClearColor(0.05, 0.05, 0.08, 1)
gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
glTranslatef(0.0, 0.0, -6)

glEnable(GL_DEPTH_TEST)

drawing_points = []
current_color = [1.0, 0.0, 0.0]

def draw_lines():
    if len(drawing_points) < 2:
        return
    glLineWidth(3)
    glBegin(GL_LINE_STRIP)
    glColor3f(*current_color)
    for point in drawing_points:
        glVertex3f(point[0], point[1], point[2])
    glEnd()

def get_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# ------------------ MAIN LOOP ------------------
running = True

while running:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            index_tip = hand_landmarks.landmark[8]
            thumb_tip = hand_landmarks.landmark[4]
            middle_joint = hand_landmarks.landmark[6]

            x = (index_tip.x - 0.5) * 6
            y = -(index_tip.y - 0.5) * 4
            z = -index_tip.z * 8

            distance = get_distance(
                [index_tip.x, index_tip.y],
                [thumb_tip.x, thumb_tip.y]
            )

            # Draw when index finger extended
            if index_tip.y < middle_joint.y:
                drawing_points.append([x, y, z])

            # Pinch to randomize color
            if distance < 0.04:
                current_color = np.random.rand(3).tolist()

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    draw_lines()
    pygame.display.flip()
    pygame.time.wait(10)

cap.release()
pygame.quit()
