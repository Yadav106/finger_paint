from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import pyautogui

app = Flask(__name__)

screen_width, screen_height = pyautogui.size()
is_clicking = False

print(f"screenwidth: {screen_width}, screenheight: {screen_height}")

def draw_label(img, text, pos, bg_color):
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, cv2.FILLED)
    
    end_x = pos[0] + text_size[0][0] + 2
    end_y = pos[1] + text_size[0][1] - 2
    
    
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def generate_frames():
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            is_clicking = False
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )

                    print((hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y))

                    # for whole screen
                    # x_coord = hand_landmarks.landmark[8].x * 1440
                    # y_coord = hand_landmarks.landmark[8].y * 900

                    x_coord = 823 + (hand_landmarks.landmark[8].x * (1410 - 823))
                    y_coord = 495 + (hand_landmarks.landmark[8].y * (771 - 495))

                    pyautogui.moveTo(x_coord, y_coord)

                    thumb_tip = hand_landmarks.landmark[4]
                    thumb_mcp = hand_landmarks.landmark[2]
                    distance_thumb = ((thumb_tip.x - thumb_mcp.x) ** 2 + (thumb_tip.y - thumb_mcp.y) ** 2) ** 0.5

                    print(distance_thumb)

                    if distance_thumb > 0.1:
                        draw_label(frame, "Holding Click", (50, 50), (255, 0, 0))
                        if not is_clicking:
                            pyautogui.mouseDown()
                            is_clicking = True
                    else:
                        pyautogui.mouseUp()
                        if is_clicking:
                            is_clicking = False


            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
