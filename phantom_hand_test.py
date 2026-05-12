# examples/basic_demo.py
import cv2
from phantom_hand import PhantomHandTracker

def main():
    tracker = PhantomHandTracker(screen_dim=(1280, 720), model_path="models/hand_landmarker.task", debug=True)
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        # frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        data = tracker.update(frame)

        # Draw landmarks
        for hand in ("LEFT", "RIGHT"):
            pts = data["POSITION_DATA"][hand]
            is_ghost = data["FRAME_TYPE"][hand] == "GHOST"
            color = (255, 0, 0) if is_ghost else (0, 255, 0)  # Blue ghost, green real

            for x, y, _ in pts:
                px = int(x * frame.shape[1])
                py = int(y * frame.shape[0])
                cv2.circle(frame, (px, py), 5, color, -1)

        cv2.imshow("GhostHandTracker Demo", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    tracker.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()