"""
@php_fans
"""

import math
import sys

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:
    print("MediaPipe o'rnatilmagan.")
    print("Avval virtual environment ichida mediapipe o'rnating.")
    sys.exit(1)

if not hasattr(mp, "solutions"):
    print("MediaPipe noto'g'ri ishlayapti.")
    print("mediapipe modulida 'solutions' mavjud emas.")
    print("Python 3.11 va mediapipe==0.10.14 ishlatish tavsiya qilinadi.")
    sys.exit(1)


mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils


MARKER_COLORS = [
    ("QIZIL", (0, 0, 255)),
    ("YASHIL", (0, 255, 0)),
    ("SARIQ", (0, 255, 255)),
    ("KO'K", (255, 0, 0)),
]

ERASE_COLOR = (0, 80, 255)
TEXT_COLOR = (230, 230, 230)
SCALE_COLOR = (255, 180, 60)

DRAW_THICKNESS = 6
ERASER_RADIUS = 45

MIN_SCALE = 0.35
MAX_SCALE = 2.50

COLOR_CHANGE_COOLDOWN_FRAMES = 20

MIN_STROKE_POINTS = 8
LINE_ERROR_THRESHOLD = 18
CIRCLE_ERROR_RATIO = 0.22
CIRCLE_CLOSE_RATIO = 0.45


def landmark_to_pixel(landmark, width, height):
    return int(landmark.x * width), int(landmark.y * height)


def distance(point_a, point_b):
    return math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])


def fingers_up(hand_landmarks):
    """
    Qaysi barmoqlar ko'tarilganini aniqlaydi.

    Natija:
    [thumb, index, middle, ring, pinky]
    """
    tips = [4, 8, 12, 16, 20]
    pips = [3, 6, 10, 14, 18]

    up = []

    if hand_landmarks.landmark[tips[0]].x < hand_landmarks.landmark[pips[0]].x:
        up.append(1)
    else:
        up.append(0)

    for i in range(1, 5):
        if hand_landmarks.landmark[tips[i]].y < hand_landmarks.landmark[pips[i]].y:
            up.append(1)
        else:
            up.append(0)

    return up


def overlay_canvas(frame, canvas):
    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_canvas, 10, 255, cv2.THRESH_BINARY)

    mask_inv = cv2.bitwise_not(mask)

    background = cv2.bitwise_and(frame, frame, mask=mask_inv)
    foreground = cv2.bitwise_and(canvas, canvas, mask=mask)

    return cv2.add(background, foreground)


def scale_canvas(canvas, scale_factor):
    height, width = canvas.shape[:2]

    matrix = cv2.getRotationMatrix2D(
        (width // 2, height // 2),
        0,
        scale_factor,
    )

    scaled = cv2.warpAffine(
        canvas,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    return scaled


def blur_faces_oval(frame, face_results):
    """
    FaceMesh landmarklari asosida faqat yuz konturi ichini blur qiladi.
    Qo'l yuz yonidan o'tsa, katta bounding box blur bo'lmaydi.
    """
    if not hasattr(face_results, "multi_face_landmarks"):
        return frame

    if not face_results.multi_face_landmarks:
        return frame

    height, width = frame.shape[:2]

    for face_landmarks in face_results.multi_face_landmarks:
        face_points = []

        for landmark in face_landmarks.landmark:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            face_points.append([x, y])

        face_points = np.array(face_points, dtype=np.int32)
        hull = cv2.convexHull(face_points)

        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, 255)

        kernel = np.ones((17, 17), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.GaussianBlur(mask, (41, 41), 0)

        blurred_frame = cv2.GaussianBlur(frame, (99, 99), 35)

        mask_3d = cv2.merge([mask, mask, mask]).astype(np.float32) / 255.0

        frame[:] = (
            blurred_frame * mask_3d + frame * (1.0 - mask_3d)
        ).astype(np.uint8)

    return frame


def draw_color_palette(frame, current_color_index):
    start_x = 20
    start_y = 165
    box_size = 34
    gap = 70

    cv2.putText(
        frame,
        "Marker ranglari:",
        (start_x, start_y - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        TEXT_COLOR,
        1,
        cv2.LINE_AA,
    )

    for i, (name, color) in enumerate(MARKER_COLORS):
        x1 = start_x + i * (box_size + gap)
        y1 = start_y
        x2 = x1 + box_size
        y2 = y1 + box_size

        border_color = (255, 255, 255) if i == current_color_index else (90, 90, 90)
        border_thickness = 3 if i == current_color_index else 1

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, border_thickness)

        cv2.putText(
            frame,
            name,
            (x1, y2 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            TEXT_COLOR,
            1,
            cv2.LINE_AA,
        )


def draw_hud(frame, mode, scale_value, current_color_index):
    panel_width = 930
    panel_height = 245

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_width, panel_height), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    current_color_name, current_color = MARKER_COLORS[current_color_index]

    if mode == "YOZISH":
        mode_color = current_color
    elif mode == "O'CHIRISH":
        mode_color = ERASE_COLOR
    elif mode == "KATTALASHTIRISH/KICHIKLASHTIRISH":
        mode_color = SCALE_COLOR
    elif mode == "RANG ALMASHTIRISH":
        mode_color = SCALE_COLOR
    else:
        mode_color = (170, 170, 170)

    cv2.putText(
        frame,
        f"Rejim: {mode}",
        (15, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        mode_color,
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        f"Tanlangan rang: {current_color_name}",
        (15, 67),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        current_color,
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        "1 qo'l: ko'rsatkich = yozish | bosh barmoq = o'chirish",
        (15, 97),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        TEXT_COLOR,
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        "2 qo'l: ikkala ko'rsatkich orasini ochib-yoping = kattalashtirish/kichiklashtirish",
        (15, 122),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (185, 185, 185),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        f"Musht qilib keyin 5 barmoqni oching = rang almashtirish | Scale: {scale_value:.2f}",
        (15, 147),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (185, 185, 185),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        "Chiziq va aylana avtomatik tekislanadi | Yuz konturi ichida blur",
        (15, 222),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (185, 185, 185),
        1,
        cv2.LINE_AA,
    )

    draw_color_palette(frame, current_color_index)


def draw_raw_stroke(target, points, color, thickness):
    if len(points) < 2:
        return

    for i in range(1, len(points)):
        cv2.line(
            target,
            points[i - 1],
            points[i],
            color,
            thickness,
            cv2.LINE_AA,
        )


def point_line_distance(point, start, end):
    px, py = point
    x1, y1 = start
    x2, y2 = end

    dx = x2 - x1
    dy = y2 - y1

    if dx == 0 and dy == 0:
        return distance(point, start)

    numerator = abs(dy * px - dx * py + x2 * y1 - y2 * x1)
    denominator = math.hypot(dx, dy)

    return numerator / denominator


def commit_smart_stroke(canvas, stroke_points, color):
    """
    Chizilgan shaklni tekshiradi:
    - Agar chiziqqa o'xshasa: to'g'ri chiziq qilib chizadi
    - Agar aylanaga o'xshasa: to'g'ri aylana qilib chizadi
    - Aks holda: oddiy chizilgan shaklda qoldiradi
    """
    if len(stroke_points) < MIN_STROKE_POINTS:
        draw_raw_stroke(canvas, stroke_points, color, DRAW_THICKNESS)
        return "ODDIY"

    points = np.array(stroke_points, dtype=np.float32)

    start = stroke_points[0]
    end = stroke_points[-1]
    stroke_width = max(1.0, float(np.max(points[:, 0]) - np.min(points[:, 0])))
    stroke_height = max(1.0, float(np.max(points[:, 1]) - np.min(points[:, 1])))
    stroke_size = max(stroke_width, stroke_height)

    start_end_distance = distance(start, end)

    line_errors = [
        point_line_distance((int(point[0]), int(point[1])), start, end)
        for point in points
    ]
    average_line_error = float(np.mean(line_errors))

    if start_end_distance > 80 and average_line_error < LINE_ERROR_THRESHOLD:
        cv2.line(
            canvas,
            start,
            end,
            color,
            DRAW_THICKNESS,
            cv2.LINE_AA,
        )
        return "TEKIS CHIZIQ"

    center_x = int(np.mean(points[:, 0]))
    center_y = int(np.mean(points[:, 1]))
    center = (center_x, center_y)

    distances_to_center = np.sqrt(
        (points[:, 0] - center_x) ** 2 + (points[:, 1] - center_y) ** 2
    )

    radius = float(np.mean(distances_to_center))

    if radius > 20:
        radius_error = float(np.std(distances_to_center) / radius)
        is_closed = start_end_distance < radius * CIRCLE_CLOSE_RATIO

        if is_closed and radius_error < CIRCLE_ERROR_RATIO and stroke_size > 45:
            cv2.circle(
                canvas,
                center,
                int(radius),
                color,
                DRAW_THICKNESS,
                cv2.LINE_AA,
            )
            return "TEKIS AYLANA"

    draw_raw_stroke(canvas, stroke_points, color, DRAW_THICKNESS)
    return "ODDIY"


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Kamera ochilmadi. Kamera ruxsatlarini tekshiring.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    ret, frame = cap.read()

    if not ret:
        print("Kameradan tasvir olinmadi.")
        cap.release()
        return

    frame = cv2.flip(frame, 1)
    height, width = frame.shape[:2]

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    preview_canvas = np.zeros((height, width, 3), dtype=np.uint8)

    prev_point = None
    prev_two_hand_distance = None
    current_scale = 1.0

    current_color_index = 0
    color_change_cooldown = 0
    previous_hand_was_fist = False

    stroke_points = []
    is_currently_drawing = False
    last_shape_status = ""

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    ) as hands, mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as face_mesh:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Kamera bilan aloqa uzildi.")
                break

            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            hands_results = hands.process(rgb)
            face_results = face_mesh.process(rgb)

            mode = "TAYYOR"
            index_points = []
            thumb_point = None

            preview_canvas[:] = 0

            if color_change_cooldown > 0:
                color_change_cooldown -= 1

            if hands_results.multi_hand_landmarks:
                hand_count = len(hands_results.multi_hand_landmarks)

                for hand_landmarks in hands_results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_draw.DrawingSpec(color=(80, 200, 255), thickness=2, circle_radius=4),
                        mp_draw.DrawingSpec(color=(255, 200, 80), thickness=2),
                    )

                    index_point = landmark_to_pixel(
                        hand_landmarks.landmark[8],
                        width,
                        height,
                    )
                    index_points.append(index_point)

                if hand_count >= 2 and len(index_points) >= 2:
                    if is_currently_drawing and stroke_points:
                        selected_color = MARKER_COLORS[current_color_index][1]
                        last_shape_status = commit_smart_stroke(canvas, stroke_points, selected_color)

                    is_currently_drawing = False
                    stroke_points = []

                    mode = "KATTALASHTIRISH/KICHIKLASHTIRISH"
                    prev_point = None

                    point_a = index_points[0]
                    point_b = index_points[1]

                    current_distance = distance(point_a, point_b)

                    if prev_two_hand_distance is not None and prev_two_hand_distance > 0:
                        scale_factor = current_distance / prev_two_hand_distance

                        if 0.90 <= scale_factor <= 1.10:
                            new_scale = current_scale * scale_factor

                            if MIN_SCALE <= new_scale <= MAX_SCALE:
                                canvas = scale_canvas(canvas, scale_factor)
                                current_scale = new_scale

                    prev_two_hand_distance = current_distance

                    cv2.line(frame, point_a, point_b, SCALE_COLOR, 3, cv2.LINE_AA)
                    cv2.circle(frame, point_a, 12, SCALE_COLOR, 2, cv2.LINE_AA)
                    cv2.circle(frame, point_b, 12, SCALE_COLOR, 2, cv2.LINE_AA)

                else:
                    prev_two_hand_distance = None

                    hand_landmarks = hands_results.multi_hand_landmarks[0]
                    up = fingers_up(hand_landmarks)

                    index_point = landmark_to_pixel(
                        hand_landmarks.landmark[8],
                        width,
                        height,
                    )

                    thumb_point = landmark_to_pixel(
                        hand_landmarks.landmark[4],
                        width,
                        height,
                    )

                    finger_count = sum(up)
                    is_fist = finger_count <= 1
                    all_fingers_open = finger_count == 5

                    if previous_hand_was_fist and all_fingers_open and color_change_cooldown == 0:
                        if is_currently_drawing and stroke_points:
                            selected_color = MARKER_COLORS[current_color_index][1]
                            last_shape_status = commit_smart_stroke(canvas, stroke_points, selected_color)

                        is_currently_drawing = False
                        stroke_points = []
                        prev_point = None

                        current_color_index = (current_color_index + 1) % len(MARKER_COLORS)
                        color_change_cooldown = COLOR_CHANGE_COOLDOWN_FRAMES

                        color_name, _ = MARKER_COLORS[current_color_index]
                        print(f"Marker rangi o'zgardi: {color_name}")

                    previous_hand_was_fist = is_fist

                    is_color_change_pose = all_fingers_open
                    is_erasing = up[0] == 1 and up[1] == 0 and not is_color_change_pose
                    is_drawing = up[1] == 1 and not is_erasing and not is_color_change_pose

                    if is_color_change_pose:
                        mode = "RANG ALMASHTIRISH"
                        prev_point = None

                        cv2.putText(
                            frame,
                            "Rang almashtirish: musht qilib keyin 5 barmoqni oching",
                            (20, height - 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            SCALE_COLOR,
                            2,
                            cv2.LINE_AA,
                        )

                    elif is_erasing:
                        if is_currently_drawing and stroke_points:
                            selected_color = MARKER_COLORS[current_color_index][1]
                            last_shape_status = commit_smart_stroke(canvas, stroke_points, selected_color)

                        is_currently_drawing = False
                        stroke_points = []

                        mode = "O'CHIRISH"

                        cv2.circle(
                            canvas,
                            thumb_point,
                            ERASER_RADIUS,
                            (0, 0, 0),
                            -1,
                            cv2.LINE_AA,
                        )

                        prev_point = None

                        cv2.circle(
                            frame,
                            thumb_point,
                            ERASER_RADIUS,
                            ERASE_COLOR,
                            2,
                            cv2.LINE_AA,
                        )

                    elif is_drawing:
                        mode = "YOZISH"

                        selected_color = MARKER_COLORS[current_color_index][1]
                        is_currently_drawing = True
                        stroke_points.append(index_point)

                        if prev_point is not None:
                            cv2.line(
                                preview_canvas,
                                prev_point,
                                index_point,
                                selected_color,
                                DRAW_THICKNESS,
                                cv2.LINE_AA,
                            )

                        draw_raw_stroke(preview_canvas, stroke_points, selected_color, DRAW_THICKNESS)
                        prev_point = index_point

                        cv2.circle(
                            frame,
                            index_point,
                            12,
                            selected_color,
                            2,
                            cv2.LINE_AA,
                        )

                    else:
                        if is_currently_drawing and stroke_points:
                            selected_color = MARKER_COLORS[current_color_index][1]
                            last_shape_status = commit_smart_stroke(canvas, stroke_points, selected_color)

                        is_currently_drawing = False
                        stroke_points = []
                        prev_point = None

            else:
                if is_currently_drawing and stroke_points:
                    selected_color = MARKER_COLORS[current_color_index][1]
                    last_shape_status = commit_smart_stroke(canvas, stroke_points, selected_color)

                is_currently_drawing = False
                stroke_points = []
                prev_point = None
                prev_two_hand_distance = None
                previous_hand_was_fist = False

            frame = blur_faces_oval(frame, face_results)

            visible_canvas = cv2.add(canvas, preview_canvas)
            combined = overlay_canvas(frame, visible_canvas)

            draw_hud(combined, mode, current_scale, current_color_index)

            selected_color = MARKER_COLORS[current_color_index][1]

            for point in index_points:
                cv2.circle(
                    combined,
                    point,
                    7,
                    selected_color,
                    -1,
                    cv2.LINE_AA,
                )

            if thumb_point is not None:
                cv2.circle(combined, thumb_point, 7, ERASE_COLOR, -1, cv2.LINE_AA)

            if last_shape_status:
                cv2.putText(
                    combined,
                    f"Oxirgi shakl: {last_shape_status}",
                    (20, height - 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    SCALE_COLOR,
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("Barmoq bilan yozish", combined)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                break

            if key == ord("c"):
                canvas[:] = 0
                preview_canvas[:] = 0
                prev_point = None
                prev_two_hand_distance = None
                current_scale = 1.0
                color_change_cooldown = 0
                previous_hand_was_fist = False
                stroke_points = []
                is_currently_drawing = False
                last_shape_status = ""

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
