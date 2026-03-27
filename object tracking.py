"""Hand Gesture Drawing with MediaPipe (Hands / Tasks).

Uses hand landmarks (21 points) to detect:
- Index finger tip position = DRAW
- Open palm (all fingers spread) = ERASE
- If Index finger tip touches Thumb finger tip = colour change (toggle drawing color)

NO color tracking. Pure hand landmark detection.

Press 'q' to quit, 'c' to clear.
"""

import sys
import cv2
import numpy as np
import time

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

import os
import urllib.request

MODEL_PATH = "hand_landmarker.task"

# Automatically download the model if it doesn't exist
if not os.path.exists(MODEL_PATH):
    print("[INFO] Downloading hand_landmarker.task...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("[INFO] Download complete.")

try:
    hand_options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.5,
    )
    hand_landmarker = HandLandmarker.create_from_options(hand_options)
    print("[INFO] MediaPipe Tasks HandLandmarker initialized.")
except Exception as e:
    print(f"[ERROR] Could not initialize MediaPipe Tasks HandLandmarker: {e}")
    sys.exit(1)


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Camera not accessible!")
    sys.exit(1)

# Set camera resolution for smoother drawing (Lower resolution = Higher FPS)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

points = []
canvas = None
drawing = False  # Track if we're currently drawing
frame_h, frame_w = None, None

# Variables for FPS calculation
prev_frame_time = 0
new_frame_time = 0

colors = [
    (255, 0, 0),    # Blue
    (0, 255, 0),    # Green
    (0, 0, 255),    # Red
    (0, 255, 255),  # Yellow
]
current_color_index = 0
brush_thickness = 5
snap_active = False  # Track finger snap state to toggle color once per snap

# Initialize CLAHE for lighting robustness
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

print("[INFO] ✓ Hand Gesture Drawing Started!")
print("[INFO] Controls:")
print("  • Index finger extended = DRAW (continuous line, current color)")
#print("  • Closed fist = BREAK current line (keeps previous drawing)")
print("  • Extend index again = START new line")
print("  • Open palm (all fingers spread) = ERASE (red circle from palm center)")
print("  • Thumb tip touching Index tip = CHANGE drawing color")
print("  • 'c' = Clear entire canvas")
print("  • 'q' = Quit")
print("")


def get_landmarks_pixel(hand_landmarks, frame_height, frame_width):
    """Extract (x, y) pixel coordinates from hand landmarks.

    Expects an iterable of 21 normalized landmarks (as returned by
    MediaPipe Tasks HandLandmarker).
    """
    landmarks = []
    for landmark in hand_landmarks:
        x = int(landmark.x * frame_width)
        y = int(landmark.y * frame_height)
        landmarks.append((x, y))
    return landmarks


def is_index_extended(landmarks):
    """Check if index finger is extended (any direction, not just pointing down)."""
    index_tip = np.array(landmarks[8])
    index_mid = np.array(landmarks[6])
    ref_dist = np.linalg.norm(np.array(landmarks[0]) - np.array(landmarks[9]))

    dist = np.linalg.norm(index_tip - index_mid)
    return dist > ref_dist * 0.4


def is_palm_open(landmarks):
    """Check if palm is strictly open and all fingers are extended outwards."""
    wrist = np.array(landmarks[0])
    
    # For each finger, tip must be further from wrist than PIP joint
    # Thumb: 4 vs 3
    # Index: 8 vs 6
    # Middle: 12 vs 10
    # Ring: 16 vs 14
    # Pinky: 20 vs 18
    tips = [4, 8, 12, 16, 20]
    pips = [3, 6, 10, 14, 18]
    
    for tip, pip in zip(tips, pips):
        d_tip = np.linalg.norm(np.array(landmarks[tip]) - wrist)
        d_pip = np.linalg.norm(np.array(landmarks[pip]) - wrist)
        if d_tip <= d_pip:
            return False
            
    # Check if fingers are spread apart by measuring thumb to pinky span
    thumb_tip = np.array(landmarks[4])
    pinky_tip = np.array(landmarks[20])
    
    # Reference distance: wrist (0) to middle finger base (9)
    ref_dist = np.linalg.norm(wrist - np.array(landmarks[9]))
    
    # Span should be large for a fully open and spread palm
    span = np.linalg.norm(thumb_tip - pinky_tip)
    if span < ref_dist * 1.2:
        return False
        
    return True


def is_finger_snap(landmarks):
    """Check if thumb and index finger are close together (snap/pinch gesture)."""
    ref_dist = np.linalg.norm(np.array(landmarks[0]) - np.array(landmarks[9]))
    thumb_tip = np.array(landmarks[4])
    index_tip = np.array(landmarks[8])
    
    distance = np.linalg.norm(thumb_tip - index_tip)
    return distance < ref_dist * 0.15


def get_palm_center(landmarks):
    """Get the center of the palm for erasing."""
    # Use landmarks 0 (wrist), 5 (index base), 9 (middle base), 13 (ring base), 17 (pinky base)
    palm_points = [landmarks[0], landmarks[5], landmarks[9], landmarks[13], landmarks[17]]
    palm_x = sum(x for x, y in palm_points) // len(palm_points)
    palm_y = sum(y for x, y in palm_points) // len(palm_points)
    return (palm_x, palm_y)


def get_drawing_point(landmarks):
    """Get the point to draw from (index fingertip)."""
    return landmarks[8]  # Index fingertip


# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to read frame")
        break
    
    # Calculate FPS
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time + 1e-5) # Add small epsilon to prevent div by zero
    prev_frame_time = new_frame_time
    
    frame = cv2.flip(frame, 1)
    frame_h, frame_w = frame.shape[:2]
    
    # Initialize canvas on first frame
    if canvas is None:
        canvas = np.ones_like(frame) * 255
    
    # Convert frame to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # --- LIGHTING ROBUSTNESS (CLAHE) ---
    # We apply CLAHE to a copy of the frame to help MediaPipe "see" better
    # without affecting the visual output of the canvas.
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_enhanced = clahe.apply(l)
    enhanced_lab = cv2.merge((l_enhanced, a, b))
    enhanced_frame_rgb = cv2.cvtColor(cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2RGB)

    # Wrap frame for MediaPipe Tasks (using the ENHANCED frame for better tracking)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=enhanced_frame_rgb)
    
    # Blend ORIGINAL frame with canvas for the display output
    output = cv2.addWeighted(frame, 0.3, canvas, 0.7, 0)
    timestamp_ms = int(time.time() * 1000)
    results = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
    
    if results and results.hand_landmarks:
        hand_landmarks = results.hand_landmarks[0]
        landmarks = get_landmarks_pixel(hand_landmarks, frame_h, frame_w)
        
        # Check gestures first to decide if we need to draw the skeleton
        index_extended = is_index_extended(landmarks)
        palm_open = is_palm_open(landmarks)
        snap_now = is_finger_snap(landmarks)
        
        # Determine if fist is closed (not index extended and not palm open)
        fist_closed = not index_extended and not palm_open
        
        # Only draw the complex hand skeleton if we aren't actively drawing or erasing
        if fist_closed:
            for i, (x, y) in enumerate(landmarks):
                cv2.circle(output, (x, y), 4, (0, 255, 0), -1)
            
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
                (0, 5), (5, 6), (6, 7), (7, 8),  # Index
                (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
                (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
                (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
                (5, 9), (9, 13), (13, 17)  # Palm connections
            ]
            for start, end in connections:
                cv2.line(output, landmarks[start], landmarks[end], (0, 255, 0), 2)
        else:
            # If tracking/drawing, only show the active point
            if index_extended:
                draw_point = get_drawing_point(landmarks)
                cv2.circle(output, draw_point, 8, (255, 0, 0), -1)
            elif palm_open:
                erase_point = get_palm_center(landmarks)
                cv2.circle(output, erase_point, 30, (0, 0, 255), 2)

        # Finger snap: toggle color only on transition from not snapped -> snapped
        if snap_now and not snap_active:
            current_color_index = (current_color_index + 1) % len(colors)
            print(f"[INFO] Drawing color changed. Current color index: {current_color_index}")
        snap_active = snap_now
        
        # Drawing logic
        if palm_open:
            # Erasing with palm center
            erase_point = get_palm_center(landmarks)
            cv2.circle(canvas, erase_point, 30, (255, 255, 255), -1)
            cv2.circle(output, erase_point, 30, (0, 0, 255), 2)
            points = []  # Clear any current drawing points
            drawing = False
            
        elif index_extended:
            # Start or continue drawing
            draw_point = get_drawing_point(landmarks)
            points.append(draw_point)
            
            # Draw on canvas
            draw_color = colors[current_color_index]
            if len(points) > 1:
                cv2.line(canvas, points[-2], points[-1], draw_color, brush_thickness)
            cv2.circle(output, draw_point, 8, draw_color, -1)
            drawing = True
            
        elif fist_closed:
            # Break the current drawing line
            points = []  # Clear current line points
            drawing = False
            
    
    # Draw FPS on screen
    fps_text = f"FPS: {int(fps)}"
    cv2.putText(output, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 2, cv2.LINE_AA)
    
    # Display
    cv2.imshow("Hand Gesture Drawing (No Color Tracking)", output)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas = np.ones_like(frame) * 255
        points = []

cap.release()
cv2.destroyAllWindows()
hand_landmarker.close()  # explicitly close to prevent the interpreter shutdown crash
print("[INFO] Application closed.")

# --- Plot the final marking on a graph using matplotlib ---
if canvas is not None:
    print("[INFO] Generating final drawing plot...")
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        # Convert BGR (OpenCV) to RGB (Matplotlib)
        canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        
        # Create the matplotlib figure
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.imshow(canvas_rgb)
        ax.set_title("Final Hand Gesture Drawing")
        
        # Add labels and a grid
        ax.set_xlabel("X-axis (Pixels)")
        ax.set_ylabel("Y-axis (Pixels)")
        ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

        # Handwriting analysis using EasyOCR
        try:
            try:
                import easyocr
            except ImportError:
                print("[INFO] easyocr is not installed. Installing it now (this may take a few minutes)...")
                import subprocess
                import sys
                subprocess.check_call([sys.executable, "-m", "pip", "install", "easyocr", "--quiet"])
                import easyocr
            print("[INFO] Initializing EasyOCR (This might take a moment)...")
            reader = easyocr.Reader(['en'], gpu=False) 
            # NLP Enhancement: Initialize spell checker and TextBlob
            try:
                from spellchecker import SpellChecker
                spell = SpellChecker()
            except ImportError:
                print("[INFO] pyspellchecker is not installed. Installing it now...")
                import subprocess
                import sys
                subprocess.check_call([sys.executable, "-m", "pip", "install", "pyspellchecker", "--quiet"])
                from spellchecker import SpellChecker
                spell = SpellChecker()

            try:
                from textblob import TextBlob
            except ImportError:
                print("[INFO] textblob is not installed. Installing it now...")
                import subprocess
                import sys
                subprocess.check_call([sys.executable, "-m", "pip", "install", "textblob", "--quiet"])
                subprocess.check_call([sys.executable, "-m", "textblob.download_corpora", "lite", "--quiet"])
                from textblob import TextBlob

            try:
                from transformers import pipeline, TrOCRProcessor, VisionEncoderDecoderModel
                from PIL import Image
                print("[INFO] Loading advanced Hybrid OCR Engine (TrOCR + DistilBERT)...")
                # Load TrOCR for handwriting recognition
                trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
                trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
                # Load DistilBERT for semantic correction
                bert_fixer = pipeline("fill-mask", model="distilbert-base-uncased")
            except ImportError:
                print("[INFO] transformers, torch, or Pillow is not installed. Installing them now...")
                import subprocess
                import sys
                subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers", "torch", "Pillow", "--quiet"])
                from transformers import pipeline, TrOCRProcessor, VisionEncoderDecoderModel
                from PIL import Image
                trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
                trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
                bert_fixer = pipeline("fill-mask", model="distilbert-base-uncased")
            except Exception as e:
                print(f"[WARNING] Could not initialize transformers: {e}")
                bert_fixer = None

            print("[INFO] Analyzing handwriting on canvas with enhanced preprocessing...")
            
            # IMPROVED PREPROCESSING: High-contrast mask for light colors (like yellow)
            # Instead of standard grayscale, we take the minimum across R, G, B channels.
            # Why: Yellow (255,255,0) becomes 0 in the min-channel, creating perfect contrast against white (255,255,255).
            gray = np.min(canvas_rgb, axis=2)
            
            # Thresholding to isolate text (binary inverse because drawing is on white)
            _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
            # Morphological operations to thicken thin lines
            kernel = np.ones((2,2), np.uint8)
            processed_img = cv2.dilate(thresh, kernel, iterations=1)
            # Invert back to black-on-white for OCR
            final_ocr_img = cv2.bitwise_not(processed_img)

            # Read text from the processed image
            ocr_results = reader.readtext(final_ocr_img)
            
            if ocr_results:
                print("\n--- HYBRID OCR RESULTS ---")
                for (bbox, detected_raw_text, prob) in ocr_results:
                    # Strategy 0: Advanced Recognition with TrOCR
                    # Crop the word from the original canvas for better transformer features
                    # bbox: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    ymin, xmin = int(min(p[1] for p in bbox)), int(min(p[0] for p in bbox))
                    ymax, xmax = int(max(p[1] for p in bbox)), int(max(p[0] for p in bbox))
                    # Add small padding
                    h, w = canvas_rgb.shape[:2]
                    ymin, xmin = max(0, ymin-5), max(0, xmin-5)
                    ymax, xmax = min(h, ymax+5), min(w, xmax+5)
                    
                    word_crop = canvas_rgb[ymin:ymax, xmin:xmax]
                    if word_crop.size == 0:
                        text = detected_raw_text
                    else:
                        # Convert to PIL and pass through TrOCR
                        pil_img = Image.fromarray(word_crop).convert("RGB")
                        pixel_values = trocr_processor(pil_img, return_tensors="pt").pixel_values
                        generated_ids = trocr_model.generate(pixel_values)
                        text = trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                        print(f"TrOCR Recognized: '{text}' (EasyOCR detection: '{detected_raw_text}')")

                    # Strategy 1: Word-by-word correction (pyspellchecker)
                    import re
                    words = text.split()
                    spell_corrected_words = []
                    for word in words:
                        clean_word = re.sub(r'\W+', '', word).lower()
                        if clean_word:
                            correction = spell.correction(clean_word)
                            spell_corrected_words.append(correction if correction else word)
                        else:
                            spell_corrected_words.append(word)
                    spell_corrected_text = " ".join(spell_corrected_words)

                    # Strategy 2: Context-aware correction (TextBlob)
                    blob = TextBlob(text)
                    textblob_corrected = str(blob.correct())
                    
                    # Strategy 3: Semantic Correction (BERT)
                    # We only attempt this if the sentence has more than 2 words for better context
                    bert_corrected_text = text
                    if len(words) > 2:
                        try:
                            # Replace each word with [MASK] and see if BERT suggests a better fit
                            temp_words = list(words)
                            for i in range(len(temp_words)):
                                masked_sentence = " ".join([temp_words[j] if j != i else "[MASK]" for j in range(len(temp_words))])
                                predictions = bert_fixer(masked_sentence)
                                # If the top prediction is different and has reasonable score, use it
                                if predictions[0]['score'] > 0.4:
                                    temp_words[i] = predictions[0]['token_str']
                            bert_corrected_text = " ".join(temp_words)
                        except Exception as e:
                            print(f"[WARNING] BERT correction failed: {e}")

                    # Choose the best prediction (Heuristic: BERT > TextBlob > SpellChecker)
                    # We prefer BERT if it changed the sentence and the original text was "broken"
                    if bert_corrected_text.lower() != text.lower():
                        final_prediction = bert_corrected_text
                    elif textblob_corrected.lower() != text.lower():
                        final_prediction = textblob_corrected
                    else:
                        final_prediction = spell_corrected_text
                    
                    print(f"Detected: '{text}' (Confidence: {prob:.2f})")
                    if final_prediction.lower() != text.lower():
                        print(f"NLP Predicted (Context-Aware): '{final_prediction}'")
                    
                    # Draw bounding box on the matplotlib plot
                    polygon = patches.Polygon(bbox, closed=True, linewidth=2, edgecolor='red', facecolor='none')
                    ax.add_patch(polygon)
                    
                    # Show original and corrected if they differ
                    display_text = f"{text}\n({final_prediction})" if final_prediction.lower() != text.lower() else text
                    top_left = bbox[0]
                    ax.text(top_left[0], top_left[1] - 10, display_text, color='red', fontsize=10, fontweight='bold',
                            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))
                print("-------------------\n")
            else:
                print("[INFO] No readable text found by OCR.")
                
        except ImportError:
            print("[WARNING] easyocr is not installed. Skipping handwriting analysis.")
        except Exception as e:
            print(f"[ERROR] OCR failed: {e}")
        
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("[WARNING] matplotlib is not installed. Plotting skipped.")