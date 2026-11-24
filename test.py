from collections import deque
import cv2
import numpy as np
import tensorflow as tf

# 1. تحميل نموذج الـ TFLite
interpreter = tf.lite.Interpreter(model_path="plastic_sorting_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_names = ["HDPE", "PET", "PP", "others"]

# 2. فتح كاميرا الـ AUKEY الخارجية
WEBCAM_INDEX = 1
cap = cv2.VideoCapture(WEBCAM_INDEX, cv2.CAP_DSHOW)

if not cap.isOpened():
  print("لم يتم العثور على الكاميرا! تأكدي من توصيل كابل الـ USB.")
  exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# إعدادات المعدل المتحرك والثبات
WINDOW_SIZE = 10
MIN_CONFIDENCE = 75.0
STABLE_FRAMES_NEEDED = 10
PET_MIN_CONFIDENCE = 35.0

prob_history = deque(maxlen=WINDOW_SIZE)
consecutive_confirmed = 0
last_candidate = ""
locked_result = ""
locked_timer = 0
LOCK_DISPLAY_FRAMES = 30

pet_idx = class_names.index("PET")
hdpe_idx = class_names.index("HDPE")
pp_idx = class_names.index("PP")

print(
    "النظام يعمل تلقائياً مع أولوية PET ومعدل متحرك... ('q' للخروج)"
)

while cap.isOpened():
  ret, frame = cap.read()
  if not ret:
    break

  display_frame = frame.copy()

  # تجهيز الصورة للموديل
  img = cv2.resize(frame, (224, 224))
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img = img.astype(np.float32)
  img = np.expand_dims(img, axis=0)

  interpreter.set_tensor(input_details[0]["index"], img)
  interpreter.invoke()

  output_data = interpreter.get_tensor(output_details[0]["index"])[0] * 100.0
  prob_history.append(output_data)

  # حساب المتوسط المتحرك لكل فئة
  avg_probs = np.mean(prob_history, axis=0)

  # طباعة المتوسطات في الـ Terminal
  print("--- متوسط آخر إطارات ---")
  for i, name in enumerate(class_names):
    print(f"  {name}: {avg_probs[i]:.1f}%")

  sorted_indices = np.argsort(avg_probs)[::-1]
  top1_idx = sorted_indices[0]
  top2_idx = sorted_indices[1]

  top1_class = class_names[top1_idx]
  top1_conf = avg_probs[top1_idx]
  top2_class = class_names[top2_idx]
  top2_conf = avg_probs[top2_idx]

  # أولوية لـ PET إذا كانت نسبته معقولة وأعلى من HDPE و PP رغم تأثير الشفافية/others
  if (
      avg_probs[pet_idx] >= PET_MIN_CONFIDENCE
      and avg_probs[pet_idx] > avg_probs[hdpe_idx]
      and avg_probs[pet_idx] > avg_probs[pp_idx]
  ):
    top1_class = "PET"
    top1_conf = avg_probs[pet_idx]

  print(
      f">> الأعلى: {top1_class} ({top1_conf:.1f}%) | الثاني: {top2_class}"
      f" ({top2_conf:.1f}%)\n"
  )

  if locked_timer > 0:
    locked_timer -= 1
    result_text = locked_result
    text_color = (0, 255, 0)
  else:
    if top1_conf >= MIN_CONFIDENCE:
      if top1_class == last_candidate:
        consecutive_confirmed += 1
      else:
        last_candidate = top1_class
        consecutive_confirmed = 1

      if consecutive_confirmed >= STABLE_FRAMES_NEEDED:
        if (
            top1_class in ["HDPE", "PP"]
            and top2_class in ["HDPE", "PP"]
            and (top1_conf - top2_conf) < 15.0
        ):
          locked_result = f"Unclear: HDPE / PP ({top1_conf:.1f}%)"
          text_color = (0, 165, 255)
        else:
          locked_result = f"Detected: {top1_class} ({top1_conf:.1f}%)"
          text_color = (0, 255, 0)

        print(f"[✔] تأكيد نهائي: {locked_result}")
        locked_timer = LOCK_DISPLAY_FRAMES
        consecutive_confirmed = 0
        prob_history.clear()
      else:
        result_text = (
            f"Measuring... {top1_class}"
            f" ({consecutive_confirmed}/{STABLE_FRAMES_NEEDED})"
        )
        text_color = (0, 255, 255)
    else:
      consecutive_confirmed = 0
      last_candidate = ""
      result_text = "Waiting for Object..."
      text_color = (255, 255, 0)

  cv2.putText(
      display_frame,
      result_text,
      (20, 50),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.7,
      text_color,
      2,
      cv2.LINE_AA,
  )

  cv2.imshow("AUKEY Plastic Sorter", display_frame)

  key = cv2.waitKey(1) & 0xFF
  if key == ord("q"):
    break

cap.release()
cv2.destroyAllWindows()