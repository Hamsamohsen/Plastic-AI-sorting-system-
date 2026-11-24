import time
import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = "plastic_sorting_model.tflite"
LABELS = ["PP", "HDPE", "PET", "Others"]  # عدّلي الترتيب حسب ترتيب تدريب الموديل
CONF_THRESHOLD = 0.65

RECOMMENDATIONS = {
    "PP": {
        "decision": "آمن تماماً لإعادة الاستخدام المتكرر + قابل للتصنيع",
        "industries": "علب طعام، أواني منزلية، أغطية، أجزاء سيارات، ألياف",
    },
    "HDPE": {
        "decision": "توجيه إلى خطوط التصنيع الصناعي",
        "industries": "مواسير، جراكن منظفات/كيماويات، ألواح بلاستيك، حاويات",
    },
    "PET": {
        "decision": "توجيه إلى خطوط التصنيع الصناعي (rPET)",
        "industries": "خيوط بوليستر، ملابس/مفروشات، أشرطة ربط، عبوات rPET",
    },
    "Others": {
        "decision": "إعادة تدوير يدوي (DIY)",
        "industries": "أصص زرع، منظمات مكتب، ديكورات ومجسمات يدوية",
    },
}


def get_recommendation(label: str, confidence: float) -> dict:
    if confidence < CONF_THRESHOLD:
        return {
            "decision": "إعادة فحص / تحقق بصري",
            "industries": "نسبة الثقة منخفضة؛ قرّب القطعة أو حسّن الإضاءة",
        }
    return RECOMMENDATIONS.get(label, RECOMMENDATIONS["Others"])


# تحميل موديل TFLite
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

_, height, width, _ = input_details[0]["shape"]
input_dtype = input_details[0]["dtype"]

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("تعذر فتح الكاميرا. تأكد من توصيلها أو غيّر رقم الكاميرا (0 أو 1).")

last_printed_label = None
last_print_time = 0.0

print("تم تشغيل الكاميرا... اضغط 'q' في نافذة الصورة للإيقاف.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("فشل قراءة إطار من الكاميرا.")
        break

    # تجهيز الصورة حسب أبعاد الموديل
    resized = cv2.resize(frame, (width, height))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    if input_dtype == np.float32:
        input_data = np.expand_dims(rgb.astype(np.float32) / 255.0, axis=0)
    else:
        input_data = np.expand_dims(rgb.astype(input_dtype), axis=0)

    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]["index"])[0]

    # لو الموديل مخرج logits بدلاً من probabilities
    if preds.min() < 0 or preds.max() > 1.0 or not np.isclose(preds.sum(), 1.0, atol=1e-2):
        exp_preds = np.exp(preds - np.max(preds))
        preds = exp_preds / np.sum(exp_preds)

    class_idx = int(np.argmax(preds))
    confidence = float(preds[class_idx])
    label = LABELS[class_idx] if class_idx < len(LABELS) else "Others"

    rec = get_recommendation(label, confidence)
    display_key = f"{label}_{rec['decision']}"

    # طباعة التوصية في الـ Terminal عند تغيّر النتيجة أو مرور 3 ثوانٍ مع اختلاف الفئة
    now = time.time()
    if display_key != last_printed_label and (now - last_print_time) > 1.0:
        print("=" * 55)
        print(f"النوع المكتشف : {label} | نسبة الثقة: {confidence * 100:.1f}%")
        print(f"التوصية        : {rec['decision']}")
        print(f"المجالات / DIY : {rec['industries']}")
        print("=" * 55)
        last_printed_label = display_key
        last_print_time = now

    # عرض نص مختصر على نافذة الكاميرا للمتابعة البصرية
    preview_text = f"{label} ({confidence * 100:.0f}%) - {rec['decision']}"
    cv2.putText(
        frame,
        preview_text,
        (15, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )
    cv2.imshow("Plastic Classifier Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()