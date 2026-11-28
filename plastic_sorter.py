"""
Plastic AI Sorter v2 - كشف تلقائي فوري + كارت معلومات منفصل لكل نوع

  - أول ما قطعة تظهر قدام الكاميرا (بدون ضغط أي زر) بيتعرف عليها خلال أجزاء من الثانية.
  - كل نوع ليه فرع if/elif لوحده (PP / PET / HDPE / others) بيطلّع كارت فيه:
    صورة النوع + الرقم + الفوائد + المجالات + القرار. مفيش خلط بين الأنواع.
  - الكاميرا بتفتكر الخلفية الفاضية في أول ثانية، وبتحلل بس لما يبقى فيه جسم.

تشغيل:   python plastic_sorter.py            (كاميرا 1، وتجرب 0 و 2 لو مفتحتش)
         python plastic_sorter.py --cam 0 --lang en
مفاتيح:  q خروج | r ريست | b إعادة تصوير الخلفية الفاضية
"""
import argparse
import os
import sys
from collections import deque

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter
except ImportError:
    from tflite_runtime.interpreter import Interpreter


HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "plastic_sorting_model.tflite")
 # HDPE.jpg PET.jpg PP.jpg others.jpg

# ترتيب التدريب بالظبط (من النوتبوك): ['HDPE', 'PET', 'PP', 'others']
CLASS_NAMES = ["HDPE", "PET", "PP", "others"]
IDX = {n: i for i, n in enumerate(CLASS_NAMES)}

# ---------------- ضبط السرعة والحساسية (عدّل براحتك) ----------------
WINDOW_SIZE = 7            # متوسط آخر كام فريم (أقل = أسرع)
MIN_HISTORY = 2            # أقل فريمات قبل القرار
STABLE_FRAMES_NEEDED = 5   # نفس النوع كام فريم ورا بعض (~0.15 ثانية على 30fps)
SWITCH_FACTOR = 3          # لو القطعة لسه قدام الكاميرا: التغيير لنوع تاني يحتاج أدلة أقوى x3
REMOVE_FRAMES = 10          # كام فريم فاضي علشان نعتبر القطعة اتشالت
UNCLEAR_GAP = 15.0
MIN_CONF = {"HDPE": 55.0, "PET": 35.0, "PP": 65.0, "others": 80.0}  # PET حد أقل (الموديل بيظلمه)

# كشف وجود جسم (مقارنة بالخلفية الفاضية)
CALIBRATION_FRAMES = 20
PRESENCE_PIXEL_DIFF = 35   # فرق السطوع اللي يتحسب تغيير
PRESENCE_MIN_AREA = 0.08   # نسبة الصورة المتغيرة علشان نقول "فيه جسم" (4%)

CAM_W, CAM_H = 640, 480
PANEL_W = 480

COLORS = {"HDPE": (255, 140, 0), "PET": (0, 170, 255), "PP": (60, 200, 90),
          "others": (160, 160, 160), "unclear": (255, 165, 0)}

# ----------------------------------------------------------------------------
# بيانات كل نوع (كل نوع لوحده)
# ----------------------------------------------------------------------------
INFO_PP = {
    "code": "#5", "full": "Polypropylene",
    "benefits": {
        "en": ["Heat resistant - safe for food containers",
               "Light, tough, chemical resistant",
               "Recyclable and reusable many times"]},
    "uses": {"en": "Food containers, caps, car parts, ropes and fibers"},
    "decision": {"en": "Safe for repeated reuse "},
            
}
INFO_HDPE = {
    "code": "#2", "full": "High-Density Polyethylene",
    "benefits": {
        "en": ["Strong, rigid, impact resistant",
               "Resists chemicals and moisture",
               "Long lifetime, easy to recycle"]},
    "uses": {"en": "Pipes, detergent jugs, plastic lumber, crates and bins"},
    "decision": {"en": "Send to industrial manufacturing lines + can be reused for 2 times"},
}
    
INFO_PET = {
    "code": "#1", "full": "Polyethylene Terephthalate",
    "benefits": {
        "en": ["Clear, light and strong",
               "One of the most widely recycled plastics",
               "rPET saves energy vs. virgin plastic"]},
    "uses": {"en": "Polyester fiber (clothes, carpets), new bottles, strapping"},
    "decision": {"en": "Send to industrial manufacturing lines(rPET) + can be reused for 1 time"},
}            

INFO_OTHERS = {
    "code": "#7", "full": "Other / mixed plastics",
    "benefits": {
        "en": ["Mixed or unidentified plastic type",
               "Dangerous to recycle industrially",
               "Great for manual reuse projects"],
        },
    "uses": {"en": "Planters, desk organizers, handmade decor"},
    "decision": {"en": "Manual recycling (DIY)"},
}
INFO_UNCLEAR = {
    "code": "?", "full": "HDPE or PP - not sure",
    "benefits": {
        "en": ["HDPE and PP look very similar",
               "Move the item closer, improve lighting",
               "Hold it steady and scan again"]},
    "uses": {"en": "-"},
    "decision": {"en": "Unclear - re-scan"},
}


def get_info(label):
    """الـ if/elif لكل نوع: كل نوع بياخد كارته الخاصة."""
    if label == "PP":
        return INFO_PP
    elif label == "HDPE":
        return INFO_HDPE
    elif label == "PET":
        return INFO_PET
    elif label == "unclear":
        return INFO_UNCLEAR
    else:
        return INFO_OTHERS


# ----------------------------------------------------------------------------
# خطوط ونصوص
# ----------------------------------------------------------------------------
FONT_CANDIDATES = [
    r"C:\Windows\Fonts\tahoma.ttf", r"C:\Windows\Fonts\arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
]
FONT_PATH = next((p for p in FONT_CANDIDATES if os.path.exists(p)), None)


def _font(size):
    if FONT_PATH:
        try:
            return ImageFont.truetype(FONT_PATH, size)
        except OSError:
            pass
    return ImageFont.load_default()

def _shape(text, lang):
    return text

def _wrap(text, font, max_w, draw, lang):
    lines, cur = [], ""
    for w in text.split():
        trial = (cur + " " + w).strip()
        if draw.textlength(_shape(trial, lang), font=font) <= max_w:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


# ----------------------------------------------------------------------------
# الموديل
# ----------------------------------------------------------------------------
class Classifier:
    def __init__(self, model_path=MODEL_PATH):
        self.interp = Interpreter(model_path=model_path)
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]
        _, self.h, self.w, _ = self.inp["shape"]
        n_out = int(self.out["shape"][-1])
        if n_out != len(CLASS_NAMES):
            raise RuntimeError(f"Model outputs {n_out} classes but CLASS_NAMES has {len(CLASS_NAMES)}")
   
    def predict(self, frame_bgr):
        img = cv2.cvtColor(cv2.resize(frame_bgr, (self.w, self.h)), cv2.COLOR_BGR2RGB)
        # الموديل فيه Rescaling جواه => 0..255 خام (مش /255)
        x = np.expand_dims(img.astype(np.float32), axis=0)
        self.interp.set_tensor(self.inp["index"], x)
        self.interp.invoke()
        p = self.interp.get_tensor(self.out["index"])[0].astype(np.float64)
        if p.min() < 0 or not np.isclose(p.sum(), 1.0, atol=1e-2):
            e = np.exp(p - p.max())
            p = e / e.sum()
        return p * 100.0


# ----------------------------------------------------------------------------
# كشف وجود جسم (بدون ما نعتمد على الموديل في تحديد "فاضي ولا لأ")
# ----------------------------------------------------------------------------
class Presence:
    def __init__(self):
        self.bg, self._acc = None, []

    @staticmethod
    def _prep(frame):
        g = cv2.cvtColor(cv2.resize(frame, (160, 120)), cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(g, (9, 9), 0).astype(np.float32)

    def recalibrate(self):
        self.bg, self._acc = None, []

    @property
    def calibrating(self):
        return self.bg is None

    def check(self, frame):
        """يرجّع True لو فيه جسم. أثناء المعايرة بيرجّع None."""
        cur = self._prep(frame)
        if self.bg is None:
            self._acc.append(cur)
            if len(self._acc) >= CALIBRATION_FRAMES:
                self.bg = np.mean(self._acc, axis=0)
            return None
        d = np.abs((cur - cur.mean()) - (self.bg - self.bg.mean()))
        changed = (d > PRESENCE_PIXEL_DIFF).mean()
        present = changed >= PRESENCE_MIN_AREA
        if not present:
             self.bg = 0.9 * self.bg + 0.1 * cur
        return present

# ----------------------------------------------------------------------------
# منطق القرار
# ----------------------------------------------------------------------------
def decide(avg):
    order = np.argsort(avg)[::-1]
    top1, c1 = CLASS_NAMES[order[0]], float(avg[order[0]])
    top2, c2 = CLASS_NAMES[order[1]], float(avg[order[1]])
    pet = avg[IDX["PET"]]
    if (pet >= MIN_CONF["PET"] and pet > avg[IDX["HDPE"]] and pet > avg[IDX["PP"]]
            and top1 != "PET"):
        rest = [(CLASS_NAMES[i], float(avg[i])) for i in order if CLASS_NAMES[i] != "PET"]
        top1, c1 = "PET", float(pet)
        top2, c2 = rest[0]
    return top1, c1, top2, c2


class Sorter:
    lang= "en"
    def __init__(self, classifier):
        self.clf = classifier
        self.presence = Presence()
        self.history = deque(maxlen=WINDOW_SIZE)
        self.candidate, self.streak, self.absent = "", 0, 0
        self.was_present = False
        self.need_clear = False
        self.result = None
        self.refs = {}
        self.status = ("Calibrating empty background...", (0, 200, 255))

    def review_image(self, name):
        if name not in self.refs:
            img = cv2.imread(os.path.join(MODEL_PATH, f"{name}.jpg"))
            if img is None:
                img = np.full((300, 300, 3), 70, np.uint8)
                cv2.putText(img, name, (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
            self.refs[name] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.refs[name]

    def reset(self):
        self.history.clear()
        self.candidate, self.streak, self.absent = "", 0, 0
        self.was_present = self.need_clear = False
        self.result = None

    def recalibrate(self):
        self.reset()
        self.presence.recalibrate()
        self.status = ("Calibrating empty background...", (0, 200, 255))

    # ---------------- معالجة فريم ----------------
    def update(self, frame):
        present = self.presence.check(frame)
        if present is None:  # لسه بتتصوّر الخلفية الفاضية
            self.status = ("Calibrating empty background... keep the view empty", (0, 200, 255))
            return None

        if not present:
            self.absent += 1
            self.streak, self.candidate = 0, ""
            if self.absent >= REMOVE_FRAMES:
                self.need_clear = False
                self.was_present = False
                self.history.clear()
            self.status = ("Waiting for object...", (255, 255, 0))
            return None

        # جسم جديد ظهر => ابدأ من الصفر علشان فريمات قديمة ما تلخبطش النتيجة
        if not self.was_present:
            self.history.clear()
            self.was_present = True
        self.absent = 0

        self.history.append(self.clf.predict(frame))
        if len(self.history) < MIN_HISTORY:
            self.status = ("Object detected... analysing", (0, 255, 255))
            return None
        avg = np.mean(self.history, axis=0)
        top1, c1, top2, c2 = decide(avg)

        unclear = (top1 in ("HDPE", "PP") and top2 in ("HDPE", "PP")
                   and (c1 - c2) < UNCLEAR_GAP and (c1 + c2) >= MIN_CONF["HDPE"])
        cand, conf = ("unclear", c1 + c2) if unclear else (top1, c1)
        thr = MIN_CONF["HDPE"] if unclear else MIN_CONF[top1]

        if conf < thr:
            self.candidate, self.streak = "", 0
            self.status = ("Object detected... analysing", (0, 255, 255))
            return None

        if cand == self.candidate:
            self.streak += 1
        else:
            self.candidate, self.streak = cand, 1

        # القطعة لسه قدام الكاميرا وعندنا نتيجة => التغيير لنوع تاني يحتاج أدلة أقوى
        same_as_shown = self.result is not None and self.result["key"] == cand
        needed = STABLE_FRAMES_NEEDED * (SWITCH_FACTOR if (self.need_clear and not same_as_shown) else 1)

        if self.streak >= needed and not (self.need_clear and same_as_shown):
            self.result = {"key": cand, "raw": top1, "conf": conf}
            self.need_clear = True
            self.streak = 0
            self._announce(cand, conf)
            self.status = (f"Detected: {cand} ({conf:.0f}%)", (0, 255, 0))
            return self.result

        if self.need_clear and same_as_shown:
            self.status = (f"Detected: {cand} ({self.result['conf']:.0f}%)", (0, 255, 0))
        else:
            self.status = (f"Measuring... {cand} ({min(self.streak, needed)}/{needed})", (0, 255, 255))
        return None

    def _announce(self, key, conf):
        info = get_info(key)
        L = self.lang
        print("=" * 55)
        print(f"[OK] {key} {info['code']} - {info['full']}  ({conf:.1f}%)")
        print(f"  {info['decision'][L]}")
        print(f"  {info['uses'][L]}")
        print("=" * 55, flush=True)

    # ---------------- الرسم ----------------
    def render(self, frame):
        cam = cv2.resize(frame, (CAM_W, CAM_H))
        text, color = self.status
        cv2.rectangle(cam, (0, 0), (CAM_W, 38), (0, 0, 0), -1)
        cv2.putText(cam, text, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
        panel = cv2.cvtColor(self._panel(), cv2.COLOR_RGB2BGR)
        return np.hstack([cam, panel])

    def _panel(self):
        img = Image.new("RGB", (PANEL_W, CAM_H), (24, 28, 36))
        d = ImageDraw.Draw(img)
        lang = self.lang
        f_title, f_big, f_mid, f_small = _font(20), _font(46), _font(17), _font(16)

        if not self.result:
            d.text((20, 18), "Plastic Sorter", font=f_title, fill=(230, 230, 230))
            d.text((20, 200), "Show a plastic item to the camera", font=f_mid, fill=(150, 150, 150))
            d.text((20, 228), "It will be detected automatically.", font=f_mid, fill=(150, 150, 150))
            return np.array(img)

        key, conf = self.result["key"], self.result["conf"]
        info = get_info(key)           # <<< الفرع الخاص بالنوع ده بس
        col = COLORS[key]

        # صورة النوع
        ref = Image.fromarray(self.review_image(self.result["raw"]))
        ref.thumbnail((190, 170))
        d.rectangle([16, 16, 24 + ref.width, 24 + ref.height], fill=col)
        img.paste(ref, (20, 20))

        # العنوان
        tx = 24 + ref.width + 22
        title = "HDPE / PP?" if key == "unclear" else key
        d.text((tx, 14), title, font=f_big, fill=col)
        d.text((tx, 72), f"{info['code']}  {conf:.0f}%", font=f_mid, fill=(230, 230, 230))
        for i, ln in enumerate(_wrap(info["full"], f_small, PANEL_W - tx - 12, d, "en")[:2]):
            d.text((tx, 98 + i * 20), ln, font=f_small, fill=(170, 170, 170))

        def put_line(text, y, bullet=False, fill=(235, 235, 235)):
            s = ("• " + text) if bullet else text
            d.text((20, y), s, font=f_small, fill=fill)

        def head(text, y):
            d.text((20, y), text, font=f_small, fill=(120, 170, 255))

        y = 20 + max(ref.height, 120) + 18
        head("Benefits", y); y += 24
        for b in info["benefits"][lang]:
            for ln in _wrap(b, f_small, PANEL_W - 55, d, lang)[:2]:
                put_line(ln, y, bullet=True); y += 21
        y += 8
        if info["uses"][lang] != "-":
            head("Common uses", y); y += 24
            for ln in _wrap(info["uses"][lang], f_small, PANEL_W - 40, d, lang)[:2]:
                put_line(ln, y); y += 21
            y += 8
        head("Decision", y); y += 24
        for ln in _wrap(info["decision"][lang], f_small, PANEL_W - 40, d, lang)[:2]:
            put_line(ln, y, fill=col); y += 21
        return np.array(img)

# ----------------------------------------------------------------------------
def open_camera(preferred):
    order = [preferred] + [i for i in (0, 1, 2, 3) if i != preferred]
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    for idx in order:
        cap = cv2.VideoCapture(idx, backend)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
                print(f"[OK] الكاميرا شغالة على index {idx}")
                return cap
        cap.release()
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=1)
    args = ap.parse_args()

    sorter = Sorter(Classifier())
    cap = open_camera(args.cam)
    if cap is None:
        print("No camera found! Check the USB cable.")
        return
    print("Keep the camera view empty for calibration, then place the plastic in front of the camera.")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read a frame from the camera.")
            break
        sorter.update(frame)
        cv2.imshow("Plastic Sorter", sorter.render(frame))
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord("r"):
            sorter.reset()
        if k == ord("b"):
            sorter.recalibrate()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
