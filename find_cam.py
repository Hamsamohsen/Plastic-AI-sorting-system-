import cv2

for index in range(5):
  cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
  if cap.isOpened():
    ret, frame = cap.read()
    if ret and frame is not None:
      print(f"[✔] Index {index} شغال")
      cv2.imshow(f"Index {index} - Press any key", frame)
      cv2.waitKey(0)
      cv2.destroyAllWindows()
    else:
      print(f"[!] Index {index} فتح بس مفيش صورة")
  else:
    print(f"[x] Index {index} غير متاح")
  cap.release()