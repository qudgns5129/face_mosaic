import dlib, cv2
import numpy as np

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')



####################### 얼굴 마스크 씌우기 ##############################

# load overlay image
overlay = cv2.imread('img/smile.png', cv2.IMREAD_UNCHANGED)

# overlay function
def overlay_transparent(background_img, img_to_overlay_t, x, y, overlay_size=None):
  try:
    bg_img = background_img.copy()
    # convert 3 channels to 4 channels
    if bg_img.shape[2] == 3:
      bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2BGRA)

    if overlay_size is not None:
      img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)

    b, g, r, a = cv2.split(img_to_overlay_t)

    mask = cv2.medianBlur(a, 5)

    h, w, _ = img_to_overlay_t.shape
    roi = bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)]

    img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask))
    img2_fg = cv2.bitwise_and(img_to_overlay_t, img_to_overlay_t, mask=mask)

    bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)] = cv2.add(img1_bg, img2_fg)

    # convert 4 channels to 4 channels
    bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGRA2BGR)

    return bg_img
  except Exception: return background_img

####################### 얼굴 인식하기 ##############################

save_space = np.load('img/save_space.npy')[()]

def encode_face(img):
  det = detector(img, 1)

  if len(det) == 0:
    return np.empty(0)

  for k, d in enumerate(det):
    shapes = sp(img, d)
    face_descriptor = facerec.compute_face_descriptor(img, shapes)

    return np.array(face_descriptor)

video_path = 'img/video3.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
  exit()

_, img_bgr = cap.read() # (800, 1920, 3)
padding_size = 0
resized_width = 800
video_size = (resized_width, int(img_bgr.shape[0] * resized_width // img_bgr.shape[1]))
output_size = (resized_width, int(img_bgr.shape[0] * resized_width // img_bgr.shape[1] + padding_size * 2))

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
writer = cv2.VideoWriter('%s_output.mp4' % (video_path.split('.')[0]), fourcc, cap.get(cv2.CAP_PROP_FPS), output_size)

while True:
  ret, img_bgr = cap.read()
  if not ret:
    break

  img_bgr = cv2.resize(img_bgr, video_size)
  img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

  # img_bgr = cv2.copyMakeBorder(img_bgr, top=padding_size, bottom=padding_size, left=0, right=0, borderType=cv2.BORDER_CONSTANT, value=(0,0,0))
  
  det = detector(img_bgr, 1)
  for i in range(0,len(det)):
    globals()['face{}'.format(i)] = det[i]

  for k, d in enumerate(det):
    # 얼굴 센터 인식
    dlib_shape = sp(img_rgb, d)
    shape_2d = np.array([[p.x, p.y] for p in dlib_shape.parts()])

    top_left = np.min(shape_2d, axis=0)
    bottom_right = np.max(shape_2d, axis=0)
    center_x, center_y = np.mean(shape_2d, axis=0).astype(np.int)

    face_size = max(bottom_right - top_left)

    face_descriptor = facerec.compute_face_descriptor(img_rgb, dlib_shape)
    
    last_found = {'name': 'unknown', 'dist': 0.3, 'color': (0,0,255)}

    for name, saved_desc in save_space.items():
      dist = np.linalg.norm([face_descriptor] - saved_desc, axis=1)

      if dist < last_found['dist']:
        last_found = {'name': name, 'dist': dist, 'color': (255,255,255)}
      else:
        result = overlay_transparent(img_bgr, overlay, center_x + 8, center_y - 25, overlay_size=(face_size, face_size))
    cv2.rectangle(img_bgr, pt1=(d.left(), d.top()), pt2=(d.right(), d.bottom()), color=last_found['color'], thickness=2)
    cv2.putText(img_bgr, last_found['name'], org=(d.left(), d.top()), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=last_found['color'], thickness=2)




  # 저장 후 영상재생
  writer.write(img_bgr)

  cv2.imshow('result', result)
  cv2.imshow('a', img_bgr)
  if cv2.waitKey(1) == ord('q'):
    break

cap.release()
writer.release()