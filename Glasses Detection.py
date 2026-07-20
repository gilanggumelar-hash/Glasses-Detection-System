import cv2
import mediapipe as mp
import numpy as np


NOSE_BRIDGE_LANDMARKS = [6, 168, 8, 9]
GLASSES_EDGE_THRESHOLD = 0.04

COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_YELLOW = (0, 255, 255)
COLOR_CYAN = (255, 255, 0)

mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


def extractNoseBridgePoints(landmarks, frameWidth, frameHeight):
    """Mengambil koordinat piksel (x, y) dari area jembatan hidung."""
    points = []
    for idx in NOSE_BRIDGE_LANDMARKS:
        landmarkPoint = landmarks.landmark[idx]
        pixelX = int(landmarkPoint.x * frameWidth)
        pixelY = int(landmarkPoint.y * frameHeight)
        points.append((pixelX, pixelY))
    return points


def calculateRoiBoundingBox(points, frameWidth, frameHeight):
    """Menghitung koordinat kotak pembatas (Bounding Box) untuk area ROI dengan padding."""
    allX = [p[0] for p in points]
    allY = [p[1] for p in points]
    
    paddingX = 20
    paddingY = 10
    
    xMin = max(0, min(allX) - paddingX)
    xMax = min(frameWidth, max(allX) + paddingX)
    yMin = max(0, min(allY) - paddingY)
    yMax = min(frameHeight, max(allY) + paddingY)
    
    return xMin, yMin, xMax, yMax


def checkGlassesByEdgeDensity(roiImage):
    """Menganalisis keberadaan kacamata berdasarkan kerapatan garis tepi (Canny Edge)."""
    grayRoi = cv2.cvtColor(roiImage, cv2.COLOR_BGR2GRAY)
    blurredRoi = cv2.GaussianBlur(grayRoi, (3, 3), 0)
    edges = cv2.Canny(blurredRoi, 30, 150)
    
    edgeDensity = np.sum(edges > 0) / edges.size
    return edgeDensity > GLASSES_EDGE_THRESHOLD


def detectGlasses(currentFrame, faceLandmarks, frameWidth, frameHeight):
    """Fungsi utama untuk memproses ROI wajah dan menentukan status kacamata."""
    nosePoints = extractNoseBridgePoints(faceLandmarks, frameWidth, frameHeight)
    
    if len(nosePoints) < 4:
        return False

    xMin, yMin, xMax, yMax = calculateRoiBoundingBox(nosePoints, frameWidth, frameHeight)
    
    roiImage = currentFrame[yMin:yMax, xMin:xMax]
    if roiImage.size == 0:
        return False
        
    cv2.rectangle(currentFrame, (xMin, yMin), (xMax, yMax), COLOR_CYAN, 1)
    
    return checkGlassesByEdgeDensity(roiImage)


cameraCapture = cv2.VideoCapture(0)
print("Tekan 'q' pada jendela gambar untuk keluar.")

try:
    while cameraCapture.isOpened():
        isSuccess, frame = cameraCapture.read()
        if not isSuccess:
            print("Gagal mengakses kamera.")
            break
            
        frame = cv2.flip(frame, 1)
        frameHeight, frameWidth, _ = frame.shape
        
        rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detectionResults = faceMesh.process(rgbFrame)
        
        statusText = "Mencari Wajah..."
        textColor = COLOR_YELLOW
        
        if detectionResults.multi_face_landmarks:
            for faceLandmarks in detectionResults.multi_face_landmarks:
                hasGlasses = detectGlasses(frame, faceLandmarks, frameWidth, frameHeight)
                
                if hasGlasses:
                    statusText = "Kacamata: YA"
                    textColor = COLOR_GREEN
                else:
                    statusText = "Kacamata: TIDAK"
                    textColor = COLOR_RED
                    
        cv2.putText(frame, statusText, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, textColor, 2, cv2.LINE_AA)
        cv2.imshow('Deteksi Kacamata', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cameraCapture.release()
    cv2.destroyAllWindows()