from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2
from skimage.util import img_as_ubyte
from skimage.morphology import skeletonize, thin
import pytesseract
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap, QImage, QTransform
from PyQt5.QtWidgets import QMessageBox
import sys
from detect import Ui_MainWindow
from subDetectMain import Com

# MUST DOWNLOAD EXE FILE THROUGH GITHUB NOT JUST IN PYCHARM IN ORDER TO RUN THE FUNCTION FOR CHARACTER RECOGNITION
pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files\Tesseract-OCR\\tesseract.exe'

class Win(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(Win, self).__init__()
        QtGui.QWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)

    # OPEN AN IMAGE
    def OpenFile(self):
        self.name = QtWidgets.QFileDialog.getOpenFileName(self, "Select Image")[0]
        self.image = cv2.imread(self.name)
        self.pixmap = QtGui.QPixmap(self.name)
        self.pixmap = self.pixmap.scaled(900, 550, QtCore.Qt.KeepAspectRatio)
        self.label_4.setPixmap(self.pixmap)

    # CLEAR
    def Reset(self):
        self.label_4.clear()
        self.label_5.clear()

    # SAVE AN IMAGE
    def SaveImage(self):
        fname, fliter = QtWidgets.QFileDialog.getSaveFileName(self, 'Save File', 'C:\\Users\\user\\Desktop\\', "Image Files (*.jpeg);;Image Files (*.bmp);;Image Files (*.tiff)")
        if fname:
            cv2.imwrite(fname, self.image)
        else:
            print('Error')

    # ORIGINAL IMAGE
    def OriginalImage(self):
        self.image = cv2.imread(self.name)
        self.pixmap = QtGui.QPixmap(self.name)
        self.pixmap = self.pixmap.scaled(900, 550, QtCore.Qt.KeepAspectRatio)
        self.label_4.setPixmap(self.pixmap)

    # SAMPLE IMAGE FOR OBJECT DETECTION
    def sample1(self):
        self.name = 'sample1.jpeg'
        self.image = cv2.imread(self.name)
        self.pixmap = QtGui.QPixmap(self.name)
        self.pixmap = self.pixmap.scaled(900, 550, QtCore.Qt.KeepAspectRatio)
        self.label_4.setPixmap(self.pixmap)

    # SAMPLE IMAGE FOR CHARACTER RECOGNITION
    def sample2(self):
        self.name = 'sample2.jpeg'
        self.image = cv2.imread(self.name)
        self.pixmap = QtGui.QPixmap(self.name)
        self.pixmap = self.pixmap.scaled(900, 550, QtCore.Qt.KeepAspectRatio)
        self.label_4.setPixmap(self.pixmap)

    # SAMPLE IMAGE FOR CHARACTER RECOGNITION
    def sample3(self):
        self.name = 'sample3.jpg'
        self.image = cv2.imread(self.name)
        self.pixmap = QtGui.QPixmap(self.name)
        self.pixmap = self.pixmap.scaled(900, 550, QtCore.Qt.KeepAspectRatio)
        self.label_4.setPixmap(self.pixmap)

    # OBJECT DETECTION AND MEASUREMENT OF SIZE OF EACH OBJECT FOR CANNY
    def cannyDetection(self):
        image = self.image.copy()
        # image=cv2.resize(image, (400,600), interpolation = cv2.INTER_AREA)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        can = cv2.Canny(gray, 50, 100)
        _, threshold = cv2.threshold(can, 125, 1, cv2.THRESH_BINARY)
        ske = skeletonize(threshold)
        ske1 = img_as_ubyte(ske)
        dil = cv2.dilate(ske1, None, iterations=1)
        ero = cv2.erode(dil, None, iterations=1)
        thi = thin(ero)
        edged = img_as_ubyte(thi)

        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        (cnts, _) = contours.sort_contours(cnts)

        cnts = [c for c in cnts if cv2.contourArea(c) > 100]
        ref_object = cnts[0]
        box = cv2.minAreaRect(ref_object)
        box = cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        box = perspective.order_points(box)
        (tl, tr, br, bl) = box
        DistInPixel = dist.euclidean(tl, tr)
        DistInCm = 2
        PixelsPerCm = DistInPixel / DistInCm

        for cnt in cnts:
            # BOUNDING BOX OF CONTOUR
            box = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(box)
            box = np.array(box, dtype="int")

            # ORDER THE POINTS IN THE CONTOUR
            box = perspective.order_points(box)
            cv2.drawContours(image, [box.astype("int")], -1, (0, 255, 0), 2)

            for (x, y) in box:
                cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)

            (tl, tr, br, bl) = box
            (tltrX, tltrY) = ((tl[0] + tr[0]) * 0.5, (tl[1] + tr[1]) * 0.5)
            (blbrX, blbrY) = ((bl[0] + br[0]) * 0.5, (bl[1] + br[1]) * 0.5)
            (tlblX, tlblY) = ((tl[0] + bl[0]) * 0.5, (tl[1] + bl[1]) * 0.5)
            (trbrX, trbrY) = ((tr[0] + br[0]) * 0.5, (tr[1] + br[1]) * 0.5)

            cv2.circle(image, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
            cv2.circle(image, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
            cv2.circle(image, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
            cv2.circle(image, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

            cv2.line(image, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 255), 2)
            cv2.line(image, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 255), 2)

            dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
            dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

            dimA = dA / PixelsPerCm
            dimB = dB / PixelsPerCm

            cv2.putText(image, "{:.1f}cm".format(dimA), (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 0, 0), 2)
            cv2.putText(image, "{:.1f}cm".format(dimB), (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (0, 0, 0), 2)

            self.image = image
            img = QtGui.QImage(self.image, self.image.shape[1], self.image.shape[0],
                         self.image.shape[1] * 3, QImage.Format_RGB888).rgbSwapped()
            self.pixmap = QtGui.QPixmap(img)
            self.pixmap = self.pixmap.scaled(900,550, QtCore.Qt.KeepAspectRatio)
            self.label_4.setPixmap(self.pixmap)

    # OBJECT DETECTION AND MEASUREMENT OF SIZE OF EACH OBJECT FOR SOBEL
    def sobelDetection(self):
        image = self.image.copy()
        # image = cv2.resize(image, (1000, 600), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        sobelX = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        sobelY = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
        sobelX = np.uint8(np.absolute(sobelX))
        sobelY = np.uint8(np.absolute(sobelY))
        sobel = cv2.bitwise_or(sobelX, sobelY)
        _, threshold = cv2.threshold(sobel, 80, 1, cv2.THRESH_BINARY)
        ske = skeletonize(threshold)
        ske1 = img_as_ubyte(ske)
        dil = cv2.dilate(ske1, None, iterations=1)
        ero = cv2.erode(dil, None, iterations=1)
        thi = thin(ero)
        edged = img_as_ubyte(thi)

        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        (cnts, _) = contours.sort_contours(cnts)

        cnts = [c for c in cnts if cv2.contourArea(c) > 100]
        ref_object = cnts[0]
        box = cv2.minAreaRect(ref_object)
        box = cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        box = perspective.order_points(box)
        (tl, tr, br, bl) = box
        DistInPixel = dist.euclidean(tl, tr)
        DistInCm = 2
        PixelsPerCm = DistInPixel / DistInCm

        for cnt in cnts:
            # BOUNDING BOX OF CONTOUR
            box = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(box)
            box = np.array(box, dtype="int")

            # ORDER THE POINTS IN THE CONTOUR
            box = perspective.order_points(box)
            cv2.drawContours(image, [box.astype("int")], -1, (0, 255, 0), 2)

            for (x, y) in box:
                cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)

            (tl, tr, br, bl) = box
            (tltrX, tltrY) = ((tl[0] + tr[0]) * 0.5, (tl[1] + tr[1]) * 0.5)
            (blbrX, blbrY) = ((bl[0] + br[0]) * 0.5, (bl[1] + br[1]) * 0.5)
            (tlblX, tlblY) = ((tl[0] + bl[0]) * 0.5, (tl[1] + bl[1]) * 0.5)
            (trbrX, trbrY) = ((tr[0] + br[0]) * 0.5, (tr[1] + br[1]) * 0.5)

            cv2.circle(image, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
            cv2.circle(image, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
            cv2.circle(image, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
            cv2.circle(image, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

            cv2.line(image, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 255), 2)
            cv2.line(image, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 255), 2)

            dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
            dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

            dimA = dA / PixelsPerCm
            dimB = dB / PixelsPerCm

            cv2.putText(image, "{:.1f}cm".format(dimA), (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 0, 0), 2)
            cv2.putText(image, "{:.1f}cm".format(dimB), (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (0, 0, 0), 2)

            self.image = image
            img = QtGui.QImage(self.image, self.image.shape[1], self.image.shape[0],
                               self.image.shape[1] * 3, QImage.Format_RGB888).rgbSwapped()
            self.pixmap = QtGui.QPixmap(img)
            self.pixmap = self.pixmap.scaled(900, 550, QtCore.Qt.KeepAspectRatio)
            self.label_4.setPixmap(self.pixmap)

    # OBJECT DETECTION AND MEASUREMENT OF SIZE OF EACH OBJECT FOR PREWITT
    def prewittDetection(self):
        image = self.image.copy()
        # image = cv2.resize(image, (1000, 600), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        kernel_Prewitt_x = np.array([
            [-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1]])
        kernel_Prewitt_y = np.array([
            [1, 1, 1],
            [0, 0, 0],
            [-1, -1, -1]])
        pX = cv2.filter2D(gray, -1, kernel_Prewitt_x)
        pY = cv2.filter2D(gray, -1, kernel_Prewitt_y)
        pX = np.uint8(np.absolute(pX))
        pY = np.uint8(np.absolute(pY))
        prewitt = cv2.bitwise_or(pX, pY)
        _, threshold = cv2.threshold(prewitt, 15, 1, cv2.THRESH_BINARY)
        ske = skeletonize(threshold)
        ske1 = img_as_ubyte(ske)
        dil = cv2.dilate(ske1, None, iterations=1)
        ero = cv2.erode(dil, None, iterations=1)
        thi = thin(ero)
        edged = img_as_ubyte(thi)

        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        (cnts, _) = contours.sort_contours(cnts)

        cnts = [c for c in cnts if cv2.contourArea(c) > 100]
        ref_object = cnts[0]
        box = cv2.minAreaRect(ref_object)
        box = cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        box = perspective.order_points(box)
        (tl, tr, br, bl) = box
        DistInPixel = dist.euclidean(tl, tr)
        DistInCm = 2
        PixelsPerCm = DistInPixel / DistInCm

        for cnt in cnts:
            # BOUNDING BOX OF CONTOUR
            box = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(box)
            box = np.array(box, dtype="int")

            # ORDER THE POINTS IN THE CONTOUR
            box = perspective.order_points(box)
            cv2.drawContours(image, [box.astype("int")], -1, (0, 255, 0), 2)

            for (x, y) in box:
                cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)

            (tl, tr, br, bl) = box
            (tltrX, tltrY) = ((tl[0] + tr[0]) * 0.5, (tl[1] + tr[1]) * 0.5)
            (blbrX, blbrY) = ((bl[0] + br[0]) * 0.5, (bl[1] + br[1]) * 0.5)
            (tlblX, tlblY) = ((tl[0] + bl[0]) * 0.5, (tl[1] + bl[1]) * 0.5)
            (trbrX, trbrY) = ((tr[0] + br[0]) * 0.5, (tr[1] + br[1]) * 0.5)

            cv2.circle(image, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
            cv2.circle(image, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
            cv2.circle(image, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
            cv2.circle(image, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

            cv2.line(image, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 255), 2)
            cv2.line(image, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 255), 2)

            dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
            dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

            dimA = dA / PixelsPerCm
            dimB = dB / PixelsPerCm

            cv2.putText(image, "{:.1f}cm".format(dimA), (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 0, 0), 2)
            cv2.putText(image, "{:.1f}cm".format(dimB), (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (0, 0, 0), 2)

            self.image = image
            img = QtGui.QImage(self.image, self.image.shape[1], self.image.shape[0],
                               self.image.shape[1] * 3, QImage.Format_RGB888).rgbSwapped()
            self.pixmap = QtGui.QPixmap(img)
            self.pixmap = self.pixmap.scaled(900, 550, QtCore.Qt.KeepAspectRatio)
            self.label_4.setPixmap(self.pixmap)

    # CHARACTER RECOGNITION
    def characterDetection(self):
        img=self.image
        textRecognized= pytesseract.image_to_string(img)
        boxes = pytesseract.image_to_data(img)
        for a, b in enumerate(boxes.splitlines()):
            print(b)
            if a != 0:
                b = b.split()
                if len(b) == 12:
                    x, y, w, h = int(b[6]), int(b[7]), int(b[8]), int(b[9])
                    cv2.putText(img, b[11], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)

        self.w = textRecognized
        self.label_5.setText(str(textRecognized))

        self.image = img
        img = QtGui.QImage(self.image, self.image.shape[1], self.image.shape[0],
                                               self.image.shape[1] * 3, QImage.Format_RGB888).rgbSwapped()
        self.pixmap = QtGui.QPixmap(img)
        self.pixmap = self.pixmap.scaled(900, 550, QtCore.Qt.KeepAspectRatio)
        self.label_4.setPixmap(self.pixmap)

    # CALL THE FUNCTION FOR COMPARISON
    def compareEdge(self):
        self.wc=Com()
        self.wc.show()

    # HELP BUTTON
    def Help(self):
        msg = QMessageBox()
        msg.setWindowTitle("Information")
        msg.setText("Object Detection and Character Recognization")
        msg.setDetailedText("FOR OBJECT DETECTION "+"                                                 "+"1. Use flat lay image. "+"                                                 "+"2. Use 2cmX2cm object as reference to measure size. "
                            +"                                                 "+"FOR CHARACTER RECOGNIZATION "+"                                                 "+"1. Only alphabet and number to be detected. "
                            +"                                                 "+"2. Text in image must be straight and faced to us.")
        x = msg.exec_()

if __name__=='__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = Win()
    window.show()
    sys.exit(app.exec())
