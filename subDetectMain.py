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
from subDetect import Ui_CompareWindow

pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files\Tesseract-OCR\\tesseract.exe'

class Com(QtWidgets.QMainWindow, Ui_CompareWindow):
    def __init__(self):
        super(Com, self).__init__()
        QtGui.QWindow.__init__(self)
        Ui_CompareWindow.__init__(self)
        self.setupUi(self)

    # OPEN AN IMAGE
    def OpenFile(self):
        self.name = QtWidgets.QFileDialog.getOpenFileName(self, "Select Image")[0]
        self.image = cv2.imread(self.name)
        self.pixmap = QtGui.QPixmap(self.name)
        self.pixmap = self.pixmap.scaled(240, 330, QtCore.Qt.KeepAspectRatio)
        self.label.setPixmap(self.pixmap)

    # CLEAR
    def Reset(self):
        self.label.clear()
        self.label_3.clear()
        self.label_4.clear()
        self.label_5.clear()
        self.label_9.clear()
        self.label_10.clear()
        self.label_11.clear()

    # SAVE AN IMAGE
    def SaveImage(self):
        fname, fliter = QtWidgets.QFileDialog.getSaveFileName(self, 'Save File', 'C:\\Users\\user\\Desktop\\', "Image Files (*.jpeg);;Image Files (*.bmp);;Image Files (*.tiff)")
        if fname:
            cv2.imwrite(fname, self.image)
        else:
            print('Error')

    # SAMPLE IMAGE FOR OBJECT DETECTION
    def sample1(self):
        self.name = 'sample1.jpeg'
        self.image = cv2.imread(self.name)
        self.pixmap = QtGui.QPixmap(self.name)
        self.pixmap = self.pixmap.scaled(240, 330, QtCore.Qt.KeepAspectRatio)
        self.label.setPixmap(self.pixmap)

    # SAMPLE IMAGE FOR CHARACTER RECOGNITION
    def sample2(self):
        self.name = 'sample2.jpeg'
        self.image = cv2.imread(self.name)
        self.pixmap = QtGui.QPixmap(self.name)
        self.pixmap = self.pixmap.scaled(240, 330, QtCore.Qt.KeepAspectRatio)
        self.label.setPixmap(self.pixmap)

    # SAMPLE IMAGE FOR CHARACTER RECOGNITION
    def sample3(self):
        self.name = 'sample3.jpg'
        self.image = cv2.imread(self.name)
        self.pixmap = QtGui.QPixmap(self.name)
        self.pixmap = self.pixmap.scaled(240, 330, QtCore.Qt.KeepAspectRatio)
        self.label.setPixmap(self.pixmap)

    # COMPARISON OF THE THINNING TECHNIQUES
    def compareThinning(self):
        # BEFORE THINNING, CANNY
        image = self.image.copy()
        # image=cv2.resize(image, (1000,700), interpolation = cv2.INTER_AREA)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        self.can = cv2.Canny(gray, 50, 100)

        img = QtGui.QImage(self.can, self.can.shape[1], self.can.shape[0],
                           self.can.shape[1], QImage.Format_Grayscale8)
        self.pixmap = QtGui.QPixmap(img)
        self.pixmap = self.pixmap.scaled(430, 320, QtCore.Qt.KeepAspectRatio)
        self.label_3.setPixmap(self.pixmap)

        # BEFORE THINNING, SOBEL
        image = self.image.copy()
        # image = cv2.resize(image, (1000, 700), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        sobelX = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        sobelY = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
        sobelX = np.uint8(np.absolute(sobelX))
        sobelY = np.uint8(np.absolute(sobelY))
        self.sobel = cv2.bitwise_or(sobelX, sobelY)

        img = QtGui.QImage(self.sobel, self.sobel.shape[1], self.sobel.shape[0],
                           self.sobel.shape[1], QImage.Format_Grayscale8)
        self.pixmap = QtGui.QPixmap(img)
        self.pixmap = self.pixmap.scaled(430, 320, QtCore.Qt.KeepAspectRatio)
        self.label_4.setPixmap(self.pixmap)

        # BEFORE THINNING, PREWITT
        image = self.image.copy()
        # image = cv2.resize(image, (1000, 700), interpolation=cv2.INTER_AREA)
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
        self.prewitt = cv2.bitwise_or(pX, pY)

        img = QtGui.QImage(self.prewitt, self.prewitt.shape[1], self.prewitt.shape[0],
                           self.prewitt.shape[1], QImage.Format_Grayscale8)
        self.pixmap = QtGui.QPixmap(img)
        self.pixmap = self.pixmap.scaled(430, 320, QtCore.Qt.KeepAspectRatio)
        self.label_5.setPixmap(self.pixmap)

        # AFTER THINNING, CANNY
        image = self.image.copy()
        # image=cv2.resize(image, (1000,700), interpolation = cv2.INTER_AREA)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        can = cv2.Canny(gray, 50, 100)
        _, threshold = cv2.threshold(can, 125, 1, cv2.THRESH_BINARY)
        ske = skeletonize(threshold)
        ske1 = img_as_ubyte(ske)
        dil = cv2.dilate(ske1, None, iterations=1)
        ero = cv2.erode(dil, None, iterations=1)
        thi = thin(ero)
        self.edged = img_as_ubyte(thi)

        img = QtGui.QImage(self.edged, self.edged.shape[1], self.edged.shape[0],
                           self.edged.shape[1], QImage.Format_Grayscale8)
        self.pixmap = QtGui.QPixmap(img)
        self.pixmap = self.pixmap.scaled(430, 320, QtCore.Qt.KeepAspectRatio)
        self.label_9.setPixmap(self.pixmap)

        # AFTER THINNING, SOBEL
        image = self.image.copy()
        # image = cv2.resize(image, (1000, 700), interpolation=cv2.INTER_AREA)
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
        self.edged1 = img_as_ubyte(thi)

        img = QtGui.QImage(self.edged1, self.edged1.shape[1], self.edged1.shape[0],
                           self.edged1.shape[1], QImage.Format_Grayscale8)
        self.pixmap = QtGui.QPixmap(img)
        self.pixmap = self.pixmap.scaled(430, 320, QtCore.Qt.KeepAspectRatio)
        self.label_10.setPixmap(self.pixmap)

        # AFTER THINNING, PREWITT
        image = self.image.copy()
        # image = cv2.resize(image, (1000, 700), interpolation=cv2.INTER_AREA)
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
        self.edged2 = img_as_ubyte(thi)

        img = QtGui.QImage(self.edged2, self.edged2.shape[1], self.edged2.shape[0],
                           self.edged2.shape[1], QImage.Format_Grayscale8)
        self.pixmap = QtGui.QPixmap(img)
        self.pixmap = self.pixmap.scaled(430, 320, QtCore.Qt.KeepAspectRatio)
        self.label_11.setPixmap(self.pixmap)

if __name__=='__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = Com()
    window.show()
    sys.exit(app.exec())