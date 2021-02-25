import cv2
import numpy as np
import pandas as pd
from random import randrange
import os
import pathlib



class SandecoYolo:

    __version__ = "SandecoYolo v1.0.1, Criado por Sandeco Macedo. http://youtube.com/canalsandeco"


    def __init__(self, lang='pt',
                     cfg = "yolov3.cfg",
                     weights = "yolov3.weights",
                     threshold = 0.5,
                     GPUSupport = False, specific_classes=None, colors=None, font_scale=0.5):

        self.dir = pathlib.Path(__file__).parent.absolute()


        classes = self.defineClasses(lang)


        self.classes = classes
        self.cfg = cfg
        self.weights = weights
        self.threshold = threshold
        self.GPUSupport = GPUSupport
        self.net = None
        self.specific_classes = specific_classes

        self.totalDown = 0
        self.totalUp = 0
        self.coun = 0

        self.colors = []

        if colors !=None:
            self.colors = colors
        else:
            for classe in specific_classes:
                R = randrange(256)
                G = randrange(256)
                B = randrange(256)

                color = (B,G,R)
                self.colors.append(color)

        self.velocity = 1

        self.mirror = False

        #Font definition
        self.font = cv2.FONT_HERSHEY_PLAIN
        self.font_scale = font_scale



    def mirrorOn(self):
        self.mirror = True

    def setVideo(self, video):
        self.capture = cv2.VideoCapture(video)

    def nextFrame(self):
        for i in range(self.velocity):
            eof, frame = self.capture.read()

            if self.mirror:
                frame = cv2.flip(frame,1)


        return not eof, frame

    def firstFrame(self):
        return self.nextFrame()

    def setVelocity(self, velocity=1):
        self.velocity = velocity

    def defineClasses(self, lang):

        csv = pathlib.PurePath(self.dir, 'yolo-classes.csv')

        df = pd.read_csv(csv)
        if lang == 'pt':
            classes = df.por.values
        elif lang == 'en':
            classes = df.eng.values

        return classes

    def loadImg(self, image):
        if isinstance(image, str):
            self.img = cv2.imread(image)
        else:
            self.img = image

        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.height, self.width, self.channes = self.img.shape

    def loadYolo(self):
        #self.net = cv2.dnn.readNetFromDarknet(self.cfg, self.weights)
        self.net = cv2.dnn.readNet(self.cfg, self.weights)

        if self.GPUSupport:
            self.setGPUSupport()

    def setGPUSupport(self):
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    def detection(self, image):
        class_ids = []
        confidences = []
        boxes = []

        #CARREGANDO YOLO
        if self.net is None:
            self.loadYolo()

        self.loadImg(image)

        ## CRIANDO BLOB
        blob = cv2.dnn.blobFromImage(self.img, 0.007843, (416, 416), (0, 0, 0), True, crop=False)

        ## Setando blob a rede yolo
        self.net.setInput(blob)

        ## CAMADAS DA YOLO
        layer_names = self.net.getLayerNames()
        outputlayers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        ## previsão baseado nos blobs
        outs = self.net.forward(outputlayers)

        for out in outs:

            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                classe = self.classes[class_id]
                if self.specific_classes !=None:
                    if classe not in self.specific_classes:
                        continue


                confidence = scores[class_id]
                if confidence > self.threshold:

                    # onject detected
                    center_x = int(detection[0] * self.width)
                    center_y = int(detection[1] * self.height)
                    w = int(detection[2] * self.width)
                    h = int(detection[3] * self.height)

                    # rectangle co-ordinaters
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    # cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

                    boxes.append([x, y, w, h])  # put all rectangle areas
                    confidences.append(
                        float(confidence))  # how confidence was that object detected and show that percentage
                    class_ids.append(class_id)  # name of the object tha was detected

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.6)
        boxes = np.array(boxes).astype(int)

        self.boxes = np.array(boxes)
        self.class_ids = np.array(class_ids)
        self.indexes = np.array(indexes)
        self.scores = np.array(confidences)

    def getBoxes(self):
        return self.boxes

    def getArrayBoxes(self):
        bbox = []
        for i in range(len(self.boxes)):
            if i in self.indexes:
                (x, y, w, h) = self.boxes[i]
                box = (x,y,x+w,y+h)
                bbox.append(box)

        return np.array(bbox)


    def getIdClasses(self):
        return self.class_ids

    def getIndexes(self):
        return self.indexes

    def getScores(self):
        return self.scores

    def releaseVideo(self):
        self.capture.release()

    def showBBoxes(self, img, color=(0,255,255), label=False, prob = False):

        for i in range(len(self.boxes)):
            if i in self.indexes:
                x, y, w, h = self.boxes[i]
                if label:

                    label = str(self.classes[self.class_ids[i]])


                    cor = self.specific_classes.index(label)

                    #cor = np.where(self.classes == label)[0][0]

                    color = self.colors[cor]



                    (w_tx, h_tx) = cv2.getTextSize(label, self.font, fontScale=self.font_scale, thickness=1)[0]
                    box_text = ((x, y-h_tx-5), (x + w_tx + 5 ,y))

                    cv2.rectangle(img, box_text[0], box_text[1], color, cv2.FILLED)
                    cv2.putText(img, label, (x+2,y-h_tx+3), self.font, fontScale=self.font_scale, color=(0, 0, 0), thickness=1)


                cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)

        return img

    def bboxesWithID(self, objects):

        qtd = len(objects)

        for objectID in list(objects.keys()):
            centroid = objects[objectID]
            pos =  (centroid[0],centroid[1])
            cv2.putText(self.img, str(qtd), pos, self.font, 1, (255, 255, 255), 2)

        return self.bboxes()