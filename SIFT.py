import cv2 
import pickle
import matplotlib.pyplot as plt
import pysift

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input1', type=str, required=True)
parser.add_argument('--input2', type=str, required=True)
args = parser.parse_args()

def imageResizeTrain(image):
    maxD = 1024
    height,width = image.shape
    aspectRatio = width/height
    if aspectRatio < 1:
        newSize = (int(maxD*aspectRatio),maxD)
    else:
        newSize = (maxD,int(maxD/aspectRatio))
    image = cv2.resize(image,newSize)
    return image

def imageResizeTest(image):
    maxD = 1024
    height,width,channel = image.shape
    aspectRatio = width/height
    if aspectRatio < 1:
        newSize = (int(maxD*aspectRatio),maxD)
    else:
        newSize = (maxD,int(maxD/aspectRatio))
    image = cv2.resize(image,newSize)
    return image

bf = cv2.BFMatcher()
def calculateMatches(des1,des2):
    matches = bf.knnMatch(des1,des2,k=2)
    topResults1 = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            topResults1.append([m])
            
    matches = bf.knnMatch(des2,des1,k=2)
    topResults2 = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            topResults2.append([m])
    
    topResults = []
    for match1 in topResults1:
        match1QueryIndex = match1[0].queryIdx
        match1TrainIndex = match1[0].trainIdx

        for match2 in topResults2:
            match2QueryIndex = match2[0].queryIdx
            match2TrainIndex = match2[0].trainIdx

            if (match1QueryIndex == match2TrainIndex) and (match1TrainIndex == match2QueryIndex):
                topResults.append(match1)
    return topResults

def getPlot(image1,image2,keypoint1,keypoint2,matches):
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    matchPlot = cv2.drawMatchesKnn(
        image1,
        keypoint1,
        image2,
        keypoint2,
        matches,
        None,
        [255,255,255],
        flags=2
    )
    return matchPlot

def computeSIFT(image):
    return sift.detectAndCompute(image, None)

def calculateScore(matches,keypoint1,keypoint2):
    return 100 * (matches/min(keypoint1,keypoint2))


imagesBW = []
imagesBW.append(imageResizeTrain(cv2.imread(args.input1,0))) 
imagesBW.append(imageResizeTrain(cv2.imread(args.input2,0)))

sift = cv2.SIFT_create()

keypoints = []
descriptors = []
for i,image in enumerate(imagesBW):
    keypoint = []
    keypointTemp, descriptorTemp = computeSIFT(image)
    for point in keypointTemp:
        temp = cv2.KeyPoint(
            x=point.pt[0],
            y=point.pt[1],
            size=point.size,
            angle=point.angle,
            response=point.response,
            octave=point.octave,
            class_id=point.class_id
        )
        keypoint.append(temp)
    keypoints.append(keypoint)
    descriptors.append(descriptorTemp)

matches = calculateMatches(descriptors[0], descriptors[1])
score = calculateScore(len(matches),len(keypoints[0]),len(keypoints[1]))
print(score)

image1 = imageResizeTest(cv2.imread(args.input1))
image2 = imageResizeTest(cv2.imread(args.input2))
plot = getPlot(image1,image2,keypoints[0],keypoints[1],matches)
plt.imshow(plot),plt.show()
