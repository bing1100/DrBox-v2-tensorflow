import util as u
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import getopt

# Note: Copy this file to the train directory before running. 
#       Run this file after running the imagesplit file.
#       This will convert all of the splitkeypoints files into rbox files
EPSILON = 5
START = 1
END = 1001 # Change this to however many images you have +1
PRINT = False
WRITE = True
SHOW = False

trainLines = []

wBucket = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
hBucket = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
aBucket = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

f = open("./splitlist.txt", "r")
lines = (f.read()).split("\n")

for name in lines[:-1]:
    kpsFile = "./splitKeypoints/" + name + ".txt"
    RBoxFile = "./splitImages/" + name + ".tif.rbox"    # Change tif to your image extension
    imgFile = "./splitImages/" + name + ".tif"          # Change tif to your image extension

    trainLines.append(name + ".tif " + name + ".tif.rbox\n") # Change tif to your image extension

    f = open(kpsFile, "r")
    
    # Lines for kps and rbox
    lines = (f.read()).split("\n")
    rboxLines = []

    # To Show the keypoints
    if SHOW:
        img=mpimg.imread(imgFile)
        imgplot = plt.imshow(img)

    # Split each line in keypoints file
    kps = [line.split(",") for line in lines[:-1]]
    for kp in kps:
        label = kp[-1]
        xs = [int(float(i)) for i in kp[2:-1:2]]
        ys = [int(float(i)) for i in kp[3:-1:2]]

        # Show the keypoints
        if SHOW:
            plt.scatter(xs, ys)

        # Sort the keypoints to head, tail, and wings of plane
        pH = (xs[0], ys[0])
        pT = (xs[1], ys[1])
        pW1 = (xs[2], ys[2])
        pW2 = (xs[3], ys[3])

        # Find the major/longest axis for the plane and the points of the major and minor axis
        LMaj = u.line(pH, pT)
        pMaj = (pH, pT)
        pMin = (pW1, pW2)
        # if u.longer((pW1, pW2), (pH, pT)):
        #     LMaj = u.line(pW1, pW2)
        #     pMaj = (pW1, pW2)
        #     pMin = (pH, pT)

        # Find the equations of the long and short edges of the bounding box
        le = (u.linePointSlope(LMaj, pMin[0]), 
            u.linePointSlope(LMaj, pMin[1]))
        se = (u.linePointSlopeInverted(LMaj, pMaj[0]), 
            u.linePointSlopeInverted(LMaj, pMaj[1]))

        # Find the vertices of the bounding box from the lines
        v = (u.intersection(le[0], se[0]), 
            u.intersection(le[0], se[1]), 
            u.intersection(le[1], se[0]), 
            u.intersection(le[1], se[1]))

        # Show the raw vertices without epsilon
        if SHOW:
            plt.scatter(np.array(v)[:,0], np.array(v)[:,1])

        # Calculate the diagonal lines of the bounding box
        diags = (u.line(v[0], v[3]), u.line(v[1], v[2]))

        # Calculate the center, width, height, and angle of the bounding box
        c = u.intersection(diags[0], diags[1])      # Find the center by looking at the intersection
        w = u.length(v[0], v[1]) + EPSILON          # Find the width
        h = u.length(v[0], v[2]) + EPSILON          # Find the height
        a = u.angle(v[3], v[2])                     # Find the angle

        wBucket = u.bucketCount(wBucket, w, 10)
        hBucket = u.bucketCount(hBucket, h, 10)
        aBucket = u.bucketCount(aBucket, a, 10)

        if SHOW:
            plt.scatter(c[0], c[1])
            plt.annotate(a, (c[0], c[1]))

        line = str(c[0]) + " " + str(c[1]) + " " + str(w) + " " + str(h) + " " + str(label) + " " + str(a) + "\n"
        rboxLines.append(line)

        if PRINT:
            print(line)
        
    if SHOW:
        plt.show()

    if WRITE:
        with open(RBoxFile,'w') as target:
            target.writelines(rboxLines)

with open("./train.txt",'w') as target:
    target.writelines(trainLines)

# Print the prior width, height, and angle values to use.
print(wBucket)
print(hBucket)
print(aBucket)

    

    


