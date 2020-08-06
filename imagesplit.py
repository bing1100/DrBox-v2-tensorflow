import cv2
from pathos.multiprocessing import ProcessingPool as Pool

# Note: Copy this file to the train directory before running. s
NUMPROCESS = 4
NEWWIDTH = 300
NEWHEIGHT = 300
OVERLAP = 0.5

def start_points(size, split_size):
    points = [0]
    stride = int(split_size * (1-OVERLAP))
    counter = 1
    while True:
        pt = stride * counter
        if pt + split_size >= size:
            points.append(size - split_size)
            break
        else:
            points.append(pt)
        counter += 1
    return points

def checkKeyPoints(yStart, xStart, ys, xs):
    for x in xs:
        for y in ys:
            if x < xStart or (xStart + NEWWIDTH) < x:
                return False
            if y < yStart or (yStart + NEWHEIGHT) < y:
                return False
    return True
            
def processImg(num):
    kpsFile = "./keypoints/" + str(num) + ".txt"
    imgFile = "./images/" + str(num) + ".tif"

    f = open(kpsFile, "r")
    img = cv2.imread(imgFile)
    img_h, img_w, _ = img.shape
    
    # Lines for kps, split each line in keypoints file
    lines = (f.read()).split("\n")
    kps = [line.split(",") for line in lines[:-1]]

    # Find Starting Points to split the image
    X_points = start_points(img_w, NEWWIDTH)
    Y_points = start_points(img_h, NEWHEIGHT)

    name = str(num)
    frmt = 'tif'

    # Iterate over all images that can be split
    count = 0
    for i in Y_points:
        for j in X_points:
            newLines = []

            # Iterate over all keypoints that could be on split image
            kpCount = 0
            for kp in kps:
                xs = [int(float(i)) for i in kp[0::2]]
                ys = [int(float(i)) for i in kp[1::2]]
                
                # Check if the keypoints are on this image
                if checkKeyPoints(i, j, ys, xs):
                    # Add keypoints on this image to be added to the kp file
                    newLines.append(lines[kpCount] + "\n")
                    # Generate and save the split image
                    split = img[i:i+NEWHEIGHT, j:j+NEWWIDTH]
                    cv2.imwrite('./splitImages/{}_{}.{}'.format(name, count, frmt), split)
                
                kpCount += 1
            
            # Save the kps for the split image only if there are kps
            if len(newLines) != 0:
                with open('./splitKeypoints/{}_{}.txt'.format(name, count),'w') as target:
                    target.writelines(newLines)

            count += 1
    
p = Pool(NUMPROCESS)
p.map(processImg, list(range(1,1001)))


