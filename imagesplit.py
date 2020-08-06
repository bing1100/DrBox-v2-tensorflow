import cv2
from pathos.multiprocessing import ProcessingPool as Pool

processNumbers = 4
split_width = 300
split_height = 300
overlap = 0.5

def start_points(size, split_size, overlap=0):
    points = [0]
    stride = int(split_size * (1-overlap))
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
            if x < xStart or (xStart + split_width) < x:
                return False

            if y < yStart or (yStart + split_height) < y:
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
    X_points = start_points(img_w, split_width, overlap)
    Y_points = start_points(img_h, split_height, overlap)

    count = 0
    name = str(num)
    frmt = 'tif'
    for i in Y_points:
        for j in X_points:
            
            count += 1

            kpCount = 0
            newLines = []
            for kp in kps:
                xs = [int(float(i)) for i in kp[0::2]]
                ys = [int(float(i)) for i in kp[1::2]]

                if checkKeyPoints(i, j, ys, xs):
                    newLines.append(lines[kpCount] + "\n")
                    split = img[i:i+split_height, j:j+split_width]
                    cv2.imwrite('./splitImages/{}_{}.{}'.format(name, count, frmt), split)
                
                kpCount += 1
            
            if len(newLines) != 0:
                with open('./splitKeyPoints/{}_{}.txt'.format(name, count),'w') as target:
                    target.writelines(newLines)
    
p = Pool(processNumbers)
p.map(processImg, list(range(1,1001)))


