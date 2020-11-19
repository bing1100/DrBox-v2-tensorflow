import cv2
from pathos.multiprocessing import ProcessingPool as Pool

# Note: Copy this file to the directory containing the folder with all your images before running.
#       Make sure images and keypoints are name numerically from 1 to n
#       Place all images in a folder called images and all keypoints in a folder called keypoints
#       Before running this file, create folders called splitImages and splitKeypoints
#       All of the split images and corresponding keypoints will be saved in the above two folders
#       An additional file called splitlist.txt is also generated containing all the names of your
#       split images. 
LABELED = True      # True if the data has labels
NUMPROCESS = 12     # The number of cores your CPU has - for multi-processing - set to 1 if not sure (will be slow)
NEWWIDTH = 300      # Set the new width of each chunked image
NEWHEIGHT = 300     # Set the new height of each chunked image
OVERLAP = 0.20      # Set the overlap percentage of adjacent chunked images

# String->integer conversion of labels - only applies if you have additional object labels to preserve
CNUM = {
    "Boeing737": 0,
    "Boeing747": 1,
    "Boeing777": 2,
    "Boeing787": 3,
    "A220": 4,
    "A321": 5,
    "A330": 6,
    "A350": 7,
    "ARJ21": 8,
    "other": 9,
    "invalid": 10,
    "NA": 11
}

# Helper function specific for the airplane example - you will not need this
def i(string):
    switch = {
        "cp": 0,
        "kps": 3,
        "gtb": 4,
        "gtn": 3,
    }
    return switch[string]

# Calculates all of the starting points for each chunked image on the original image
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

# MODIFY ME: Given a new chunked image starting at (xStart, yStart), check to see if any of the keypoints
#   are on the chunked image
def checkKeyPoints(yStart, xStart, ys, xs):
    for x in xs:
        for y in ys:
            if x < xStart or (xStart + NEWWIDTH) < x:
                return False
            if y < yStart or (yStart + NEWHEIGHT) < y:
                return False
    return True

# Processes each image 
def processImg(num):
    kpsFile = "./keypoints/" + str(num) + ".txt" 
    imgFile = "./images/" + str(num) + ".tif"       # Change tif to the extension you are working with
    
    # MODIFY ME: if you have external labels, change this part
    if LABELED:
        labelFile = "./label_xml/" + str(num) + ".xml"

    # Opens the original image file
    f = open(kpsFile, "r")
    img = cv2.imread(imgFile)
    img_h, img_w, _ = img.shape
    
    # Lines for kps, split each line in keypoints file
    lines = (f.read()).split("\n")
    kps = [line.split(",") for line in lines[:-1]]

    # Find Starting Points to split the image
    X_points = start_points(img_w, NEWWIDTH)
    Y_points = start_points(img_h, NEWHEIGHT)

    fileList = []
    name = str(num)
    frmt = 'tif'    # Change tif to the extension you are working with

    # Iterate over all images that can be split
    count = 0
    for i in Y_points:
        for j in X_points:
            newKPLines = []

            # Iterate over all keypoints that could be on split image
            kpCount = 0
            for kp in kps:
                xs = [int(float(i)) for i in kp[0::2]]
                ys = [int(float(i)) for i in kp[1::2]]
                
                # Check if the keypoints are on this image
                if checkKeyPoints(i, j, ys, xs):
                    
                    # MODIFY ME: If you have external change this section 
                    plabel = "NA"
                    if LABELED:
                        lTree = ET.parse(labelFile)
                        lRoot = lTree.getroot()
                        plabel = "invalid"
                        for cand in lRoot.iter("object"):
                            coords = [
                                u.s2n(cand[i("gtb")][0].text),
                                u.s2n(cand[i("gtb")][1].text),
                                u.s2n(cand[i("gtb")][2].text),
                                u.s2n(cand[i("gtb")][3].text),
                            ]
                            cp = [xs[0], ys[0]]
                            if u.within(coords, cp):
                                plabel = cand[i("gtn")][0].text
                    
                    if plabel == "NA" or plabel == "invalid":
                        print(plabel)
                        
                    label = CNUM[plabel]
                    
                    # Add keypoints on this image to be added to the kp file
                    #   Shift the values accordingly to the new starting point
                    newKP = [str(xs[0] - j), str(ys[0] - i), 
                             str(xs[1] - j), str(ys[1] - i), 
                             str(xs[2] - j), str(ys[2] - i), 
                             str(xs[3] - j), str(ys[3] - i), 
                             str(xs[4] - j), str(ys[4] - i),
                             label]
                    newKPLines.append(','.join(newKP) + "\n")
                    # Generate and save the split image
                    split = img[i:i+NEWHEIGHT, j:j+NEWWIDTH]
                    cv2.imwrite('./splitImages/{}_{}.{}'.format(name, count, frmt), split)
                    fileList.append('{}_{}\n'.format(name, count))
                
                kpCount += 1
            
            # Save the kps for the split image only if there are kps
            if len(newKPLines) != 0:
                with open('./splitKeypoints/{}_{}.txt'.format(name, count),'w') as target:
                    target.writelines(newKPLines)

            count += 1

    return fileList

# Run the the complete algorithm with multiprocessing 
p = Pool(NUMPROCESS)
results = p.map(processImg, list(range(1,1001))) # Change 1001 to however many images you have (n+1)

with open('./splitlist.txt','w') as target:
    for res in results:
        target.writelines(res)

