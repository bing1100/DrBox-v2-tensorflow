from __future__ import division 
import math

# Lines (L*) are defined in the form Ax + By = C
# Points (p*) are defined in the tuple format (x, y)

def line(p1, p2):
    """
    Generates a line from two points
    :param p1, p2: two points
    :return: A, B, C line format
    """
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = p1[0]*p2[1] - p2[0]*p1[1]
    return A, B, -C

def linePointSlope(L, p):
    """
    Generates a line with the same slope from a line and a point on the new line
    :param L, p: a line and a point
    :return: A, B, C line format
    """
    A = L[0]
    B = L[1]
    C = -L[1]*p[1] - L[0]*p[0]
    return A, B, -C

def linePointSlopeInverted(L, p):
    """
    Generates a line with the inverted slope from a line and a point on the new line
    :param L, p: a line and a point
    :return: A, B, C line format
    """
    A = -L[1]
    B = L[0]
    C = -L[0]*p[1] + L[1]*p[0]
    return A, B, -C

def length(p1, p2):
    """
    Calculates the euclidean distance between two points
    :param p1, p2: two points
    :return: the distance between the two lines
    """
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def intersection(L1, L2):
    """
    Calculates the intersection between two lines
    :param L1, L2: two lines
    :return: a point that is the intersection or false if no point exists
    """
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x,y
    else:
        return False

def angle(p1, p2):
    """
    Calculates the angle of a line passing through two points
    :param p1, p2: two points on the line
    :return: the angle in degrees from 1 to 180 of the line
    """
    v = (p1[0] - p2[0], p1[1] - p2[1])
    a = math.degrees(math.atan2(v[1], v[0]))
    if a < 0:
        return 180 + a
    return 180 - a

def extend(p1, p2, val):
    """
    Calculates two new points further apart from each other by some value
    :param p1, p2, val: two points and a the value of extension
    :return: two new points ordered from left to right on the number line
    """
    s = p1
    e = p2
    if p2[0] < p1[0]:
        s = p2
        e = p1
    L = line(p1, p2)
    s[0] = s[0] - val
    e[0] = e[0] + val
    s[1] = -L[0]/L[1] * s[0] - L[2]/L[1]
    e[1] = -L[0]/L[1] * e[0] - L[2]/L[1]
    return s, e
    
def longer(cLong, cShort):
    """
    Determines the longer axis/line between 
    :param cLong, cShort: two tuple of tuples ((x1,y1), (x2,y2)) of a line segment
    :return: returns true if cLong is the longer segment, false otherwise
    """
    cLongMag = (cLong[0][0] - cLong[1][0])**2 + (cLong[0][1] - cLong[1][1])**2
    cShortMag = (cShort[0][0] - cShort[1][0])**2 + (cShort[0][1] - cShort[1][1])**2
    return cLongMag > cShortMag

def bucketCount(buckets, value, size):
    idx = int(min(value/size, len(buckets)-1))
    buckets[idx] += 1
    return buckets