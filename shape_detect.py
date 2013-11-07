
#import rect
import cv2
import time
import numpy as np
import cv2.cv as cv
import urllib2
from math import sqrt

from jira_client import JiraProxy
from jira_client import ISSUE_TYPES as itypes
from jira_client import USERS as jirausers
import os

def diffImg(t0, t1, t2):
  d1 = cv2.absdiff(t2, t1)
  d2 = cv2.absdiff(t1, t0)
  return cv2.bitwise_and(d1, d2)


def rectify(h):
  #print "a=%r, b=%r, c=%r" % (a, b, c)
  h = h.reshape((4,2))
  hnew = np.zeros((4,2),dtype = np.float32)
  
  add = h.sum(1)
  hnew[0] = h[np.argmin(add)]
  hnew[2] = h[np.argmax(add)]
  
  diff = np.diff(h,axis = 1)
  hnew[1] = h[np.argmin(diff)]
  hnew[3] = h[np.argmax(diff)]
  
  return hnew


def preprocess(img):
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  blur = cv2.GaussianBlur(gray,(5,5),2 )
  thresh = cv2.adaptiveThreshold(blur,255,1,1,11,1)
  return thresh

def imgdiff(img1,img2):
  img1 = cv2.GaussianBlur(img1,(5,5),5)
  img2 = cv2.GaussianBlur(img2,(5,5),5)
  diff = cv2.absdiff(img1,img2)
  diff = cv2.GaussianBlur(diff,(5,5),5)
  flag, diff = cv2.threshold(diff, 200, 255, cv2.THRESH_BINARY)
  return np.sum(diff)

def find_closest_rectangle(training,img):
  features = preprocess(img)
  return sorted(training.values(), key=lambda x:imgdiff(x[1],features))[0][0]


def getRectangles(im, numrectangles=4):
  gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
  blur = cv2.GaussianBlur(gray,(1,1),1000)
  flag, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY)
  contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
  return sorted(contours, key=cv2.contourArea,reverse=True)[:numrectangles]

def angle_cos(p0, p1, p2):
  d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
  return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )


# A quick test to check whether the contour is
# a connected shape
def connected(cnt):
  first = cnt[0][0]
  ast = cnt[len(cnt)-1][0]
  return abs(first[0] - last[0]) <= 1 and abs(first[1] - last[1]) <= 1

def has_no_children (idx, hierarchy, cnt):
  #print idx
  #print hierarchy
  if hierarchy is None:
    print "hierarchy is None"
    return True
  else:
    h = hierarchy[0]
    if(h[idx][2] < 0):
      #print "hierarchy has no further children"
      return True
    else:
      #print "xx - hierarchy has children"
      return False

def keep(cnt):
  return keep_card(cnt) and connected(cnt)

def keep_card(cnt):
  elem = np.array([cnt], dtype=np.int32)
  x,y,w,h = cv2.boundingRect(elem)
  
  # width and height need to be floats
  w *= 1.0
  h *= 1.0
  
  # Test it's shape - if it's too oblong or tall it's
  # probably not a real character
  if(w/h < 0.1 or w/h > 10):
    #print "\t Rejected because of shape: ("+str(x)+","+str(y)+","+str(w)+","+str(h)+")" + str(w/h)
    return False
  

  if (w/h < 1.1):
    #print "\t Rejected because of width to height ratio: " + str(w/h)
    return False

  # Test whether the box is too wide
  if(w > img_x/5):
    #print "\t Rejected because of width: " + str(w)
    return False
  
  # Test whether the box is too tall
  if(h > img_y/5):
    #print "\t Rejected because of height: " + str(h)
    return False
  
  if(h < 15):
    #print "\t Rejected because of low height: " + str(h)
    return False
  
  if(w < 20):
    #print "\t Rejected because of low width: " + str(h)
    return False
  
  return True


def find_squares(img, image_area):
  img = cv2.GaussianBlur(img, (5, 5), 0)
  allowable_area = image_area - 100000
  squares = []
  for gray in cv2.split(img):
    for thrs in xrange(0, 255, 26):
      if thrs == 0:
        bin = cv2.Canny(gray, 0, 50, apertureSize=5)
        bin = cv2.dilate(bin, None)
      else:
        retval, bin = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY)
      contours, hierarchy = cv2.findContours(bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
      #print('=========')
      #print(len(contours))    
      for idx,cnt in enumerate(contours):
        cnt_len = cv2.arcLength(cnt, True)
        cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
        current_size = cv2.contourArea(cnt)
        if len(cnt) == 4 and current_size > 100 and cv2.isContourConvex(cnt):
          cnt = cnt.reshape(-1, 2)
          max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in xrange(4)])
          if max_cos < 0.1:
            if current_size < allowable_area:
              if keep_card(cnt) and has_no_children(idx,hierarchy,cnt):
                squares = add_contour(cnt,squares)
                #print(len(squares))
  return squares

def find_triangles(img, image_area):
  img = cv2.GaussianBlur(img, (5, 5), 0)
  allowable_area = image_area - 100000
  squares = []
  for gray in cv2.split(img):
    for thrs in xrange(0, 255, 26):
      if thrs == 0:
        bin = cv2.Canny(gray, 0, 50, apertureSize=5)
        bin = cv2.dilate(bin, None)
      else:
        retval, bin = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY)
      contours, hierarchy = cv2.findContours(bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
      print('=========')
      print(len(contours))    
      for idx,cnt in enumerate(contours):
        cnt_len = cv2.arcLength(cnt, True)
        cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
        current_size = cv2.contourArea(cnt)
        if len(cnt) == 3 and current_size > 100 and cv2.isContourConvex(cnt):
          #cnt = cnt.reshape(-1, 2)
          #max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in xrange(4)])
          #if max_cos < 0.1:
          #  if current_size < allowable_area:
          squares.append(cnt)
  return squares



def find_circles(img):
  img = cv2.medianBlur(img,5)
  circles = cv2.HoughCircles(img,cv2.cv.CV_HOUGH_GRADIENT,1,10,param1=100,param2=30,minRadius=5,maxRadius=20)
  # if circles is None:
  #   print 'no circles'
  # else:
  #   print circles
  #circles = np.uint16(np.around(circles))
  return circles


def add_contour(contour, contours):
  for cont in contours:
    #print "checking for duplicates"
    if (overlaps(cont,contour)):
      #print "duplicate"
      return contours
  
  #print "no match. appending a contour"
  contours.append(contour)
  return contours

def overlaps(contour1, contour2):	
  contourArray1 = np.array([contour1], dtype=np.int32)
  contourArray2 = np.array([contour2], dtype=np.int32)
  x1,y1,w1,h1 = cv2.boundingRect(contourArray1)
  x2,y2,w2,h2 = cv2.boundingRect(contourArray2)
  # two rectangles overlap (x-wise) if the distance to their centers is smaller than the average width.
  if (abs(x1+w1/2-x2-w2/2) >= (w1+w2)/2):
    return False
  if (abs(y1+h1/2-y2-h2/2) >= (h1+h2)/2):
    return False
  return True

def getPersonByColor(color):
  if color == "yellow":
    return jirausers["rayland"]
  if color == "red":
    return jirausers["matt"]
  if color == "green":
    return jirausers["kelly"]
  return jirausers["matt"]



#storage = cv.CreateMemStorage(0)
delay = 0
max_num_rectangles = 4

cam = cv2.VideoCapture(0)
#feed = cv.CaptureFromCAM(0)
#frame = cv.QueryFrame(feed)

winName = "Hackathon 2013-2"
cv2.namedWindow(winName, cv2.CV_WINDOW_AUTOSIZE)

# Read three images first:
img = cv2.cvtColor(cam.read()[1], 0)
#gray = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)

#t_plus = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)

#img = cv2.copyMakeBorder(orig_img, 50,50,50,50,cv2.BORDER_CONSTANT)
hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# Calculate the width and height of the image
img_y = len(img)
img_x = len(img[0])
image_height = img_y
image_half = img_y / 2
print "Image is " + str(len(img)) + "x" + str(len(img[0]))

while True:
  image_area = gray.size        # this is area of the image
  squares = find_squares(img, image_area)
  cv2.drawContours( img, squares, -1, (0, 255, 0), 3 )
  cv2.imshow('squares', img)
  
  # for triangle in find_triangles(img, image_area):
  #   #elem = np.array([triangle], dtype=np.int32)
  #   #x,y,w,h = cv2.boundingRect(elem)
  #   cv2.drawContours(img,[triangle],0,(0,255,0),-1)
  #   print ("triangle - ")
  #   print triangle

  for cnt in find_squares(img, image_area):
    elem = np.array([cnt], dtype=np.int32)
    x,y,w,h = cv2.boundingRect(elem)
    cv2.drawContours( img, [cnt], -1, (0, 255, 0), 3 )
    
    isUpper = True
    stateOrientation = "unknown"
    if y > image_half:
      stateOrientation = "lower"
    #  print "upper!!!!"
    else:
    #  print "lower!!!!"
      stateOrientation = "upper"

    cx,cy = x+w/2, y+h/2
    color = hsv[cy,cx,0]
    #print "vvvvvvvvvvvv"
    #print color
    clrName = "unknown"
    if (color < 10 or color > 170):
    #  print('Red')
      clrName = "red"
             #res.append([cx,cy,'R'])
    elif(50 < color < 70):
    #  print('Green')
      clrName = "green"
             #res.append([cx,cy,'G'])
    elif(20 < color < 40):
    #  print('Yellow')
      clrName = "yellow"
             #res.append([cx,cy,'Y'])
    elif(98 < color < 145):
    #  print('Blue')
      clrName = "blue"
             #res.append([cx,cy,'B'])
    #print "^^^^^^^^^"

    sub_card = img[y:y+h, x:x+w]
    dot_quadrant = 0
    sub_card_y = len(sub_card)
    sub_card_x = len(sub_card[0])
    #print "sub sub sub"
    #print sub_card_x, sub_card_y
	
    circle_x = -1
    circle_y = -1
	
    gray = cv2.cvtColor(sub_card,cv2.COLOR_BGR2GRAY)
    circles = find_circles(gray)
    if not circles is None:
      #cnt = circles[0]
      for cnt in circles[0,:]:
      #print "circle #%d" %i
        circle_x = cnt[0]
        circle_y = cnt[1]
        #print cnt[0],cnt[1]
        #print "+++++++++++++++++++"
        #print cnt[2]

        cv2.circle(sub_card,(cnt[0],cnt[1]),cnt[2],(0,255,0),1)  # draw the outer circle
        #cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)     # draw the center of the circle

      cv2.imshow('detected circles',sub_card)
      #cv2.drawContours(img,contours,-1,(0,255,0),-1)
    if not (circle_x == -1 and circle_y == -1):
      if circle_x<sub_card_x/2 and circle_y<sub_card_y/2:
        dot_quadrant=1
      elif circle_x>sub_card_x/2 and circle_y<sub_card_y/2:
        dot_quadrant=2
      elif circle_x<sub_card_x/2 and circle_y>sub_card_y/2:
	    dot_quadrant=3
      elif circle_x>sub_card_x/2 and circle_y>sub_card_y/2:       
        dot_quadrant=4


    if (dot_quadrant == 0 or clrName == "unknown"):
      print "no dot quad or orientation"
      print circle_x, circle_y
      print "color"
      print color
    else:
      
      card_file_name = "cards/card_" + clrName + "_" + stateOrientation + "_" + str(dot_quadrant) + ".jpg"
      card_jira_meta_file = "cards/card_" + clrName + "_" + stateOrientation + "_" + str(dot_quadrant) + ".json"
      print(card_file_name)
      #cv2.imshow('sub_cards',sub_card)
      jp = JiraProxy()

      if os.path.exists(card_file_name):
        print "file exists ", card_file_name
      else:
        print "New card detected... adding to JIRA"
        print "writing file ", card_file_name
        print "create task in JIRA"
        cv2.imwrite(card_file_name, sub_card)
        person = getPersonByColor(clrName)
        task = jp.create_issue(3,"Hackathon Task #" + str(dot_quadrant), "Hackathon Task #" + str(dot_quadrant) + " description.",person)
        print "vvvvvvvvvvvvvvv"
        if not task.has_key('status_code'):
          print task
          taskid = task["id"]
          print "attaching image... "
          attachment_response = jp.add_attachment(taskid,card_file_name)
          print attachment_response
          print "done uploading attachment"
          print taskid
        else:
          print "Bad things with sending to JIRA"
          print task

        #save a "meta" file with task id 
    # if len(approx)==5:
    #   print "pentagon"
    #   cv2.drawContours(img,[cnt],0,255,-1)
    # elif len(approx)==3:
    #   print "triangle"
    #   cv2.drawContours(img,[cnt],0,(0,255,0),-1)
    # elif len(approx)==4:
    #   print "square"
    #   cv2.drawContours(img,[cnt],0,(0,0,255),-1)
    # elif len(approx) == 9:
    #   print "half-circle"
    #   cv2.drawContours(img,[cnt],0,(255,255,0),-1)
    # elif len(approx) > 15:
    #   print "circle"
    #   cv2.drawContours(img,[cnt],0,(0,255,255),-1)
  
  #cv2.imshow(winName,img)
  time.sleep(delay)
  
  # cv2.imshow( winName, diffImg(t_minus, t, t_plus) )
      #COLOR_BGR2HSV
  #frame = cv.QueryFrame(feed)
  img = cv2.cvtColor(cam.read()[1], 0)
  #gray = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
  #hsv = cv2.cvtColor(orig_img,cv2.COLOR_BGR2HSV)
  #img = cv2.copyMakeBorder(orig_img, 50,50,50,50,cv2.BORDER_CONSTANT)
  hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
  gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  key = cv2.waitKey(10)
  if key == 27:
   # cv.ClearMemStorage(storage)
    cv2.destroyWindow(winName)
    break

print "Goodbye"




