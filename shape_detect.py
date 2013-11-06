

import cv2
import time
import numpy as np
import cv2.cv as cv
import urllib2
from math import sqrt


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
  print idx
  #print hierarchy
  if hierarchy is None:
    print "hierarchy is None"
    return True
  else:
    h = hierarchy[0]
    if(h[idx][2] < 0):
      print "hierarchy has no further children"
      return True
    else:
      print "xx - hierarchy has children"
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
    print "\t Rejected because of shape: ("+str(x)+","+str(y)+","+str(w)+","+str(h)+")" + str(w/h)
    return False
  
  # Test whether the box is too wide
  if(w > img_x/5):
    print "\t Rejected because of width: " + str(w)
    return False
  
  # Test whether the box is too tall
  if(h > img_y/5):
    print "\t Rejected because of height: " + str(h)
    return False
  
  if(h < 15):
    print "\t Rejected because of low height: " + str(h)
    return False
  
  if(w < 20):
    print "\t Rejected because of low width: " + str(h)
    return False
  
  return True


def find_squares(img, image_area):
  img = cv2.GaussianBlur(img, (5, 5), 0)
  allowable_area = image_area - 100000
  #image_area = img.size
#  image_width = img.width
#  image_height = img.height
  
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


      
	    #       if hierarchy is None:
	    # print "hierarchy is None"
	    # return True
	    #       else:
	    #         h = hierarchy[0]
	    #         for component in zip(contours, h):
	    #           currentContour = component[0]
	    #           contourArray = np.array([currentContour], dtype=np.int32)
	    #           currentHierarchy = component[1]
	    #           x,y,w,h = cv2.boundingRect(currentContour)
	    #           if currentHierarchy[2] < 0:
	    #             # these are the innermost child components
	    #             cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
	    #           elif currentHierarchy[3] < 0:
	    #             # these are the outermost parent components
	    #             cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)

      
      for idx,cnt in enumerate(contours):
        cnt_len = cv2.arcLength(cnt, True)
        cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
        current_size = cv2.contourArea(cnt)
        if len(cnt) == 4 and current_size > 100 and cv2.isContourConvex(cnt):
          cnt = cnt.reshape(-1, 2)
          max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in xrange(4)])
          if max_cos < 0.1:
            if current_size < allowable_area:
              #x,y,w,h = cv2.boundingRect(cnt)
              if keep_card(cnt) and has_no_children(idx,hierarchy,cnt):
                print('YES. keeping this contour')
                squares.append(cnt)
            # print("---------v")
            #          print(current_size)
            #          print(image_area)
            #          print("---------^")
  #return sorted(squares, key=cv2.contourArea,reverse=True)[:5]
  return squares


def find_circles(img):
  img = cv2.medianBlur(img,5)
  circles = cv2.HoughCircles(img,cv2.cv.CV_HOUGH_GRADIENT,1,10,param1=100,param2=30,minRadius=5,maxRadius=20)
  circles = np.uint16(np.around(circles))
  return circles


def overlaps(contour1, contour2):
  contourArray1 = np.array([contour1], dtype=np.int32)
  contourArray2 = np.array([contour2], dtype=np.int32)
  x1,y1,w1,h1 = cv2.boundingRect(contourArray1)
  # x1right = x1+w1
  #   x1left = x1
  #   x1top = y1+h1
  #   x1bottom = y1

  x2,y2,w2,h2 = cv2.boundingRect(contourArray2)
  return (x1+w1 > x2 and x1 < x2+w2 and y1+h1 < y2 and y1 > y2+h2)

#storage = cv.CreateMemStorage(0)
delay = 5
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

print "Image is " + str(len(img)) + "x" + str(len(img[0]))

while True:
  image_area = gray.size        # this is area of the image
  squares = find_squares(img, image_area)
  cv2.drawContours( img, squares, -1, (0, 255, 0), 3 )
  cv2.imshow('squares', img)
  
  for cnt in find_squares(img, image_area):
    elem = np.array([cnt], dtype=np.int32)
    x,y,w,h = cv2.boundingRect(elem)
    cv2.drawContours( img, [cnt], -1, (0, 255, 0), 3 )
    
    cx,cy = x+w/2, y+h/2
    color = hsv[cy,cx,0]
    
    if (color < 10 or color > 170):
      print('R')
             #res.append([cx,cy,'R'])
    elif(50 < color < 70):
      print('G')
             #res.append([cx,cy,'G'])
    elif(20 < color <40):
      print('Y')
             #res.append([cx,cy,'Y'])
    elif(110 < color < 130):
      print('B')
             #res.append([cx,cy,'B'])
    
    sub_card = img[y:y+h, x:x+w]
    card_file_name = "cards/card_" + str(y) + ".jpg"
   # print(card_file_name)
    cv2.imshow('sub_cards',sub_card)
    #cv2.imwrite(card_file_name, sub_card)


    
    #peri = cv2.arcLength(elem,True)
    #approx = rectify(cv2.approxPolyDP(elem,0.02*peri,True))
			    
			    # box = np.int0(approx)
			    # cv2.drawContours(im,[box],0,(255,255,0),6)
			    # imx = cv2.resize(im,(1000,600))
			    # cv2.imshow('a',imx)
    
    #h = np.array([ [0,0],[449,0],[449,449],[0,449] ],np.float32)
    
    #transform = cv2.getPerspectiveTransform(approx,h)
    #warp = cv2.warpPerspective(img,transform,(450,450))

#    circles = find_circles(grey_square)
#    cimg = cv2.cvtColor(grey_square,cv2.COLOR_RGB2GRAY)
#    for i in circles[0,:]:
#      cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),1)  # draw the outer circle
      #cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)     # draw the center of the circle

#    cv2.imshow('detected circles',cimg)
    #  cv2.drawContours(im,contours,-1,(0,255,0),-1)
    
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




