#!/usr/bin/python3

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pytesseract
import json
import sys
import os
import re

# the setrecursionlimit function is 
# used to modify the default recursion 
# limit set by python. Using this,  
# we can increase the recursion limit 
# to satisfy our needs   
sys.setrecursionlimit(10**6) 

debug = False
newlineChar = False
configJsonPath = "config.json"
if not os.path.isfile(configJsonPath):
  raise ValueError("could not find '{}'".format(configJsonPath))

index = 0
while index < len(sys.argv):
  arg = sys.argv[index]
  if arg[0] == '-':
    if arg == '--debug' or arg == '-d':
      debug = True
      print("Debug is {}".format(debug))
    elif arg[0:len("--newline")] == "--newline":
      _, value = arg.split('=')
      newlineChar = value
    elif arg == "-n":
      newlineChar = sys.argv[index+1]
      index += 1
  else:
    imagePath = arg
    if not os.path.isfile(imagePath):
      raise ValueError("image '{}' does not exist".format(imagePath))
  index += 1

with open(configJsonPath, 'r') as in_file:
  config = json.load(in_file)

pytesseract.pytesseract.tesseract_cmd = config["tesseract"]["path"]
customConfig = config["tesseract"]["options"]
if not newlineChar:
  newlineChar = config["newline"]



def debugImage(image, imagePath, imageNumber=1, title=None, grayscale=False):
  filepath, extension = os.path.splitext(imagePath)
  plt.figure(figsize=(10,10))
  if title:
    plt.title(title)
  if grayscale:
    plt.imshow(image, cmap='gray'); 
  else:
    plt.imshow(image); 
  plt.xticks([]); 
  plt.yticks([])
  plt.savefig('{}-{}.png'.format(filepath,imageNumber),bbox_inches='tight')
  plt.close()

def findTop(image, x, y, value, visited ):
  if y == 0:
    return 0
  top = y
  if value == image[y-1,x]:
    visited[y-1,x] = 1
    top = min(top, findTop(image, x, y-1, value, visited))
  if x < image.shape[1]-1 and visited[y,x+1] == 0:
    visited[y,x+1] = 1
    if value == image[y, x+1]:
      top = min(top, findTop(image, x+1, y, value, visited))
  if x > 0 and visited[y,x-1] == 0:
    visited[y,x-1] = 1
    if value == image[y, x-1]:
      top = min(top, findTop(image, x-1, y,value, visited))
  if y < image.shape[0] - 1 and visited[y+1,x] == 0:
    visited[y+1,x] = 1
    if value == image[y+1, x]:
      top = min(top, findTop(image, x, y+1, value, visited))
  return top

def findLeft(image, x, y, value, visited ):
  if x == 0:
    return 0
  left = x
  if value == image[y,x-1]:
    visited[y, x-1] = 1
    left = min(left, findLeft(image, x-1, y, value, visited))
  if y > 0 and visited[y-1,x] == 0:
    visited[y-1,x] = 1
    if value == image[y-1, x]:
      left = min(left, findLeft(image, x, y-1, value, visited))
  if y < image.shape[0]-1 and visited[y+1,x] == 0:
    visited[y+1,x] = 1
    if value == image[y+1, x]:
      left = min(left, findLeft(image, x, y+1, value, visited))
  if x < image.shape[1]-1 and visited[y,x+1] == 0:
    visited[y,x+1] = 1
    if value == image[y, x+1]:
      left = min(left, findLeft(image, x+1, y, value, visited))
  return left

def findBottom(image, x, y, value, visited ):
  if y >= image.shape[0]-1:
    return image.shape[0]-1
  bottom = y
  if value == image[y+1,x]:
    visited[y+1,x] = 1
    bottom = max(bottom, findBottom(image, x, y+1, value, visited))
  if x < image.shape[1]-1 and visited[y,x+1] == 0:
    visited[y,x+1] = 1
    if value == image[y, x+1]:
      bottom = max(bottom, findBottom(image, x+1, y,value, visited))
  if x > 0 and visited[y,x-1] == 0 :
    visited[y,x-1] = 1
    if value == image[y, x-1]:
      bottom = max(bottom, findBottom(image, x-1, y, value, visited))
  if y > 0 and visited[y-1,x] == 0:
    visited[y-1,x] = 1
    if value == image[y-1, x]:
      bottom = max(bottom, findBottom(image, x, y-1, value, visited))
  return bottom

def findRight(image, x, y, value, visited ):
  if x >= image.shape[1]-1:
    return image.shape[1]-1
  right = x
  if value == image[y,x+1]:
    visited[y,x+1] = 1
    right = max(right, findRight(image, x+1, y, value, visited))
  if y > 0 and visited[y-1,x] == 0:
    visited[y-1,x] = 1
    if value == image[y-1, x]:
      # print("RIGHT, GO UP {} {} {} {}".format(x,y, image.shape[0], right))
      right = max(right, findRight(image, x, y-1, value, visited))
  if y < image.shape[0]-1 and visited[y+1,x] == 0:
    visited[y+1,x] = 1
    if value == image[y+1, x]:
      # print("RIGHT, GO DOWN {} {} {}".format(x,y, image.shape[0]))
      right = max(right, findRight(image, x, y+1, value, visited))
  if x > 0 and visited[y,x-1] == 0:
    visited[y,x-1] = 1
    if value == image[y, x-1]:
      right = max(right, findRight(image, x-1, y, value, visited))
  return right

def _getBoundingBoxForPixel(image, x, y, value, visited):
  top = findTop(image, x, y, value, visited)
  visited = np.zeros(image.shape, dtype=np.int8)
  bottom = findBottom(image, x, y, value, visited)
  visited = np.zeros(image.shape, dtype=np.int8)
  left = findLeft(image, x, y, value, visited)
  visited = np.zeros(image.shape, dtype=np.int8)
  right = findRight(image, x, y, value, visited)
  return {
    "width": right - left, 
    "height": bottom - top, 
    "top": top, 
    "bottom": bottom, 
    "left": left, 
    "right": right,
    "visited": visited
  }

def getBoundingBoxForPixel(image, x, y, value):
  """
    Given a pixel position we then want to say what the maximum width is that its connected to
    and what the maximum height its connected to.  This will give us a bounding box for the
    pixel.

    This is currently subject to local tops e.g. if you are at the bottom left of an S
    then it won't be joined to the top of the S.
  """
  visited = np.zeros(image.shape, dtype=np.int8)
  return _getBoundingBoxForPixel(image, x, y, value, visited)
  
def setPixelsMovingToTop(image, x, y, oldvalue, newvalue, visited ):
  image[y,x] = newvalue
  visited[y,x] = 1
  if y > 0 and oldvalue == image[y-1,x]:
    visited[y-1,x] = 1
    setPixelsMovingToTop(image, x, y-1, oldvalue, newvalue, visited)
  if x < image.shape[1]-1 and oldvalue == image[y, x+1] and visited[y,x+1] ==0:
    visited[y,x+1] = 1
    setPixelsMovingToTop(image, x+1, y, oldvalue, newvalue, visited)
  if x > 0 and oldvalue == image[y, x-1] and visited[y,x-1] ==0:
    visited[y,x-1] = 1
    setPixelsMovingToTop(image, x-1, y, oldvalue, newvalue, visited)
  

def setPixelsMovingToLeft(image, x, y, oldvalue, newvalue, visited ):
  image[y,x] = newvalue
  visited[y,x] = 1
  if x > 0 and oldvalue == image[y,x-1]:
    visited[y,x-1] = 1
    setPixelsMovingToLeft(image, x-1, y, oldvalue, newvalue, visited)
  if y > 0 and oldvalue == image[y-1, x] and visited[y-1,x] ==0:
    visited[y-1,x] = 1
    setPixelsMovingToLeft(image, x, y-1, oldvalue, newvalue, visited)
  if y < image.shape[0]-1 and oldvalue == image[y+1, x] and visited[y+1,x] ==0:
    visited[y+1,x] = 1
    setPixelsMovingToLeft(image, x, y+1, oldvalue, newvalue, visited)

def setPixelsMovingToBottom(image, x, y, oldvalue, newvalue, visited ):
  image[y,x] = newvalue
  visited[y,x] = 1
  if y < image.shape[0]-1 and oldvalue == image[y+1,x]:
    visited[y+1,x] = 1
    setPixelsMovingToBottom(image, x, y+1, oldvalue, newvalue, visited)
  if x < image.shape[1]-1 and oldvalue == image[y, x+1] and visited[y,x+1] ==0:
    visited[y,x+1] = 1
    setPixelsMovingToBottom(image, x+1, y, oldvalue, newvalue, visited)
  if x > 0 and oldvalue == image[y, x-1] and visited[y,x-1] ==0:
    visited[y,x-1] = 1
    setPixelsMovingToBottom(image, x-1, y, oldvalue, newvalue, visited)

def setPixelsMovingToRight(image, x, y, oldvalue, newvalue, visited ):
  image[y,x] = newvalue
  visited[y,x] = 1
  if x < image.shape[1]-1 and oldvalue == image[y,x+1]:
    visited[y,x+1]=1
    setPixelsMovingToRight(image, x+1, y, oldvalue, newvalue, visited)
  if y > 0 and oldvalue == image[y-1, x] and visited[y-1,x] ==0:
    visited[y-1,x] = 1
    setPixelsMovingToRight(image, x, y-1, oldvalue, newvalue, visited)
  if y < image.shape[0]-1 and oldvalue == image[y+1, x] and visited[y+1,x] ==0:
    visited[y+1,x] = 1
    setPixelsMovingToRight(image, x, y+1, oldvalue, newvalue, visited)

def setBoundingBoxForPixel(image, x, y, oldvalue, newvalue):
  visited = np.zeros(image.shape, dtype=np.int8)
  setPixelsMovingToTop(image, x, y, oldvalue, newvalue, visited)
  setPixelsMovingToBottom(image, x, y, oldvalue, newvalue, visited)
  setPixelsMovingToLeft(image, x, y, oldvalue, newvalue, visited)
  setPixelsMovingToRight(image, x, y, oldvalue, newvalue, visited)


def extractText(image, imagePath):
  # grab the image dimensions
  h = image.shape[0]
  w = image.shape[1]

  if debug:
    debugImage(image, imagePath, 0, "Starting image")

  ## Remove noise and preserve edges
  ## diameter of pixel neighourhood
  ## sigmaColor - the larger this value the more distant the color in the neighbourhood will influence color
  ## sigmaSpace - the larger this value the more pixels further away will influence the colour blending
  image= cv2.bilateralFilter(image,5, 55,60)
  if debug:
    debugImage(image, imagePath, 1, "After filter")

  ## convert to greyscale
  image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  if debug:
    debugImage(image, imagePath, 2, "Conversion to greyscale", True)

  # Here are the possible options for thresholding THRESH_BINARY_INV
  # seems to work best.
  # It returns the threshold used and the image.
  thresholdLevel, image = cv2.threshold(image, 240, 255, cv2.THRESH_BINARY_INV)
  if debug:
    debugImage(image, imagePath, 3, "After thresholding (level used: {})".format(thresholdLevel), True) 
  
  # if we have more black than white then we have that as our expected text colour
  black = 0
  for y in range(0,h):
    for x in range(0,w):
      if image[y,x] == 0:
        black += 1
  if black / (h * w) > 0.8: # lots of black so text is white
    textColor = 255
    backgroundColor = 0
  else: # lots of white so text is black
    textColor = 0
    backgroundColor = 255

  # this is out custom code that captures a connected block within the image
  n = 300
  nn = 100
  if debug:
    print("Processing connected shapes")
  globalVisited = np.zeros(image.shape)
  for y in range(0,h):
    for x in range(0,w):
      if globalVisited[y,x] == 0 and image[y,x] == 0:
        bounds = getBoundingBoxForPixel(image,x,y, textColor)
        if bounds["height"] < 7 or (bounds["height"] / image.shape[0]) > 0.8:
          if debug:
            print("{}: ({},{}) w={} h={} top={} bottom={}".format(n, x, y, bounds["width"], bounds["height"], bounds["top"], bounds["bottom"]))
            debugImage(bounds["visited"], imagePath, n, "connected shape", True)
            n += 1
          setBoundingBoxForPixel(image, x, y, textColor, backgroundColor)
          if debug:
            debugImage(image, imagePath, nn, "image after removal of connected shape", True)
            nn += 1
        globalVisited =  np.logical_or(globalVisited, bounds["visited"])
  
  if debug:
    debugImage(image, imagePath, 4, "Remove small items and large blocks of stuff", True)

  # TODO tesseract is not great at finding spaces e.g. I LOVE and IT LOOK
  #      after the last bit of processing we have a pretty good idea of the space
  #      between letters so we could write our own space detector...

  return image


image = np.array(Image.open(imagePath))
image = extractText(image, imagePath)
text = pytesseract.image_to_string(image, lang='eng', config=customConfig)
text = text.strip()                 ## remove last newline
text = re.sub("\n\n+", "\n", text)  ## replace multiple newlines with a single one
print(text.replace('\n', newlineChar))  ## convert newline to <br/>

