import logging
import cv2
from face_swap import apply_mask, correct_colours, mask_from_points, warp_image_3d
import numpy
from compare import calculateDelaunayTriangles, warpTriangle
from utilities.cli import takeCliOptionSelection
from utilities.image import readImage, writeImage
from utilities.videoprocessor import VideoProcessor
from images.imagelist import imageList as images
from videos.videolist import videoList as videos
from detectors.detectorlist import detectorList as detectors
from predictors.predictorlist import predictorList as predictors

def faceswapCli(outputDirName) -> str:
  outputDirPath = '{0}/{1}'.format(outputDirName, 'faceswap')
  try:
    detectorSelection = takeCliOptionSelection([d[1] for d in detectors], 'detector')
    predictorSelection = takeCliOptionSelection([p[1] for p in predictors], 'predictor')
    sourceSelection = takeCliOptionSelection([img[0] for img in images], 'source image')

    imagesCopy = images.copy()
    imagesCopy.pop(sourceSelection)
    targetOptions = imagesCopy + videos
    targetSelection = takeCliOptionSelection([t[0] for t in targetOptions], 'target')

    (detectorClass, detectorName) = detectors[detectorSelection]
    (predictorClass, predictorName) = predictors[predictorSelection]
    (srcPath, srcName, _) = images[sourceSelection]
    (targetPath, targetName, targetExtension) = targetOptions[targetSelection]

    detector = detectorClass()
    predictor = predictorClass()

    outputPath = '{0}/{1}-{2}-{3}-{4}.{5}'.format(outputDirPath, srcName, targetName, predictorName, detectorName, targetExtension)

    isVideo = targetExtension == 'mp4'
    if isVideo:
      faceSwapInVideo(detector, predictor, srcPath, targetPath, outputPath)
    else:
      faceSwap(detector, predictor, srcPath, targetPath, outputPath)

  except Exception as exception:
    logging.warning(exception)
    return

  return outputDirPath

def faceSwap(detector, predictor, srcPath, targetPath, outputPath):
  srcImage = readImage(srcPath)
  srcFaces = detector.detectFacesInImage(srcImage)
  if len(srcFaces) != 1:
    logging.info('source image should have 1 face. Aborting!')
    return
  srcFaceFeatures = predictor.predictFeaturesInImage(srcImage, srcFaces[0])

  targetImage = readImage(targetPath)
  targetFaces = detector.detectFacesInImage(targetImage)
  if len(targetFaces) == 0:
      return writeImage(targetImage, outputPath)

  targetFacesWithFeatures = [predictor.predictFeaturesInImage(targetImage, face) for face in targetFaces]

  outputImage = targetImage
  for targetFaceFeatures in targetFacesWithFeatures:
    outputImage = _faceSwap(srcImage, srcFaceFeatures, outputImage, targetFaceFeatures)
  writeImage(outputImage, outputPath)

def faceSwapInVideo(detector, predictor, srcPath, targetPath, outputPath):
  srcImage = readImage(srcPath)
  srcFaces = detector.detectFacesInImage(srcImage)
  if len(srcFaces) != 1:
    logging.info('source image should have 1 face. Aborting!')
    return
  srcFaceFeatures = predictor.predictFeaturesInImage(srcImage, srcFaces[0])
  
  def process(image):
    faces = detector.detectFacesInImage(image)
    if len(faces) == 0:
      return image

    facesWithFeatures = [predictor.predictFeaturesInImage(image, face) for face in faces]
    outputImage = image
    for targetFaceFeatures in facesWithFeatures:
      outputImage = _faceSwap(srcImage, srcFaceFeatures, outputImage, targetFaceFeatures)
    return outputImage
  
  videoprocessor = VideoProcessor(targetPath, outputPath, process)
  videoprocessor.start()

def _faceSwap(srcImage, srcFaceFeatures, targetImage, targetFacesWithFeatures):
  outputImage = face_swap_custom(srcImage, targetImage, srcFaceFeatures, targetFacesWithFeatures)
  return outputImage

def face_swap_custom(srcImage, targetImage, srcFaceFeatures, targetFaceFeatures):
  # h, w = targetImage.shape[:2]

  # srcHull = srcFaceFeatures
  # targetHull = targetFaceFeatures
  # # srcHull = []
  # # targetHull = []
  # # hullIndex = cv2.convexHull(targetFaceFeatures, returnPoints = False)
  # # for i in range(0, len(hullIndex)):
  # #   srcHull.append(srcFaceFeatures[int(hullIndex[i])])
  # #   targetHull.append(targetFaceFeatures[int(hullIndex[i])])

  # # 3d warp
  # warped_src_face = warp_image_3d(srcImage, srcHull, targetHull, (h, w))
  
  # # Mask for blending
  # mask = mask_from_points((h, w), targetHull, erode_flag=False)
  # mask_src = numpy.mean(warped_src_face, axis=2) > 0
  # mask = numpy.asarray(mask * mask_src, dtype=numpy.uint8)

  # # Correct color
  # warped_src_face = apply_mask(warped_src_face, mask)
  # dst_face_masked = apply_mask(targetImage, mask)
  # warped_src_face = correct_colours(dst_face_masked, warped_src_face, targetHull)
  
  # #Poisson Blending
  # r = cv2.boundingRect(mask)
  # center = ((r[0] + int(r[2] / 2), r[1] + int(r[3] / 2)))
  # output = cv2.seamlessClone(warped_src_face, targetImage, mask, center, cv2.NORMAL_CLONE)
  # return output


  warpedImage = targetImage.copy()

  srcHull = []
  targetHull = []
  hullIndex = cv2.convexHull(targetFaceFeatures, returnPoints = False)
  for i in range(0, len(hullIndex)):
    srcHull.append(srcFaceFeatures[int(hullIndex[i])])
    targetHull.append(targetFaceFeatures[int(hullIndex[i])])

  # Find delanauy traingulation for convex hull points
  h, w = targetImage.shape[:2]
  rect = (0, 0, w, h)
     
  dt = calculateDelaunayTriangles(rect, targetHull)
    
  if len(dt) == 0:
    return

  # Apply affine transformation to Delaunay triangles
  for i in range(0, len(dt)):
    t1 = []
    t2 = []
    
    #get points for img1, img2 corresponding to the triangles
    for j in range(0, 3):
      t1.append(srcHull[dt[i][j]])
      t2.append(targetHull[dt[i][j]])
    
    warpTriangle(srcImage, warpedImage, t1, t2)

  # Mask for blending
  mask = mask_from_points((h, w), targetFaceFeatures)
  mask_src = numpy.mean(warpedImage, axis=2) > 0
  mask = numpy.asarray(mask * mask_src, dtype=numpy.uint8)

  # Correct color
  warpedImage = apply_mask(warpedImage, mask)
  dst_face_masked = apply_mask(targetImage, mask)
  warpedImage = correct_colours(dst_face_masked, warpedImage, targetFaceFeatures)
  
  # Poisson Blending
  r = cv2.boundingRect(mask)
  center = ((r[0] + int(r[2] / 2), r[1] + int(r[3] / 2)))
  output = cv2.seamlessClone(warpedImage, targetImage, mask, center, cv2.NORMAL_CLONE)
  return output

  # # Calculate Mask
  # hull8U = []
  # for i in range(0, len(targetHull)):
  #     hull8U.append((targetHull[i][0], targetHull[i][1]))
  
  # mask = numpy.zeros(targetImage.shape, dtype = targetImage.dtype)  
  # writeImage(mask, "output/t2.jpg")
  
  # cv2.fillConvexPoly(mask, numpy.int32(hull8U), (255, 255, 255))
  # print('a')
  # r = cv2.boundingRect(numpy.float32([targetHull]))    
  # print('b')
  # center = ((r[0]+int(r[2]/2), r[1]+int(r[3]/2)))
  # print('c', center)
      
  # # # Clone seamlessly.
  # output = cv2.seamlessClone(numpy.uint8(warpedImage), targetImage, mask, center, cv2.NORMAL_CLONE)
  # writeImage(output, "output/t3.jpg")
  # return output
    