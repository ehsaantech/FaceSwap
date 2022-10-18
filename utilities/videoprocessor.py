import cv2
import logging

class VideoProcessor(object):
    def __init__(self, videoPath, outputPath, process):
        logging.info('Processing video from -> {}'.format(videoPath))
        self.process = process
        self.outputPath = outputPath
        self.video = cv2.VideoCapture(videoPath)
        self.writer = cv2.VideoWriter(outputPath, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), self.video.get(cv2.CAP_PROP_FPS), (int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        self.frameCount = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT)) + 1

    def start(self):
        frameIndex = 1
        while self.video.isOpened():
            logging.info('Processing frame {0} of {1}'.format(frameIndex, self.frameCount))
            frameIndex += 1
            readSuccess, dstImg = self.video.read()
            if readSuccess:
                processedImage = self.process(dstImg)
                self.writer.write(processedImage)
            else:
                break

        self.video.release()
        self.writer.release()
        logging.info('Done processing video at {}'.format(self.outputPath))