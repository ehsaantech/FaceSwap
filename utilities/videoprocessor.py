import cv2
import logging

class VideoProcessor(object):
    def __init__(self, videoPath, outputPath, process):
        logging.info('Processing video from -> {}'.format(videoPath))
        self.process = process
        self.video = cv2.VideoCapture(videoPath)
        self.writer = cv2.VideoWriter(outputPath, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), self.video.get(cv2.CAP_PROP_FPS), (int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    def start(self):
        while self.video.isOpened():
            readSuccess, dstImg = self.video.read()
            if readSuccess:
                processedImage = self.process(dstImg)
                self.writer.write(processedImage)
            else:
                break

        self.video.release()
        self.writer.release()
        logging.info('Done processing video')