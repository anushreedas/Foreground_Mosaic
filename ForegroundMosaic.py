"""
ForegroundMosaic.py

This program builds a background model using MOG
and extracts moving foreground object at different positions
and creates foreground mosaic using the above.

@author: Anushree Das (ad1707)
"""
import cv2 as cv
from os import listdir
from os import path
import numpy as np


def findBrightest(imglist):
    """
    Return the brightest image from the list of images
    :param imglist: list of images
    :return: brightest image
    """
    # convert images to grayscale
    greyImages = []
    for frame in imglist:
        greyImages.append(cv.cvtColor(frame, cv.COLOR_BGR2GRAY))

    # calculate the sum of the pixels of each image
    # and find the index of image with maximum sum
    max = 0
    for i in range(len(greyImages)):
        if np.sum(greyImages[i]-greyImages[max]) > 0:
            max = i

    # return image with maximum sum
    return imglist[max]


def createMosaic(imglist):
    # find height of first image
    h = imglist[0].shape[0]

    # image resizing
    im_list_resize = [cv.resize(img,(int(img.shape[1] * h / img.shape[0]),h), interpolation =cv.INTER_CUBIC)
                      for img in imglist]

    # concat and return final image
    return cv.hconcat(im_list_resize)


def mog(filename):
    """
    Build background model using MOG and extract moving foreground object at different positions
    and create foreground mosaic
    :param filename: path to video file
    :return: None
    """
    print('Processing video..',filename)
    # create MOG2 BackgroundSubtractor object
    mog = cv.createBackgroundSubtractorMOG2(
        history=2000, varThreshold=16, detectShadows=True)

    # Create a capture object
    cap = cv.VideoCapture(cv.samples.findFileOrKeep(filename))

    # number of frames from which to find the brightest image for processing
    noframes = 6
    keep_processing = True
    # keep count of frames processed for building background model
    counter = 0

    print('Building background model..')
    # Read until video is completed or 500 frames are processed
    while (keep_processing):
        # if video file successfully open then read frame from video
        if (cap.isOpened):
            counter += 1
            # to store list of frames to select brightest of them
            frames = []
            for i in range(noframes):
                ret, frame = cap.read()
                # exit when we reach the end of the video or 500 frames are processed
                if ret == 0 or counter > 500:
                    keep_processing = False
                    continue
                frames.append(frame)

        if not frames:
            break

        # build background model using the brightest frame
        mog.apply(findBrightest(frames))
        # clear list for next set of images
        frames.clear()

    # get name of video
    name, file_extension = path.splitext(filename)

    # get background model
    bgmodel = mog.getBackgroundImage()
    # save background model
    cv.imwrite(name+'background.jpg', bgmodel)
    # show background model
    cv.imshow('Background Model', bgmodel)
    cv.waitKey(0)
    cv.destroyAllWindows()

    keep_processing = True
    # reset video
    cap = cv.VideoCapture(cv.samples.findFileOrKeep(filename))

    # list of foreground masks and foreground mosaic images
    foregroundMasks = []
    foregroundmosaic = []
    # set first foreground masks as a black image to use it later for calculating change map
    foregroundMasks.append(np.zeros((bgmodel.shape[0],bgmodel.shape[1]), dtype = "uint8"))
    # set first foreground mosaic image as the background model
    foregroundmosaic.append(bgmodel)

    print('Creating foreground mosaic..')
    # Read until video is completed or 500 frames are processed
    while (keep_processing):
        # if video file successfully open then read frame from video
        if (cap.isOpened):
            # to store list of frames to select brightest of them
            frames = []
            for i in range(noframes):
                ret, frame = cap.read()
                # exit when we reach the end of the video
                if ret == 0:
                    keep_processing = False
                    continue
                frames.append(frame)

            if not frames:
                break
            # get brightest frame
            frame = findBrightest(frames)
            # clear list for next set of images
            frames.clear()

            # apply MOG2 background subtraction on the frame
            fgmask = mog.apply(frame)

            # create frame for diver's mask
            # kernel for erosion
            erosionkernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 4))
            # apply erosion to remove small changes that happen in the background
            erosion = cv.erode(fgmask, erosionkernel, iterations=6)

            # preprocessing
            # remove big changes caused by water splash
            # or when the diver just enters the frame
            # or when diver comes out of the water after diving
            erosion[-350:, :] = 0
            erosion[-400:, :200] = 0
            erosion[:, :100] = 0
            erosion[:, -100:] = 0

            # kernel for dilation
            dilationkernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (100, 180))
            # apply dilation to make the region big enough to include the diver and
            # all his/her limbs, hairs, and other small features
            dilation = cv.dilate(erosion, dilationkernel, iterations=2)

            # threshold to remove any leftover big changes in the foreground
            # apart from the diver(being the largest change)
            ret, fgthresh = cv.threshold(dilation, 254, 255, cv.THRESH_TOZERO)

            # check if there is any foreground object present in this frame
            if cv.countNonZero(fgthresh) > 0:
                # if current frame and the last frame in the foreground mosaic don't overlap
                if cv.countNonZero(cv.bitwise_and(foregroundMasks[-1],foregroundMasks[-1],mask=fgthresh)) <= 0:
                    # get only the diver's mask from the foreground mask before all the processing
                    # using the frame we created above
                    divermask = cv.bitwise_and(fgmask, fgmask, mask=fgthresh)

                    # kernel for erosion
                    erosionkernel = np.ones((5, 5), np.uint8)
                    # kernel for dilation
                    dilationkernel = np.ones((12, 10), np.uint8)
                    # apply erosion to mask
                    divermask = cv.erode(divermask, erosionkernel, iterations=1)
                    divermask = cv.dilate(divermask, dilationkernel, iterations=3)

                    # store the diver's mask for this frame for later use
                    foregroundMasks.append(divermask)
                    frameimg = frame.copy()
                    # extract foreground object(diver) image
                    fgimg = cv.bitwise_and(frameimg, frameimg, mask=divermask)

                    mosaicimg = bgmodel.copy()
                    # create a “change map” of where the foreground has been changed
                    mosaicimg[:, :, 1] -= (cv.bitwise_and(mosaicimg[:, :, 1], mosaicimg[:, :, 1], mask=foregroundMasks[-2]))
                    mosaicimg[:, :, 0] -= (cv.bitwise_and(mosaicimg[:, :, 0], mosaicimg[:, :, 0], mask=foregroundMasks[-2]))

                    # remove part of image where foreground object will go
                    mask_inv = (cv.bitwise_not(divermask))
                    mosaicimg = cv.bitwise_and(mosaicimg,mosaicimg,mask = mask_inv)
                    # add this new foreground region(diver) to the foreground mosaic
                    mosaicimg += fgimg

                    # store the new foreground mosaic
                    foregroundmosaic.append(mosaicimg)

    # concat all foreground mosaic images
    foregroundmosaic = createMosaic(foregroundmosaic)
    cv.imwrite(name + '_foregroundmosaic.jpg', foregroundmosaic)
    # foregroundmosaic = cv.imread(name + '_foregroundmosaic.jpg')
    cv.imshow('foregroundmosaic',foregroundmosaic)
    cv.waitKey(0)
    cv.destroyAllWindows()


def main():
    # src = 'diving_videos/IMG_1794__2175.m4v'
    # src = 'diving_videos'

    # list of  video formats
    ext = [".mp4", ".avi", ".mov",".m4v"]

    # take filename or directory name from user
    src = input('Enter filename or directory name:')

    if path.exists(src):
        # if input is directory
        if path.isdir(src):
            # create foreground mosaic for all videos in that directory one by one
            for filename in listdir(src):
                _, file_extension = path.splitext(filename)
                if file_extension.lower() in ext:
                    mog(src + "/" + filename)
        else:
            # create foreground mosaic for video filename provided by the user
            mog(src)
    else:
        print("File doesn't exist")


if __name__ == "__main__":
    main()