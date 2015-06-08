#ifndef __CV_FILTER_H__
#define __CV_FILTER_H__

#include <opencv2/opencv.hpp>

// Note to readers: These implementations are for me to gain familiarity with
// filtering by convolution, but you probably don't want to actually use them
// yourself (OpenCV's built-in versions of the same will be better and faster).

// Convolve with a separated 3x3 kernel
cv::Mat filter(const cv::Mat &image, cv::Vec3i convC, cv::Vec3i convR);

// Convolve with an arbitrary kernel (assumed to be matrix of float (CV_32F))
cv::Mat filter(const cv::Mat &image, const cv::Mat &kernel);

// Perform Sobel's edge detection on the given image
cv::Mat sobel(const cv::Mat &image);

// Two-dimensional gaussian function
float gaussian(int x, int y, int sigma=1);

cv::Mat gaussianKernel(cv::Size size, int sigma=1);

#endif
