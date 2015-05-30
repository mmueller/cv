#ifndef __CV_FILTER_H__
#define __CV_FILTER_H__

#include <opencv2/opencv.hpp>

// Convolve with a separated 3x3 kernel
cv::Mat filter(const cv::Mat &image, cv::Vec3i convC, cv::Vec3i convR);

// Convolve with an arbitrary kernel (assumed to be matrix of float (CV_32F))
cv::Mat filter(const cv::Mat &image, const cv::Mat &kernel);

// Perform Sobel's edge detection on the given image
cv::Mat sobel(const cv::Mat &image);

#endif
