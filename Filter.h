#ifndef __CV_FILTER_H__
#define __CV_FILTER_H__

#include <opencv2/opencv.hpp>

cv::Mat filter(const cv::Mat &image, cv::Vec3i convC, cv::Vec3i convR);
cv::Mat sobel(const cv::Mat &image);

#endif
