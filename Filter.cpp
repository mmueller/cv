#include <opencv2/highgui/highgui.hpp>
#include <iostream>

#include "Filter.h"

#define WINDOW_NAME "Filtering example"

// convC: column vector of the separated convolution kernel
// convR: row vector of the separated convolution kernel
cv::Mat filter(const cv::Mat &image, cv::Vec3i convC, cv::Vec3i convR) {
    cv::Mat grayscale, padded;
    cv::Mat result = cv::Mat::zeros(image.size(), CV_8U);
    cv::cvtColor(image, grayscale, CV_BGR2GRAY);
    cv::copyMakeBorder(grayscale, padded, 1, 1, 1, 1, cv::BORDER_REPLICATE);
    cv::Size size = image.size();

    // With separable kernels it seems like you apply the row vector first
    for (int x = 0; x < size.width; x++) {
        for (int y = 0; y < size.height; y++) {
            int padX = x+1;
            int padY = y+1;
            result.at<uchar>(cv::Point(x, y)) = (
              padded.at<uchar>(cv::Point(padX-1, padY)) * convR[0] +
              padded.at<uchar>(cv::Point(padX,   padY)) * convR[1] +
              padded.at<uchar>(cv::Point(padX+1, padY)) * convR[2]
            ) / 9;
        }
    }
    // Then the column vector. Need to re-pad the image.
    cv::copyMakeBorder(result, padded, 1, 1, 1, 1, cv::BORDER_REPLICATE);
    cv::normalize(result, result, 0x00, 0xff, cv::NORM_MINMAX);
    for (int x = 0; x < size.width; x++) {
        for (int y = 0; y < size.height; y++) {
            int padX = x+1;
            int padY = y+1;
            result.at<uchar>(cv::Point(x, y)) = (
              padded.at<uchar>(cv::Point(padX, padY-1)) * convC[0] +
              padded.at<uchar>(cv::Point(padX, padY))   * convC[1] +
              padded.at<uchar>(cv::Point(padX, padY+1)) * convC[2]
            ) / 9;
        }
    }
    cv::normalize(result, result, 0x00, 0xff, cv::NORM_MINMAX);
    cv::Mat colorResult;
    cv::cvtColor(result, colorResult, CV_GRAY2BGR);
    return colorResult;
}

cv::Mat sobel(const cv::Mat &image) {
    // Source: https://en.wikipedia.org/wiki/Sobel_operator
    cv::Vec3i convXC(1, 2, 1);
    cv::Vec3i convXR(-1, 0, 1);
    cv::Vec3i convYC(1, 0, -1);
    cv::Vec3i convYR(1, 2, 1);
    cv::Mat sobelX = filter(image, convXC, convXR);
    cv::Mat sobelY = filter(image, convYC, convYR);
    cv::Mat sobelXFloat, sobelYFloat;
    sobelX.convertTo(sobelXFloat, CV_32FC3);
    sobelY.convertTo(sobelYFloat, CV_32FC3);
    cv::Mat magnitude;
    cv::magnitude(sobelXFloat, sobelYFloat, magnitude);
    cv::Mat result;
    magnitude.convertTo(result, CV_8UC3);
    return result;
}

static void usage(const std::string &program) {
    std::cerr << "Usage:\n";
    std::cerr << "  " << program << " -i [image path]\n";
    std::cerr << "  " << program << " -v [video path]\n";
}

#ifdef FILTER_MAIN
int main(int argc, char *argv[]) {
    if (argc != 3) {
        usage(argv[0]);
        return 1;
    }
    if (strcmp(argv[1], "-i") == 0) {
        cv::Mat image = cv::imread(argv[2]);
        if (!image.data) {
            std::cerr << "imread: " << argv[2] << ": sadness\n";
            return 1;
        }
        cv::imshow(WINDOW_NAME, sobel(image));
        for (;;) {
            if (cv::waitKey(0) == 27) {
                break;
            }
        }
    } else if (strcmp(argv[1], "-v") == 0) {
        cv::VideoCapture capture(argv[2]);
        while (capture.isOpened()) {
            cv::Mat inFrame;
            if (!capture.grab()) {
                std::cerr << "grab failed\n";
                break;
            }
            capture.retrieve(inFrame);
            if (inFrame.empty()) {
                std::cerr << "empty frame\n";
                break;
            }
            cv::imshow(WINDOW_NAME, sobel(inFrame));
            if (cv::waitKey(20) == 27) {
                break;
            }
        }
    } else {
        usage(argv[0]);
        return 1;
    }
    return 0;
}
#endif