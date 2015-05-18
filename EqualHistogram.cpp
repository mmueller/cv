// Ex 3.6: Histogram equalization
// Compute the gray level (luminance) histogram for an image and equalize it
// so that the tones look better (and the image is less sensitive to exposure
// settings).
//
// Note: Not actually using luminance yet.

#include <opencv2/opencv.hpp>

#include <iostream>

#include "Histogram.h"

#define WINDOW_NAME "Histogram Equalizer"

cv::Mat equalHistogram(const cv::Mat &image) {
    cv::Mat equalized(image.size(), CV_8UC3);
    Histogram h = Histogram(image).cumulative();
    size_t totalRed = h.red[255];
    size_t totalGreen = h.green[255];
    size_t totalBlue = h.blue[255];
    for (int x = 0; x < image.size().width; x++) {
        for (int y = 0; y < image.size().height; y++) {
            cv::Point p(x, y);
            const cv::Vec3b &origValues = image.at<cv::Vec3b>(p);
            cv::Vec3b &newValues = equalized.at<cv::Vec3b>(p);
            newValues[0] = (double)(h.blue[origValues[0]]*255)/totalBlue+0.5;
            newValues[1] = (double)(h.green[origValues[1]]*255)/totalGreen+0.5;
            newValues[2] = (double)(h.red[origValues[2]]*255)/totalRed+0.5;
        }
    }
    return equalized;
}

cv::Mat makeDisplayImage(const cv::Mat &image, const cv::Mat &imageHist,
                         const cv::Mat &equalized,
                         const cv::Mat &equalizedHist) {
    cv::Size inSize = image.size();
    cv::Size outSize(inSize.width * 2 + 2, inSize.height);
    cv::Mat displayImage(outSize, CV_8UC3, cv::Scalar(0, 0, 0));

    image.copyTo(displayImage(cv::Rect(cv::Point(0, 0), inSize)));
    imageHist.copyTo(displayImage(cv::Rect(
                       cv::Point(0, inSize.height-imageHist.size().height),
                       imageHist.size())));
    cv::putText(displayImage, "input", cv::Point(0, 20),
                cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0, 255, 0));

    equalized.copyTo(displayImage(cv::Rect(cv::Point(inSize.width+2, 0),
                                           inSize)));
    equalizedHist.copyTo(displayImage(cv::Rect(
                     cv::Point(inSize.width,
                               inSize.height-equalizedHist.size().height),
                     equalizedHist.size())));
    cv::putText(displayImage, "output", cv::Point(inSize.width+2, 20),
                cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0, 255, 0));

    return displayImage;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " [image file]\n";
        return 1;
    }

    cv::Mat image = cv::imread(argv[1]);
    if (!image.data) {
        std::cerr << "imread: " << argv[1] << ": didn't work out\n";
        return 1;
    }

    std::cout << "Press ESC in the window to quit.\n";
    cv::Mat equalized = equalHistogram(image);
    cv::Mat displayImage = makeDisplayImage(
                             image, Histogram(image).draw(),
                             equalized, Histogram(equalized).draw());
    cv::namedWindow(WINDOW_NAME, cv::WINDOW_AUTOSIZE);
    cv::imshow(WINDOW_NAME, displayImage);
    for (;;) {
        if (cv::waitKey(0) == 27) {
            break;
        }
    }

    return 0;
}
