#include "Histogram.h"

Histogram::Histogram() : blue {0}, green {0}, red {0} {
}

Histogram::Histogram(const cv::Mat &image) : blue {0}, green {0}, red {0} {
    cv::Size size = image.size();
    for (int x = 0; x < size.width; x++) {
        for (int y = 0; y < size.height; y++) {
            const cv::Vec3b &values = image.at<cv::Vec3b>(cv::Point(x, y));
            blue[values.val[0]]++;
            green[values.val[1]]++;
            red[values.val[2]]++;
        }
    }
}

Histogram::~Histogram() {
}

Histogram Histogram::cumulative() const {
    Histogram hOut;
    hOut.blue[0] = blue[0];
    hOut.green[0] = green[0];
    hOut.red[0] = red[0];
    for (size_t i = 0; i < 256; i++) {
        hOut.blue[i] = hOut.blue[i-1] + blue[i];
        hOut.green[i] = hOut.green[i-1] + green[i];
        hOut.red[i] = hOut.red[i-1] + red[i];
    }
    return hOut;
}

cv::Mat Histogram::draw() const {
    cv::Mat output(cv::Size(256, 100), CV_8UC3, cv::Scalar(0, 0, 0));
    size_t max = 0;

    // For normalization, find max (which we'll map to 100)
    for (unsigned i = 0; i < 256; i++) {
        if (max < red[i]) {
            max = red[i];
        }
        if (max < green[i]) {
            max = green[i];
        }
        if (max < blue[i]) {
            max = blue[i];
        }
    }

    // Now draw it
    for (unsigned i = 1; i < 256; i++) {
        for (unsigned c = 0; c < 3; c++) {
            cv::Point p1, p2;
            switch (c) {
                case 0:
                    p1 = cv::Point(i-1, 100-blue[i-1]*100/max);
                    p2 = cv::Point(i, 100-blue[i]*100/max);
                    break;
                case 1:
                    p1 = cv::Point(i-1, 100-green[i-1]*100/max);
                    p2 = cv::Point(i, 100-green[i]*100/max);
                    break;
                case 2:
                    p1 = cv::Point(i-1, 100-red[i-1]*100/max);
                    p2 = cv::Point(i, 100-red[i]*100/max);
                    break;
            }
            cv::LineIterator it(output, p1, p2, 8);
            for (int i = 0; i < it.count; i++, it++) {
                ((cv::Vec3b *)(*it))->val[c] = 255;
            }
        }
    }
    return output;
}
