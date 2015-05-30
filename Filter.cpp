#include <opencv2/highgui/highgui.hpp>
#include <iostream>

#include "Filter.h"

#define WINDOW_NAME "Filtering example"

// convC: column vector of the separated convolution kernel
// convR: row vector of the separated convolution kernel
cv::Mat filter(const cv::Mat &image, cv::Vec3i convC, cv::Vec3i convR) {
    cv::Mat grayscale, padded;
    cv::Mat result = cv::Mat::zeros(image.size(), CV_32F);
    // Work in grayscale, 32-bit floating point space to avoid loss of precision
    cv::cvtColor(image, grayscale, CV_BGR2GRAY);
    cv::copyMakeBorder(grayscale, padded, 1, 1, 1, 1, cv::BORDER_REPLICATE);
    cv::Mat padded32F;
    padded.convertTo(padded32F, CV_32F);
    cv::Size size = image.size();

    // With separable kernels it seems like you apply the row vector first
    for (int x = 0; x < size.width; x++) {
        for (int y = 0; y < size.height; y++) {
            int padX = x+1;
            int padY = y+1;
            result.at<float>(cv::Point(x, y)) = (
              padded32F.at<float>(cv::Point(padX-1, padY)) * convR[0] +
              padded32F.at<float>(cv::Point(padX,   padY)) * convR[1] +
              padded32F.at<float>(cv::Point(padX+1, padY)) * convR[2]
            );
        }
    }
    // Then the column vector. Need to re-pad the image.
    cv::copyMakeBorder(result, padded32F, 1, 1, 1, 1, cv::BORDER_REPLICATE);
    for (int x = 0; x < size.width; x++) {
        for (int y = 0; y < size.height; y++) {
            int padX = x+1;
            int padY = y+1;
            result.at<float>(cv::Point(x, y)) = (
              padded32F.at<float>(cv::Point(padX, padY-1)) * convC[0] +
              padded32F.at<float>(cv::Point(padX, padY))   * convC[1] +
              padded32F.at<float>(cv::Point(padX, padY+1)) * convC[2]
            );
        }
    }
    cv::Mat result8U;
    result.convertTo(result8U, CV_8U);
    cv::Mat colorResult;
    cv::cvtColor(result, colorResult, CV_GRAY2BGR);
    return colorResult;
}

cv::Mat filter(const cv::Mat &image, const cv::Mat &kernel) {
    cv::Size size = image.size();
    cv::Size kSize = kernel.size();
    // Wow, OpenCV doesn't make it possible to be agnostic to type...
    if (image.channels() != 3) {
        std::cerr << "filter only supports 3-channel images right now\n";
        return image.clone();
    }
    if (kernel.type() != CV_32F) {
        std::cerr << "filter only supports CV_32F kernels right now\n";
        return image.clone();
    }
    if (kSize.width % 2 == 0 || kSize.height % 2 == 0) {
        std::cerr << "filter only supports kernels with odd dimensions\n";
        return image.clone();
    }
    cv::Mat floatImage;
    image.convertTo(floatImage, CV_32FC3);
    cv::Mat result(size, CV_32FC3);
    for (int x = 0; x < size.width; x++) {
        for (int y = 0; y < size.height; y++) {
            for (int c = 0; c < image.channels(); c++) {
                float value = 0;
                for (int i = 0; i < kSize.width; i++) {
                    for (int j = 0; j < kSize.height; j++) {
                        // Bounding srcX and srcY this way is basically the
                        // same as using padding replicate, but without
                        // constructing an entire copy of the image.
                        int srcX = x + i - kSize.width/2;
                        if (srcX < 0) { srcX = 0; }
                        if (srcX >= size.width) { srcX = size.width-1; }
                        int srcY = y + j - kSize.height/2;
                        if (srcY < 0) { srcY = 0; }
                        if (srcY >= size.height) { srcY = size.height-1; }
                        cv::Point imgPoint(srcX, srcY);
                        cv::Point kPoint(kSize.width-i-1, kSize.height-j-1);
                        value += floatImage.at<cv::Vec3f>(imgPoint).val[c] *
                                 kernel.at<float>(kPoint);

                    }
                }
                result.at<cv::Vec3f>(cv::Point(x, y)).val[c] = value;
            }
        }
    }
    cv::Mat ucharResult;
    result.convertTo(ucharResult, CV_8UC3);
    return ucharResult;
}

cv::Mat sobel(const cv::Mat &image) {
    // Source: https://en.wikipedia.org/wiki/Sobel_operator
    cv::Vec3i convXC(1, 2, 1);
    cv::Vec3i convXR(1, 0, -1);
    cv::Vec3i convYC(1, 0, -1);
    cv::Vec3i convYR(1, 2, 1);
    cv::Mat sobelX = filter(image, convXC, convXR);
    cv::Mat sobelY = filter(image, convYC, convYR);
    // Technically I think there is supposed to be a 1/4 scale factor applied
    // to the sobel kernels, but no one seems to do this in practice.
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

static cv::Mat identityKernel() {
    cv::Mat identity = cv::Mat::zeros(3, 3, CV_32F);
    identity.at<float>(cv::Point(1, 1)) = 1;
    return identity;
}

static cv::Mat boxKernel(cv::Size size) {
    cv::Mat box = cv::Mat::ones(size, CV_32F);
    box /= size.width * size.height;
    return box;
}

static cv::Mat gaussianKernel() {
    cv::Mat gaussian = (cv::Mat_<float>(5, 5) <<
        1.0,  4.0,  7.0,  4.0, 1.0,
        4.0, 16.0, 26.0, 16.0, 4.0,
        7.0, 26.0, 41.0, 26.0, 7.0,
        4.0, 16.0, 26.0, 16.0, 4.0,
        1.0,  4.0,  7.0,  4.0, 1.0);
    return gaussian / 273.0;
}

static cv::Mat rightShiftKernel() {
    cv::Mat rightShift = (cv::Mat_<float>(5, 5) <<
        0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 1.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0);
    return rightShift;
}

static cv::Mat unsharpKernel() {
    cv::Mat unsharp = gaussianKernel();
    unsharp *= -1;
    unsharp.at<float>(cv::Point(2, 2)) += 2;
    return unsharp;
}

#ifdef FILTER_MAIN
int main(int argc, char *argv[]) {
    if (argc != 3) {
        usage(argv[0]);
        return 1;
    }

    // Open up the source image or video
    cv::Mat image;
    cv::VideoCapture capture;
    if (strcmp(argv[1], "-i") == 0) {
        image = cv::imread(argv[2]);
        if (!image.data) {
            std::cerr << "imread: " << argv[2] << ": sadness\n";
            return 1;
        }
    } else if (strcmp(argv[1], "-v") == 0) {
        capture.open(argv[2]);
        if (!capture.isOpened()) {
            std::cerr << "VideoCapture::open failed\n";
            return 1;
        }
    } else {
        usage(argv[0]);
        return 1;
    }

    // Main loop
    std::cerr << "Use keys in the display window to control filtering:\n"
              << std::endl
              << "  i: Identity filter (default)\n"
              << "  b: Box filter\n"
              << "  g: Gaussian filter\n"
              << "  r: Right shift\n"
              << "  u: Unsharp filter based on Gaussian\n"
              << std::endl
              << "Press ESC to quit.\n";
    cv::Mat kernel = identityKernel();
    while (true) {
        char lastKeyPress;
        if (image.data) {
            cv::imshow(WINDOW_NAME, filter(image, kernel));
            lastKeyPress = cv::waitKey(0);
        } else {
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
            cv::imshow(WINDOW_NAME, filter(inFrame, kernel));
            lastKeyPress = cv::waitKey(10);
        }
        switch (lastKeyPress) {
            case 'i': kernel = identityKernel(); break;
            case 'b': kernel = boxKernel(cv::Size(5, 5)); break;
            case 'g': kernel = gaussianKernel(); break;
            case 'r': kernel = rightShiftKernel(); break;
            case 'u': kernel = unsharpKernel(); break;
        }
        if (lastKeyPress == 27) {
            break;
        }
    }
    return 0;
}
#endif
