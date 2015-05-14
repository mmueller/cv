// Ex 3.1: Color balance Write a simple application to change the color
// balance of an image by multiplying each color value by a different
// user-specified constant.

#include <opencv2/opencv.hpp>

#include <iostream>

#define WINDOW_NAME "Color Balance"
#define SLIDER_NAME_R "Red Multiplier (x100)"
#define SLIDER_NAME_G "Green Multiplier (x100)"
#define SLIDER_NAME_B "Blue Multiplier (x100)"

// 1. Do you get different results if you take out the gamma transformation
// before or after doing the multiplication?
//
// Seems to amplify the effect of the scaling factor.
#define DO_GAMMA_TRANSFORM 1
#define GAMMA_EXPONENT 2.2

struct ColorBalanceData {
    cv::Mat originalImage;
    int percentB;
    int percentG;
    int percentR;
};

// 'percent' parameter is discarded and the three percent values in data are
// used instead.
void updateImage(int percent, void *untypedData) {
    ColorBalanceData *data = static_cast<ColorBalanceData *>(untypedData);
    cv::Mat modifiedImage;
    cv::Mat displayImage;
    int width = data->originalImage.size().width;
    int height = data->originalImage.size().height;

    // Work in a 32-bit float RGB space to minimize lossy operations.
    data->originalImage.convertTo(modifiedImage, CV_32FC3);
    
    if (DO_GAMMA_TRANSFORM) {
        cv::pow(modifiedImage, 1/GAMMA_EXPONENT, modifiedImage);
    }

    // Create the channel-wise (BGR) scale factor vector
    cv::Vec3f factor((float) data->percentB / 100,
                     (float) data->percentG / 100,
                     (float) data->percentR / 100);

    // Apply the scale factor at each element
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            cv::Point p(i, j);
            cv::Vec3f &values = modifiedImage.at<cv::Vec3f>(p);
            values = values.mul(factor);
        }
    }

    if (DO_GAMMA_TRANSFORM) {
        cv::pow(modifiedImage, GAMMA_EXPONENT, modifiedImage);
    }

    // Convert back to 8-bit BGR for display.
    modifiedImage.convertTo(displayImage, CV_8UC3);
    cv::imshow(WINDOW_NAME, displayImage);
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
    cv::imshow(WINDOW_NAME, image);

    // Using percent since we're stuck with integers in the highgui trackbar
    ColorBalanceData data = { image, 100, 100, 100 };
    cv::namedWindow(WINDOW_NAME, cv::WINDOW_AUTOSIZE);
    cv::createTrackbar(SLIDER_NAME_R, WINDOW_NAME, &data.percentR, 200,
                       updateImage, &data);
    cv::createTrackbar(SLIDER_NAME_G, WINDOW_NAME, &data.percentG, 200,
                       updateImage, &data);
    cv::createTrackbar(SLIDER_NAME_B, WINDOW_NAME, &data.percentB, 200,
                       updateImage, &data);
    for (;;) {
        // Everything happens in updateImage from here out.
        if (cv::waitKey(0) == 27) {
            break;
        }
    }

    return 0;
}
