// Utilities for computing, manipulating, and drawing histograms.

#include <opencv2/opencv.hpp>

// For now, assumes BGR, 8 bit unsigned channels
class Histogram {

  public:
    Histogram();

    // Input image MUST be BGR (8UC3)
    Histogram(const cv::Mat &image);
    ~Histogram();

    // Integrate this histogram and return a new, cumulative histogram.
    Histogram cumulative() const;

    // Draw the histogram (currently hard-coded to return a 255 x 100 image)
    cv::Mat draw() const;

    // Number of pixels with the given intensity value (per channel)
    size_t blue[256];
    size_t green[256];
    size_t red[256];
};
