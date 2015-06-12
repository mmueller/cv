#include <opencv2/opencv.hpp>
#include <limits>
#include "Filter.h"

#define WINDOW_NAME "Interest point detector"

// I made these up
#define MORAVEC_WINDOW_SIZE 15
#define MORAVEC_THRESHOLD   (MORAVEC_WINDOW_SIZE * MORAVEC_WINDOW_SIZE * 40)

#define HARRIS_WINDOW_SIZE 3
#define HARRIS_THRESHOLD 50000

typedef std::list<cv::Point> PointList;

// Compute Sum of Squared Differences of two regions (must be same size).
// Currently expects single channel 32-bit float.
float ssd(const cv::Mat &r1, const cv::Mat &r2) {
    cv::Size size = r1.size();
    if (size != r2.size()) {
        std::cerr << "ssd called on different sized patches\n";
        return -1;
    }
    if (r1.type() != CV_32F || r2.type() != CV_32F) {
        std::cerr << "ssd called with type other than CV_32F\n";
        return -1;
    }
    float result = 0;
    for (int x = 0; x < size.width; x++) {
        for (int y = 0; y < size.height; y++) {
            cv::Point p(x, y);
            float diff = r1.at<float>(p) - r2.at<float>(p);
            result += diff*diff;
        }
    }
    return result;
}

static cv::Rect rectAtPoint(cv::Point p, int windowSize) {
    return cv::Rect(p.x-windowSize/2, p.y-windowSize/2,
                    windowSize, windowSize);
}

// Return 8 connected points (neighbors of) p
static PointList getNeighbors(cv::Point p) {
    int neighborOffsets[][2] = {
        {-1, -1}, {0, -1}, {1, -1},
        {-1,  0},          {1,  0},
        {-1,  1}, {0,  1}, {1,  1}
    };
    PointList result;
    for (int n = 0; n < 8; n++) {
        result.push_back(cv::Point(p.x+neighborOffsets[n][0],
                                   p.y+neighborOffsets[n][1]));
    }
    return result;
}

// Moravec corner detection: my cheesy version
static PointList moravec(const cv::Mat &image) {
    const int windowSize = MORAVEC_WINDOW_SIZE;
    cv::Mat grayscale;
    cv::Mat input; // We'll actually work with this one
    cv::cvtColor(filter(image, gaussianKernel(cv::Size(5, 5))),
                 grayscale, CV_BGR2GRAY);
    grayscale.convertTo(input, CV_32F);
    cv::Size size = input.size();

    // Define boundaries for points under consideration (must fit in window
    // centered on point).
    int minX = windowSize/2;
    int minY = windowSize/2;
    int maxX = size.width - windowSize/2 - 1;
    int maxY = size.height - windowSize/2 - 1;

    // First: Compute corner strength at every pixel in the image
    cv::Mat cornerStrength = cv::Mat::zeros(size, CV_32F);
    for (int x = minX; x <= maxX; x++) {
        for (int y = minY; y <= maxY; y++) {
            cv::Rect r1 = rectAtPoint(cv::Point(x, y), windowSize);
            float minssd = std::numeric_limits<float>::infinity();
            PointList neighbors = getNeighbors(cv::Point(x, y));
            PointList::const_iterator p;
            for (p = neighbors.begin(); p != neighbors.end(); ++p) {
                if (p->x < minX || p->x > maxX ||
                    p->y < minY || p->y > maxY) {
                    continue;
                }
                cv::Rect r2 = rectAtPoint(*p, windowSize);
                float diff = ssd(input(r1), input(r2));
                if (diff < minssd) {
                    minssd = diff;
                }
            }
            cornerStrength.at<float>(cv::Point(x, y)) = minssd;
        }
    }
        
    // Second: Scan corner strength map for local maxima.
    PointList result;
    for (int x = minX; x <= maxX; x++) {
        for (int y = minY; y <= maxY; y++) {
            cv::Point p1(x, y);
            bool isMax = true;
            float s1 = cornerStrength.at<float>(p1);
            if (s1 < MORAVEC_THRESHOLD) {
                continue;
            }
            PointList neighbors = getNeighbors(p1);
            PointList::const_iterator p2;
            for (p2 = neighbors.begin(); p2 != neighbors.end(); ++p2) {
                float s2 = cornerStrength.at<float>(*p2);
                if (s1 < s2) {
                    isMax = false;
                    break;
                }
            }
            if (isMax) {
                result.push_back(p1);
            }
        }
    }
    
    return result;
}

static cv::Mat computeHarrisMatrix(const cv::Mat &dx, const cv::Mat &dy,
                                   int x, int y) {
    int winSize = HARRIS_WINDOW_SIZE;
    cv::Mat m = cv::Mat::zeros(cv::Size(2, 2), CV_32F);
    for (int i = 0; i < winSize; i++) {
        for (int j = 0; j < winSize; j++) {
            cv::Point p(x + i - winSize/2, y + j - winSize/2);
            float Ix = dx.at<float>(p);
            float Iy = dy.at<float>(p);
            m.at<float>(0, 0) += Ix * Ix;
            m.at<float>(0, 1) += Ix * Iy;
            m.at<float>(1, 0) += Ix * Iy;
            m.at<float>(1, 1) += Iy * Iy;
        }
    }
    return m;
}

static PointList harris(const cv::Mat &image) {
    PointList result;
    cv::Mat grayscale;
    cv::Mat input; // We'll actually work with this one
    cv::cvtColor(filter(image, gaussianKernel(cv::Size(5, 5))),
                 grayscale, CV_BGR2GRAY);
    grayscale.convertTo(input, CV_32F);
    cv::Size size = input.size();
    int winSize = HARRIS_WINDOW_SIZE;

    // Compute first derivative in x- and y-direction
    cv::Mat dx, dy;
    cv::Scharr(input, dx, -1, 1, 0);
    cv::Scharr(input, dy, -1, 0, 1);
    // harris operator applied to input
    cv::Mat harrisMat = cv::Mat::zeros(size, CV_32F);

    for (int x = winSize/2; x < size.width - winSize/2; x++) {
        for (int y = winSize/2; y < size.height - winSize/2; y++) {
            cv::Mat m = computeHarrisMatrix(dx, dy, x, y);
            float f = cv::determinant(m) / cv::trace(m)[0];
            harrisMat.at<float>(cv::Point(x, y)) = f;
        }
    }

    cv::threshold(harrisMat, harrisMat, HARRIS_THRESHOLD, -1,
                  cv::THRESH_TOZERO);

    // Find local maxima
    for (int x = winSize/2; x < size.width - winSize/2; x++) {
        for (int y = winSize/2; y < size.height - winSize/2; y++) {
            cv::Point p(x, y);
            bool localMaximum = true;
            PointList neighbors = getNeighbors(p);
            PointList::const_iterator n;
            for (n = neighbors.begin(); n != neighbors.end(); ++n) {
                if (harrisMat.at<float>(p) <= harrisMat.at<float>(*n)) {
                    localMaximum = false;
                    break;
                }
            }
            if (localMaximum) {
                result.push_back(p);
            }
        }
    }
    return result;
}

static cv::Mat renderInterestPoints(const PointList &points,
                                    const cv::Mat &image, cv::Scalar color) {
    cv::Mat result;
    image.copyTo(result);
    for (PointList::const_iterator p = points.begin(); p != points.end(); ++p) {
        cv::line(result, cv::Point(p->x, p->y-2), cv::Point(p->x, p->y+2),
                 color);
        cv::line(result, cv::Point(p->x-2, p->y), cv::Point(p->x+2, p->y),
                 color);
    }
    return result;
}

static void usage(const std::string &program) {
    std::cerr << "Usage:\n"
              << "  " << program << " -i [image path]\n"
              << "  " << program << " -v [video path]\n";
}

#ifdef INTEREST_MAIN
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
            std::cerr << "imread: " << argv[2] << ": nada\n";
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
    std::cerr << "Press ESC in the image window to quit.\n";
    while (true) {
        char lastKeyPress = -1;
        if (image.data) {
            cv::Mat result = image.clone();
            std::cerr << "Computing Harris interest points... ";
            result = renderInterestPoints(harris(image), result,
                                          cv::Scalar(0, 0, 255));
            std::cerr << "Done.\n";
            std::cerr << "Computing Moravec interest points... ";
            result = renderInterestPoints(moravec(image), result,
                                          cv::Scalar(0, 255, 0));
            std::cerr << "Done.\n";
            cv::imshow(WINDOW_NAME, result);
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
            cv::Mat result = renderInterestPoints(harris(inFrame), inFrame,
                                                  cv::Scalar(0, 0, 255));
            cv::imshow(WINDOW_NAME, result);
            lastKeyPress = cv::waitKey(1);
        }
        if (lastKeyPress == 27) {
            break;
        }
    }

    return 0;
}
#endif
