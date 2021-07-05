#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include "cmath";

using namespace std;
using namespace cv;



static std::string BASE_PATH = "C:\\Users\\lars\\Documents\\Uni\\SS21\\gdv\\Praktikum5";

static double sinLUT[360], cosLUT[360];
const static double deg2grad = 0.0174533; // taken from googles calculator tool

double sine(int angle) {
    if (abs(angle) < 360) angle = angle % 360;

    if (angle < 0) {
        angle = abs(angle);
        return -1*sinLUT[angle];
    }
    return sinLUT[angle];
}

double cosine(int angle) {
    if (abs(angle) < 360) angle = angle % 360;

    if (angle < 0) {
        angle = abs(angle);
        return -1*cosLUT[angle];
    }
    return cosLUT[angle];
}

void fillLUT() {
    for (int i = 0; i < 360; i++) {
        sinLUT[i] = sin(deg2grad * i);
        cosLUT[i] = cos(deg2grad * i);
    }
}

void aufgabe1() {
    // Liniendetektion mit Hough-Raeumen 
    // Schritt 1 : Convert to grayscale
    cv::Mat img = cv::imread(BASE_PATH + "\\Zaun.png"); // Read the file
    if (img.empty()) // Check for invalid input
    {
        std::cout << "Could not open or find the frame" << std::endl;
        return;
    }

    // Schritt 2 : Canny Edge detection 
    cv::Mat img_gray;
    cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
    cv::Mat edges;
    double threshold = 150;
    cv::Canny(img_gray, edges, threshold, threshold * 2.5);

    // Schritt 3 : Akkumulatorarray 
    int diag = round(cv::sqrt((edges.rows * edges.rows) + (edges.cols * edges.cols)));
    // Schritt 4
    Mat acc = cv::Mat::zeros(180, 2*diag+1, CV_32S);

    for (int y = 0; y < edges.rows; y++) {
        for (int x = 0; x < edges.cols; x++) {
            if ((int)edges.at<char>(y, x) != 0) {
                for (int a = -90; a < 90; a++) {
                    int d = (x * cosine(a)) + (y * sine(a));
                    acc.at<int>(a+90, d + diag) += 1;
                }
            }
        }
    }

    int max_vals[5] = { 0,0,0,0,0 };
    std::pair<int, int> max_vals_content[5];

    for (int i = 0; i < 180; i++) {
        for (int j = 0; j < 2*diag+1; j++) {
            for (int c = 0; c < 5; c++) {
                if (max_vals[c] < acc.at<int>(i, j)) {
                    max_vals[c] = acc.at<int>(i, j);
                    max_vals_content[c] = std::pair<int, int>(i-90, j-diag);
                    break;
                }
            }
        }
    }

    int x0 = 0, x1 = edges.rows - 1;
    for (int i = 0; i < 5; i++) {
        int d = max_vals_content[i].second;
        int angle = max_vals_content[i].first;
        int y0 = (d - x0 * cosine(angle)) / sine(angle);
        int y1 = (d - x1 * cosine(angle)) / sine(angle);
        cv::line(img, cv::Point(x0, y0), cv::Point(x1, y1), cv::Scalar(255, 255, 0));
    }

    acc.convertTo(acc, CV_8U, 1); // [0..1] -> [0..255] range
    cv::namedWindow("Hough", cv::WINDOW_AUTOSIZE); // Create a window for display.
    cv::imshow("Hough", acc);
    cv::waitKey(0);

    cv::namedWindow("Res", cv::WINDOW_AUTOSIZE); // Create a window for display.
    cv::imshow("Res", img);
    cv::waitKey(0);
}


void aufgabe2() {
    // Dilation und Erosion
    cv::Mat img = cv::imread(BASE_PATH + "\\Morphology.png"); // Read the file
    if (img.empty()) // Check for invalid input
    {
        std::cout << "Could not open or find the frame" << std::endl;
        return;
    }
    cv::Mat dilationRes, erosionRes;
    cv::dilate(img, dilationRes, cv::Mat());
    cv::erode(img, erosionRes, cv::Mat());
    cv::namedWindow("Dilation 1IT", cv::WINDOW_AUTOSIZE); // Create a window for display.
    cv::imshow("Dilation 1IT", dilationRes);
    cv::namedWindow("Erosion 1IT", cv::WINDOW_AUTOSIZE); // Create a window for display.
    cv::imshow("Erosion 1IT", erosionRes);
    for (int i = 0; i < 10; i++) {
        cv::dilate(dilationRes, dilationRes, cv::Mat());
        cv::erode(erosionRes, erosionRes, cv::Mat());
    }
    cv::namedWindow("Dilation 10IT", cv::WINDOW_AUTOSIZE); // Create a window for display.
    cv::imshow("Dilation 10IT", dilationRes);
    cv::namedWindow("Erosion 10IT", cv::WINDOW_AUTOSIZE); // Create a window for display.
    cv::imshow("Erosion 10IT", erosionRes);
    cv::waitKey(0);

    // 1) Der Effekt wird übermäßig eingesetzt und alles verschwimmt zu einer weißen Masse
    //    oder nur zu schwarz
    // 2) Sobald informationen ganz verloren gehen, z.B. wenn der schwarze Kreis im 'o' ganz 
    //    verschwunden ist, kann er nicht wieder hergestellt werden
}

void aufgabe3() {
    cv::Mat img = cv::imread(BASE_PATH + "\\Gut_Reisen.png"); // Read the file
    if (img.empty()) // Check for invalid input
    {
        std::cout << "Could not open or find the frame" << std::endl;
        return;
    }
    cv::Mat img_gray, dst;
    cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
    cv::Mat edges;
    double threshold = 150;
    cv::Canny(img_gray, edges, threshold, threshold * 2.5);

    cv::namedWindow("Canny", cv::WINDOW_AUTOSIZE); // Create a window for display.
    cv::imshow("Canny", edges);

    vector<Vec2f> lines; // will hold the results of the detection
    HoughLines(edges, lines, 1, CV_PI / 180, 200, 0, 0); // runs the actual detection
   
    int dist_thresh = 5;
    Point bottom0, bottom1, top0, top1;
    bottom0.x = bottom1.x = bottom0.y = bottom1.y = 10000000;
    top0.x = top1.x = top0.y = top1.y = 0;

    for (size_t i = 0; i < lines.size(); i++)
    {
        float rho = lines[i][0], theta = lines[i][1];
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a * rho, y0 = b * rho;
        pt1.x = cvRound(x0 + 1000 * (-b));
        pt1.y = cvRound(y0 + 1000 * (a));
        pt2.x = cvRound(x0 - 1000 * (-b));
        pt2.y = cvRound(y0 - 1000 * (a));

        if (abs(pt1.x - pt2.x) < 100) {
            continue;
        }
        if ((pt1.y < 30) || (abs(pt1.y - img.cols) < 50)) {
            continue;
        }
        if ((pt2.y < 30) || (abs(pt2.y - img.cols) < 50)) {
            continue;
        }
        if (pt1.y <= bottom0.y) {
            bottom0 = pt1; 
            bottom1 = pt2;
        }
        if (pt1.y >= top0.y) {
            top0 = pt1; 
            top1 = pt2;
        }
    }
    
    cv::Rect region(Point(0, bottom0.y), Point(img.cols, top0.y));
    Mat subregion = img(region).clone();

    cvtColor(subregion, subregion, COLOR_BGR2GRAY);
    cv::threshold(subregion, subregion, 230, 255, THRESH_BINARY);
    cv::dilate(subregion, subregion, cv::Mat());
    cv::erode(subregion, subregion, cv::Mat());
    cv::erode(subregion, subregion, cv::Mat());
    cv::dilate(subregion, subregion, cv::Mat());
    cv::GaussianBlur(subregion, subregion, cv::Size(3, 3), 1);

    cv::namedWindow("Res", cv::WINDOW_AUTOSIZE); // Create a window for display.
    cv::imshow("Res", subregion);
    cv::waitKey(0);
}

int main()
{
    fillLUT();
    aufgabe1();
    aufgabe2();
    aufgabe3();
    return 0;
}



