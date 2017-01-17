#include <iostream>
#include <opencv2/opencv.hpp>
#include <ctype.h>
#include <algorithm> // for copy
#include <iterator> // for ostream_iterator
#include <vector>
#include <ctime>
#include <time.h>
#include <sstream>
#include <fstream>
#include <string>
#include <opencv2/xfeatures2d.hpp>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

#define MAX_FRAME 25
#define CLOCK_PER_SEC 1000
#define STAR_MAXSIZE 30
#define STAR_RESPONSE_TH 20
#define STAR_LINEBACKPROJ 10
#define STAR_LINEBACKBIN 8
#define STAR_NONMAXSUPP 3

char filename1[256], filename2[256];

Mat imgout, imresize1, imresize2, img1, img2;

Size ukuran(640,480);

clock_t time_star, begin;

vector<KeyPoint> keypoint[2];

Ptr<StarDetector> star = StarDetector::create(STAR_MAXSIZE,STAR_RESPONSE_TH,STAR_LINEBACKPROJ,STAR_LINEBACKBIN,STAR_NONMAXSUPP);

int main()
{   
    for(int frame = 0; frame < MAX_FRAME; frame++)
    {
        sprintf(filename1, "/media/dikysepta/DATA/Final Project/Datasets/dataset/sequences/00/image_0/%06d.png", frame);
        sprintf(filename2, "/media/dikysepta/DATA/Final Project/Datasets/dataset/sequences/00/image_0/%06d.png", frame+1);

        imresize1 = imread(filename1,IMREAD_GRAYSCALE);
        imresize2 = imread(filename2,IMREAD_GRAYSCALE);

        resize(imresize1, img1, ukuran);
        resize(imresize2, img2, ukuran);

        begin = clock();
        star->detect(img1,keypoint[0]);
        star->detect(img2,keypoint[1]);
        time_star = (clock() - begin)/CLOCK_PER_SEC;

        drawKeypoints(img1, keypoint[0], imgout, Scalar::all(-1),DrawMatchesFlags::DEFAULT);
        imshow("Keypoint detected", imgout);
        waitKey(100);
    }

    return 0;
}

