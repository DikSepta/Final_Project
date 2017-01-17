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
#define MATCHING_TH 0.3

char filename1[256], filename2[256];

Mat imgout, imresize1, imresize2, img1, img2;

Size ukuran(640,480);

clock_t time_star, begin;

Ptr<StarDetector> star = StarDetector::create(STAR_MAXSIZE,STAR_RESPONSE_TH,STAR_LINEBACKPROJ,STAR_LINEBACKBIN,STAR_NONMAXSUPP);
Ptr<BRISK> brisk = BRISK::create();

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

        vector<KeyPoint> keypoint[2];

        begin = clock();
        star->detect(img1,keypoint[0]);
        star->detect(img2,keypoint[1]);
        time_star = (clock() - begin)/CLOCK_PER_SEC;

        Mat descriptor1, descriptor2;

        brisk->compute(img1, keypoint[0], descriptor1);
        brisk->compute(img2, keypoint[1], descriptor2);

        vector<DMatch> matches, good_matches;

        BFMatcher matcher(NORM_HAMMING);
        matcher.match(descriptor1, descriptor2, matches);

        double max_dist = 0; double min_dist = 10000;

        //-- Quick calculation of max and min distances between keypoints
        for( int i = 0; i < descriptor1.rows; i++ )
        {
            double dist = matches[i].distance;
            if( dist < min_dist ) min_dist = dist;
            if( dist > max_dist ) max_dist = dist;
        }

        //-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
        //-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
        //-- small)
        //-- PS.- radiusMatch can also be used here.

        for( int i = 0; i < descriptor1.rows; i++ )
        {
            if( matches[i].distance <= MATCHING_TH*(max_dist-min_dist) )
            {
                good_matches.push_back( matches[i]);
            }
        }

        Mat img_matches;

        drawMatches(img1, keypoint[0], img2, keypoint[1], good_matches, img_matches);
        imshow("Keypoint Matched", img_matches);

        drawKeypoints(img1, keypoint[0], imgout, Scalar::all(-1),DrawMatchesFlags::DEFAULT);
        //imshow("Keypoint detected", imgout);

        cout << min_dist << endl << max_dist << endl << good_matches.size() << endl;

        waitKey(2);
    }
    return 0;
}

