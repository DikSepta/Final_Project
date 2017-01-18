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

#define MAX_FRAME 100
#define CLOCK_PER_SEC 1000
#define STAR_MAXSIZE 30
#define STAR_RESPONSE_TH 20
#define STAR_LINEBACKPROJ 10
#define STAR_LINEBACKBIN 8
#define STAR_NONMAXSUPP 3
#define MATCHING_TH 0.4

double focal = 718.856; //focal lenght
Point2d pp(607.1928,185.2157); //principle point

char filename1[256], filename2[256], filename3[256];

Mat imgout, imresize1, imresize2, imresize3, img1, img2, img3, R_f, t_f;
Mat K = (Mat_<float>(3,3) << 718.856, 0, 607.1928, 0, 718.856, 185.2157, 0, 0, 1);

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

        vector<KeyPoint> keypoint[3];

        begin = clock();
        star->detect(img1,keypoint[0]);
        star->detect(img2,keypoint[1]);
        time_star = (clock() - begin)/CLOCK_PER_SEC;

        Mat descriptor1, descriptor2, descriptor3;

        brisk->compute(img1, keypoint[0], descriptor1);
        brisk->compute(img2, keypoint[1], descriptor2);

        vector<DMatch> matches, good_matches;

        BFMatcher matcher(NORM_HAMMING);
        matcher.match(descriptor2, descriptor1, matches);

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

        //memindahkan ke vector point yang berurutan agar bisa digunakan findessentialmat()
        vector<Point2f> point1, point2, point3, matched_point1, matched_point2, matched_point3;

        KeyPoint::convert(keypoint[0], point1);
        KeyPoint::convert(keypoint[1], point2);

        for(unsigned int i = 0; i < good_matches.size() ; i++)
        {
            int k = good_matches[i].trainIdx;
            matched_point1.push_back(point1[k]);
            k = good_matches[i].queryIdx;
            matched_point2.push_back(point2[k]);
        }

        Mat E = findEssentialMat(matched_point1, matched_point2, focal, pp, RANSAC, 0.9999, 1.0);
        Mat R, t, mask;

        recoverPose(E, matched_point1, matched_point2, R, t, focal, pp, mask);

        Mat proj_mat_1, proj_mat_2;
        hconcat(Mat::eye(3,3,CV_64F), Mat::zeros(3,1,CV_64F), proj_mat_1);
        hconcat(R, t, proj_mat_2);

        proj_mat_1 = K * proj_mat_1;
        proj_mat_2 = K * proj_mat_2;

        R_f = R.t();
        t_f = -R.t()*t;

        sprintf(filename3, "/media/dikysepta/DATA/Final Project/Datasets/dataset/sequences/00/image_0/%06d.png", frame+2);

        imresize3 = imread(filename3, IMREAD_GRAYSCALE);
        resize(imresize3, img3, ukuran);

        star->detect(img3, keypoint[2]);
        brisk->compute(img3, keypoint[2], descriptor3);

        matches.clear();
        vector<DMatch> good_matches2;

        matcher.match(descriptor3, descriptor2, matches);

        max_dist = 0; min_dist = 10000;

        //-- Quick calculation of max and min distances between keypoints
        for( int i = 0; i < descriptor2.rows; i++ )
        {
            double dist = matches[i].distance;
            if( dist < min_dist ) min_dist = dist;
            if( dist > max_dist ) max_dist = dist;
        }

        for( int i = 0; i < descriptor2.rows; i++ )
        {
            if( matches[i].distance <= MATCHING_TH*(max_dist-min_dist) )
            {
                good_matches2.push_back( matches[i]);
            }
        }
        //mencari feature match dari feature frame k-1 dengan feature frame k
        KeyPoint::convert(keypoint[2], point3);
        unsigned int index = 0;
        vector<Point2f> pnp_frame_1, pnp_frame_2; //variabel untuk modal triangulasi
        vector<Point2f> pnp_2d_point;
        Mat pnp_4d_point;
        vector<Point3f> pnp_3d_point;

        for(unsigned int i = 0; i < good_matches2.size(); i++)
        {
            for(unsigned int j = 0; j < good_matches.size(); j++)
            {
                if(good_matches2.at(i).trainIdx == good_matches.at(j).queryIdx)
                {
                    pnp_frame_1.push_back(matched_point1[j]);
                    pnp_frame_2.push_back(matched_point2[j]);
                    pnp_2d_point.push_back(point3[good_matches2.at(i).trainIdx]);
                    index++;
                }
            }
        }
        for(int i = 0; i < index; i++)
        {
            pnp_3d_point.push_back(pnp_4d_point);
        }
        triangulatePoints(proj_mat_1, proj_mat_2, pnp_frame_1, pnp_frame_2, pnp_3d_point);
        Mat R_vec, t_vec;
        solvePnP(pnp_3d_point, pnp_2d_point, K, 0, R_vec, t_vec, false, SOLVEPNP_ITERATIVE);


//        cout << good_matches.size() << "  " << index << endl;
        cout << pnp_3d_point.cols << endl;
        Mat img_matches;

//        drawMatches(img1, keypoint[0], img2, keypoint[1], good_matches, img_matches);
//        imshow("Keypoint Matched", img1);

//        drawKeypoints(img1, keypoint[0], imgout, Scalar::all(-1),DrawMatchesFlags::DEFAULT);
//        imshow("Keypoint detected", img2);

        waitKey(250);
    }
    return 0;
}

