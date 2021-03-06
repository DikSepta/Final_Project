#include <iostream>
#include <opencv2/opencv.hpp>
#include <ctype.h>
#include <algorithm> // for copy
#include <iterator> // for ostream_iterator
//#include <vector>
#include <ctime>
#include <time.h>
#include <sstream>
#include <fstream>
#include <string>
#include <opencv2/xfeatures2d.hpp>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

#define MAX_FRAME 1000
#define CLOCK_PER_SEC 1000
#define STAR_MAXSIZE 30
#define STAR_RESPONSE_TH 20
#define STAR_LINEBACKPROJ 10
#define STAR_LINEBACKBIN 8
#define STAR_NONMAXSUPP 3
#define MATCHING_TH 0.3

Mat captureImage(int frame);
vector<DMatch> matchFeature(Mat descriptor_1, Mat descriptor_2);
void matchTwoPoints(vector<DMatch> good_matches, vector<KeyPoint> keypoint_1,  vector<KeyPoint> keypoint_2, vector<Point2f> &point_1, vector<Point2f> &point_2);
void matchThreePoints(vector<DMatch> good_matches_1, vector<DMatch> good_matches_2,
                      vector<KeyPoint> keypoint_1, vector<KeyPoint> keypoint_2, vector<KeyPoint> keypoint_3,
                      vector<Point2f> &point_1, vector<Point2f> &point_2, vector<Point2f> &point_3);
Mat LinearLSTriangulation(Point3d u,Matx34d P,Point3d u1,Matx34d P1);
void TriangulatePoints(const vector<Point2f> pt_set1 ,const vector<Point2f> pt_set2 ,const Mat Kinv,const Matx34d P,const Matx34d P1,vector<Point3d>& pointcloud);

Ptr<StarDetector> star = StarDetector::create(STAR_MAXSIZE,STAR_RESPONSE_TH,STAR_LINEBACKPROJ,STAR_LINEBACKBIN,STAR_NONMAXSUPP);
Ptr<BRISK> brisk = BRISK::create();
Ptr<SIFT> sift = SIFT::create();
Ptr<SURF> surf = SURF::create(400,4,3,false,true);


double focal = 718.856; //focal lenght
Point2d pp(607.1928,185.2157); //principle point

Mat img1, img2, img3;
Mat R_f = Mat::eye(3,3,CV_64F);
Mat t_f = Mat::zeros(3,1,CV_64F);
Mat K = (Mat_<double>(3,3) << 718.856, 0, 607.1928, 0, 718.856, 185.2157, 0, 0, 1);

clock_t time_star, begin;

char text[100];
int fontFace = FONT_HERSHEY_PLAIN;
double fontScale = 1;
int thickness = 1;
cv::Point textOrg(10, 50);
Mat traj = Mat::zeros(600, 600, CV_8UC3);

Mat proj_mat_1, proj_mat_2;

vector<KeyPoint> keypoint_1, keypoint_2, keypoint_3;
Mat descriptor_1, descriptor_2, descriptor_3;
vector<Point2f> point1, point2, match_1, match_2, match_3;
vector<DMatch> good_matches_2;

vector<Point3d> pnp_3d_point;
Mat pnp_4d_point; //homogenous coordinat

Mat R_pnp;
Mat R_vec, t_vec;
Mat mask1;
double _dc[] = {0,0,0,0};

int main()
{

    /*Inisialisasi Mode*/
    img1 = captureImage(0);
    img2 = captureImage(1);

    /*extract feature and compute descriptors*/
//    FAST(img1, keypoint_1, 20, true);
//    FAST(img2, keypoint_2, 20, true);
    star->detect(img1,keypoint_1);
    star->detect(img2,keypoint_2);

    brisk->compute(img1, keypoint_1, descriptor_1);
    brisk->compute(img2, keypoint_2, descriptor_2);

    /*Matching feature from two images*/
    vector<DMatch> good_matches_1 = matchFeature(descriptor_1, descriptor_2);

    /*Membuat vector point sesuai feature yang cocok agar bisa digunakan findessentialmat()*/
    matchTwoPoints(good_matches_1, keypoint_1, keypoint_2, point1, point2);

    /*Hitung Essential matrix using ransac*/
    Mat E = findEssentialMat(point1, point2, focal, pp, RANSAC, 0.9999, 1.0);

    /*Hitung extrinsic matrix*/
    Mat R, t, mask;
    recoverPose(E, point1, point2, R, t, focal, pp, mask);

    /*Hitung projection matrix frame 0 dan 1*/
    hconcat(Mat::eye(3,3,CV_64F), Mat::zeros(3,1,CV_64F), proj_mat_1);
    hconcat(R, t, proj_mat_2);
    proj_mat_1 = K * proj_mat_1;
    proj_mat_2 = K * proj_mat_2;
    //double scale = 0.8586941/t.at<double>(2,0);
//    cout << proj_mat_1 << endl << proj_mat_2;

//    /*Hitung posisi kamera*/
//    R_f = R.t();
//    t_f = -R.t()*t;

//    /*iterasi*/
    for(unsigned int frame = 2; frame < MAX_FRAME; frame++)
    {
        Mat img4 = captureImage(frame-1);
        img3 = captureImage(frame);//frame);
        /*compute feature keypoint and descriptor*/
        //FAST(img3, keypoint_3, 20, true);
        star->detect(img3, keypoint_3);

        brisk->compute(img3, keypoint_3, descriptor_3);

        /*match feature with prev feature*/
        good_matches_2 = matchFeature(descriptor_2, descriptor_3);
        /*Find correspondence feature from frame k-2, k-1, and k*/
        matchThreePoints(good_matches_1, good_matches_2, keypoint_1, keypoint_2, keypoint_3, match_1, match_2, match_3);

        Mat imgout;
        //drawMatches(img4, keypoint_3, img3, keypoint_2, good_matches_2, imgout, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        drawKeypoints(img3, keypoint_3, imgout, Scalar(0,255,255),DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        imshow("matches", imgout);
        /*Solve PNP to find projection matrix and camera pose for frame k*/
        /*Find triangulated point using correspondenc point from frame k-2 and k-1*/
        triangulatePoints(proj_mat_1, proj_mat_2, match_1, match_2, pnp_4d_point);
        /*Convert 4d homogenous coordinat to 3d point untuk solve pnp*/
        for(int i = 0; i < pnp_4d_point.cols; i++)
            pnp_3d_point.push_back(Point3f(0,0,0));

        for(int i = 0; i < pnp_4d_point.cols; i++)
        {
            pnp_3d_point.at(i).x = pnp_4d_point.at<float>(0,i)/pnp_4d_point.at<float>(3,i);
            pnp_3d_point.at(i).y = pnp_4d_point.at<float>(1,i)/pnp_4d_point.at<float>(3,i);
            pnp_3d_point.at(i).z = pnp_4d_point.at<float>(2,i)/pnp_4d_point.at<float>(3,i);
        }

        /*find rotation matrix and translasion matrix*/
        solvePnPRansac(pnp_3d_point, match_3, K, Mat(1,4,CV_64FC1,_dc),R_vec, t_vec, false, 1000, 1, 0.99999, mask1, SOLVEPNP_EPNP);
        //solvePnP(pnp_3d_point, match_3, K, Mat(1,4,CV_64FC1,_dc), R_vec, t_vec, false, SOLVEPNP_ITERATIVE);
        Rodrigues(R_vec, R_pnp);

        R_f = R_pnp.t();
        t_f = -R_f*t_vec;
        /*tampilkan variabel untuk diamati*/
        cout << "Frame " << frame << endl;
        cout << countNonZero(mask1) << "  ";
        cout << "Key1:" << keypoint_1.size() << " Key2:" << keypoint_2.size() << " Key3:" << keypoint_3.size();
        cout << " Match1:" << good_matches_1.size()<< " Match2:" << good_matches_2.size();
        cout << " Match3:" << match_1.size();
        cout << " X:" << t_f.at<double>(0,0) << " Y:" << t_f.at<double>(1,0) << " Z:" << t_f.at<double>(2,0) << endl;
        /*update variable*/
        descriptor_2 = descriptor_3.clone();
        good_matches_1 = good_matches_2;

        keypoint_1 = keypoint_2;
        keypoint_2 = keypoint_3;

        proj_mat_1 = proj_mat_2.clone();
        hconcat(R_pnp, t_vec, proj_mat_2);
        Mat temp = K * proj_mat_2;
        proj_mat_2.release();
        proj_mat_2 = temp.clone();

        pnp_3d_point.erase(pnp_3d_point.begin(), pnp_3d_point.end());
        pnp_4d_point.release();
        R_pnp.release();
        t_vec.release();

        namedWindow( "Trajectory", WINDOW_AUTOSIZE );// Create a window for display.

        int x = int(t_f.at<double>(0)) + 300;
        int y = -int(t_f.at<double>(2)) + 500;
        circle(traj, Point(x, y) ,1, CV_RGB(255,0,0), 2);

        rectangle(traj, Point(10, 30), Point(550, 50), CV_RGB(0,0,0), CV_FILLED);
        sprintf(text, "Koordinat: x = %.3fm y = %.3fm z = %.3fm", t_f.at<double>(0), t_f.at<double>(1), t_f.at<double>(2));
        putText(traj, text, textOrg, fontFace, fontScale, Scalar::all(255), thickness, 8);

        imshow( "Trajectory", traj );

        waitKey(1);
    }
    return 0;
}

Mat captureImage(int frame)
{
    Size ukuran(640,480);
    Mat image, imresize;
    char filename[256];

    sprintf(filename, "/media/dikysepta/DATA/Final Project/Datasets/dataset/sequences/00/image_0/%06d.png", frame);

    imresize = imread(filename, IMREAD_GRAYSCALE);

    resize(imresize, image, ukuran);

    return imresize;
}

/*descriptor_1 is train image, descriptor_2 is query image*/
vector<DMatch> matchFeature(Mat descriptor_1, Mat descriptor_2)
{
    vector<DMatch> matches;//, good_matches;

    BFMatcher matcher(NORM_HAMMING);
    matcher.match(descriptor_2, descriptor_1, matches);

    double max_dist = 0; double min_dist = 10000;
    //-- Quick calculation of max and min distances between keypoints
    for(unsigned int i = 0; i < matches.size(); i++ )
    {
        double dist = matches.at(i).distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
    }
    cout << "max "<<max_dist<<" min "<< min_dist;
    int count = 0;
    int bound = matches.size();
    for( int i = 0; i < bound; i++ )
    {
        if( matches.at(count).distance > MATCHING_TH*(max_dist-min_dist) )
        {
            matches.erase(matches.begin() + count);
        }
        else
            count++;
    }
    cout << " match "<<matches.size()<<" bound "<< bound << endl;

    return matches;
}

/*match 2 array of keypoint, output 2 array of 2d point*/
void matchTwoPoints(vector<DMatch> good_matches, vector<KeyPoint> keypoint_1, vector<KeyPoint> keypoint_2, vector<Point2f> &point_1, vector<Point2f> &point_2)
{
    KeyPoint::convert(keypoint_1, point_1);
    KeyPoint::convert(keypoint_2, point_2);

    vector<Point2f> temp_point_1, temp_point_2;
    for(unsigned int i = 0; i < good_matches.size(); i++)
    {
       temp_point_1.push_back(point_1.at(good_matches.at(i).trainIdx));
       temp_point_2.push_back(point_2.at(good_matches.at(i).queryIdx));
    }
    for(unsigned int i = 0; i < good_matches.size(); i++)
    {
        point_1.at(i).x = temp_point_1.at(i).x;
        point_1.at(i).y = temp_point_1.at(i).y;
        point_2.at(i).x = temp_point_2.at(i).x;
        point_2.at(i).y = temp_point_2.at(i).y;
    }
    point_1.erase(point_1.begin()+good_matches.size(), point_1.end());
    point_2.erase(point_2.begin()+good_matches.size(), point_2.end());
}

/*match 3 array of keypoint, output 3 array of point2d*/
void matchThreePoints(vector<DMatch> good_matches_1, vector<DMatch> good_matches_2,
                      vector<KeyPoint> keypoint_1, vector<KeyPoint> keypoint_2, vector<KeyPoint> keypoint_3,
                      vector<Point2f> &point_1, vector<Point2f> &point_2, vector<Point2f> &point_3)
{
    KeyPoint::convert(keypoint_1, point_1);
    KeyPoint::convert(keypoint_2, point_2);
    KeyPoint::convert(keypoint_3, point_3);
    unsigned int indeks = 0;

    for(unsigned int i = 0; i < good_matches_2.size(); i++)
    {
        for(unsigned int j = 0; j < good_matches_1.size(); j++)
        {
            if(good_matches_2.at(i).trainIdx == good_matches_1.at(j).queryIdx)
            {
                indeks++;
            }
        }
    }
    int temp_point_1[2][indeks], temp_point_2[2][indeks], temp_point_3[2][indeks];
    indeks = 0;
    for(unsigned int i = 0; i < good_matches_2.size(); i++)
    {
        for(unsigned int j = 0; j < good_matches_1.size(); j++)
        {
            if(good_matches_2.at(i).trainIdx == good_matches_1.at(j).queryIdx)
            {
                temp_point_1[0][indeks] = point_1.at(good_matches_1.at(j).trainIdx).x;
                temp_point_1[1][indeks] = point_1.at(good_matches_1.at(j).trainIdx).y;
                temp_point_2[0][indeks] = point_2.at(good_matches_2.at(i).trainIdx).x;
                temp_point_2[1][indeks] = point_2.at(good_matches_2.at(i).trainIdx).y;
                temp_point_3[0][indeks] = point_3.at(good_matches_2.at(i).queryIdx).x;
                temp_point_3[1][indeks] = point_3.at(good_matches_2.at(i).queryIdx).y;
                indeks++;
            }
        }
    }

    for(unsigned int i = 0; i < indeks; i++)
    {
        point_1.at(i).x = temp_point_1[0][i];
        point_1.at(i).y = temp_point_1[1][i];
        point_2.at(i).x = temp_point_2[0][i];
        point_2.at(i).y = temp_point_2[1][i];
        point_3.at(i).x = temp_point_3[0][i];
        point_3.at(i).y = temp_point_3[1][i];
    }
    point_1.erase(point_1.begin()+indeks, point_1.end());
    point_2.erase(point_2.begin()+indeks, point_2.end());
    point_3.erase(point_3.begin()+indeks, point_3.end());
}

/**
 From "Triangulation", Hartley, R.I. and Sturm, P., Computer vision and image understanding, 1997
 */
Mat LinearLSTriangulation(Point3d u,       //homogenous image point (u,v,1)
                   Matx34d P,       //camera 1 matrix
                   Point3d u1,      //homogenous image point in 2nd camera
                   Matx34d P1       //camera 2 matrix
                                   )
{
//    build matrix A for homogenous equation system Ax = 0
//    assume X = (x,y,z,1), for Linear-LS method
//    which turns it into a AX = B system, where A is 4x3, X is 3x1 and B is 4x1
    Matx43d A(u.x*P(2,0)-P(0,0),    u.x*P(2,1)-P(0,1),      u.x*P(2,2)-P(0,2),
          u.y*P(2,0)-P(1,0),    u.y*P(2,1)-P(1,1),      u.y*P(2,2)-P(1,2),
          u1.x*P1(2,0)-P1(0,0), u1.x*P1(2,1)-P1(0,1),   u1.x*P1(2,2)-P1(0,2),
          u1.y*P1(2,0)-P1(1,0), u1.y*P1(2,1)-P1(1,1),   u1.y*P1(2,2)-P1(1,2)
              );
    Mat B = (Mat_<double>(4,1) <<    -(u.x*P(2,3)    -P(0,3)),
                      -(u.y*P(2,3)  -P(1,3)),
                      -(u1.x*P1(2,3)    -P1(0,3)),
                      -(u1.y*P1(2,3)    -P1(1,3)));

    Mat X;
    solve(A,B,X,DECOMP_SVD);

    return X;
}

//Triagulate points
void TriangulatePoints(const vector<Point2f> pt_set1,
                       const vector<Point2f> pt_set2,
                       const Mat Kinv,
                       const Matx34d P,
                       const Matx34d P1,
                       vector<Point3d>& pointcloud)
{
    pointcloud.clear();

    unsigned int pts_size = pt_set1.size();
    for (unsigned int i=0; i < pts_size; i++)
    {
        Point2f kp = pt_set1[i];
        Point3d u(kp.x,kp.y,1.0);
        Mat um = Kinv * Mat(u);
        u.x = um.at<double>(0,0);
        u.y = um.at<double>(1,0);
        u.z = um.at<double>(2,0);
        Point2f kp1 = pt_set2[i];
        Point3d u1(kp1.x,kp1.y,1.0);
        Mat um1 = Kinv * Mat(u1);
        u1.x = um1.at<double>(0,0);
        u1.y = um1.at<double>(1,0);
        u1.z = um1.at<double>(2,0);

        Mat X = LinearLSTriangulation(u,P,u1,P1);
        pointcloud.push_back(Point3d(X.at<double>(0,0),X.at<float>(0,1),X.at<float>(0,2)));
    }
}
