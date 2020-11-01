
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;

// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        {
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}

void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for (auto it1 = boundingBoxes.begin(); it1 != boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0, 150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top = 1e8, left = 1e8, bottom = 0.0, right = 0.0;
        float xwmin = 1e8, ywmin = 1e8, ywmax = -1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin < xw ? xwmin : xw;
            ywmin = ywmin < yw ? ywmin : yw;
            ywmax = ywmax > yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top < y ? top : y;
            left = left < x ? left : x;
            bottom = bottom > y ? bottom : y;
            right = right > x ? right : x;
            
            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 0, 0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d, #cls=%d", it1->boxID, (int)it1->lidarPoints.size(), it1->classID);
        putText(topviewImg, str1, cv::Point2f(left - 250, bottom + 50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax - ywmin);
        putText(topviewImg, str2, cv::Point2f(left - 250, bottom + 125), cv::FONT_ITALIC, 2, currColor);
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if (bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}

// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    double sumDist = 0.0;
    for (auto &match : kptMatches)
    {
        if (boundingBox.roi.contains(kptsCurr[match.trainIdx].pt))
        {
            boundingBox.kptMatches.emplace_back(match);
            sumDist += match.distance;
        }
    }

    double meanDist = sumDist / boundingBox.kptMatches.size();

    boundingBox.kptMatches.erase(std::remove_if(boundingBox.kptMatches.begin(), boundingBox.kptMatches.end(), [&meanDist](auto& match){
                                                return match.distance < 0.7*meanDist;}), boundingBox.kptMatches.end());

}

// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr,
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    std::vector<double>distRatios;
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end()-1; ++it1)
    {
        cv::KeyPoint keyCurrOuter = kptsCurr[it1->trainIdx];
        cv::KeyPoint keyPrevOuter = kptsPrev[it1->queryIdx];

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
        {
            double minDist = 100.0;
            cv::KeyPoint keyCurrInner = kptsCurr[it2->trainIdx];
            cv::KeyPoint keyPrevInner = kptsPrev[it2->queryIdx];

            double distCurr = cv::norm(keyCurrOuter.pt - keyCurrInner.pt);
            double distPrev = cv::norm(keyPrevOuter.pt - keyPrevInner.pt);
            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            {
                double distRatio = distCurr/distPrev;
                distRatios.push_back(distRatio);
            }

        }
    }

    if(distRatios.size() == 0)
    {
        TTC = NAN;
        return;
    }
    std::sort(distRatios.begin(), distRatios.end());
    int medIdx = floor(distRatios.size()/2);
    double medDistRatio = distRatios.size()%2 == 0 ? (distRatios[medIdx-1] + distRatios[medIdx])/2.0 : distRatios[medIdx];
    double dT = 1.0/frameRate;
    TTC = -dT/(1-medDistRatio);


}

void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    std::vector<double> currDistances, prevDistances;
    std::sort(lidarPointsCurr.begin(), lidarPointsCurr.end(), [](auto lidPt1, auto lidPt2) { return lidPt1.x < lidPt2.x; });

    std::sort(lidarPointsPrev.begin(), lidarPointsPrev.end(), [](auto lidPt1, auto lidPt2) { return lidPt1.x < lidPt2.x; });

    int medCurrIdx = floor(lidarPointsCurr.size() / 2);
    int medPrevidx = floor(lidarPointsPrev.size() / 2);

    double medCurrX = lidarPointsCurr.size() % 2 == 0 ? (lidarPointsCurr[medCurrIdx - 1].x + lidarPointsCurr[medCurrIdx].x) / 2.0 : lidarPointsCurr[medCurrIdx].x;
    double medPrevX = lidarPointsPrev.size() % 2 == 0 ? (lidarPointsPrev[medPrevidx - 1].x + lidarPointsPrev[medPrevidx].x) / 2.0 : lidarPointsPrev[medPrevidx].x;

    double dT = 1.0 / frameRate;
    TTC = dT * medCurrX / (medPrevX - medCurrX);
    // TTC = dT * lidarPointsCurr[0].x / (lidarPointsPrev[0].x - lidarPointsCurr[0].x);
}

void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    std::multimap<int, int> bboxIdMap;
    int currId = -1;
    int prevId = -1;
    int prevMaxId = 0;
    for (auto it = matches.begin(); it != matches.end(); ++it)
    {
        cv::KeyPoint currKeypoint = currFrame.keypoints[it->trainIdx];
        cv::KeyPoint prevKeypoint = prevFrame.keypoints[it->queryIdx];

        for (auto bbox : currFrame.boundingBoxes)
        {
            if (bbox.roi.contains(currKeypoint.pt))
            {
                currId = bbox.boxID;
                break;
            }
        }
        for (auto bbox : prevFrame.boundingBoxes)
        {
            if (bbox.roi.contains(prevKeypoint.pt))
            {
                prevId = bbox.boxID;
                break;
            }
        }
        bboxIdMap.insert(std::make_pair(currId, prevId));

        prevMaxId = std::max(prevMaxId, prevId);
    }

    std::vector<int> currFrameBboxIds;
    for (auto &bbox : currFrame.boundingBoxes)
    {
        currFrameBboxIds.push_back(bbox.boxID);
    }

    for (int cid : currFrameBboxIds)
    {
        auto prevMatchedIds = bboxIdMap.equal_range(cid);

        
        std::vector<int> countMaxId(prevMaxId + 1, 0);

        
        for (auto it = prevMatchedIds.first; it != prevMatchedIds.second; it++)
        {
            if (it->second != -1)
            {
                
                countMaxId[it->second] += 1;
            }
        } 

        int pid = std::distance(countMaxId.begin(), std::max_element(countMaxId.begin(), countMaxId.end()));
        bbBestMatches.insert(std::make_pair(pid, cid));
    }


}
