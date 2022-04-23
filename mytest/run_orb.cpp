#include <iostream>
#include <fstream>
#include <algorithm>
#include <chrono>

#include <opencv2/core/core.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>

#include "System.h"
#include "mytest/json.hpp"

using namespace std;
using json = nlohmann::json;

void LoadImages(vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps);

int main(int argc, char **argv)
{
    if (argc != 4)
    {
        cerr << endl
             << "Usage: ./run_orb path_to_vocabulary path_to_settings path_to_sequence" << endl;
        return 1;
    }

    // Retrieve paths to images
    vector<string> vstrImageFilenamesRGB;
    vector<string> vstrImageFilenamesD;
    vector<double> vTimestamps;
    LoadImages(vstrImageFilenamesRGB, vstrImageFilenamesD, vTimestamps);

    // Check consistency in the number of images and depthmaps
    int nImages = vstrImageFilenamesRGB.size();
    if (vstrImageFilenamesRGB.empty())
    {
        cerr << endl
             << "No images found in provided path." << endl;
        return 1;
    }
    else if (vstrImageFilenamesD.size() != vstrImageFilenamesRGB.size())
    {
        cerr << endl
             << "Different number of images for rgb and depth." << endl;
        return 1;
    }

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM(argv[1], argv[2], ORB_SLAM2::System::RGBD, true);

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl
         << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl
         << endl;

    // Main loop
    cv::Mat imRGB, imD;
    cv::Rect roi_rect = cv::Rect(0, 180, 640, 480);
    for (int ni = 0; ni < nImages; ni++)
    {
        // Read image and depthmap from file
        imRGB = cv::imread(string(argv[3]) + "/" + vstrImageFilenamesRGB[ni], CV_LOAD_IMAGE_UNCHANGED);
        imRGB = imRGB(roi_rect);
        imD = cv::imread(string(argv[3]) + "/" + vstrImageFilenamesD[ni], CV_LOAD_IMAGE_UNCHANGED);
        imD = imD(roi_rect);
        double tframe = vTimestamps[ni];

        if (imRGB.empty())
        {
            cerr << endl
                 << "Failed to load image at: "
                 << string(argv[3]) << "/" << vstrImageFilenamesRGB[ni] << endl;
            return 1;
        }

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

        // Pass the image to the SLAM system
        SLAM.TrackRGBD(imRGB, imD, tframe);

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

        double ttrack = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();

        vTimesTrack[ni] = ttrack;

        // Wait to load the next frame
        double T = 0;
        if (ni < nImages - 1)
            T = vTimestamps[ni + 1] - tframe;
        else if (ni > 0)
            T = tframe - vTimestamps[ni - 1];

        if (ttrack < T)
            usleep((T - ttrack) * 1e6);
    }

    // Stop all threads
    SLAM.Shutdown();

    // Tracking time statistics
    sort(vTimesTrack.begin(), vTimesTrack.end());
    float totaltime = accumulate(vTimesTrack.begin(), vTimesTrack.end(), 0);
    cout << "-------" << endl
         << endl;
    cout << "median tracking time: " << vTimesTrack[nImages / 2] << endl;
    cout << "mean tracking time: " << totaltime / nImages << endl;

    // Save camera trajectory
    SLAM.SaveTrajectoryTUM("CameraTrajectory.txt");
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

    // ground truth
    ifstream fin(string(argv[3]) + "/" + "scene_camera.json");
    json j;
    fin >> j;
    ofstream fout("ground_truth.txt");
    fout << fixed;
    fout << setprecision(6) << 0.0 << " " << setprecision(9) << 0.0 << " " << 0.0 << " " << 0.0 << "\n";
    for (int i = 10; i < 1000; i += 10)
    {
        double t = i * 0.005;
        double depth_scale = 1000.0;
        string img_number_2 = to_string(i);
        string img_number_1 = "0";
        auto R1_gt = j[img_number_1]["cam_R_w2c"];
        auto t1_gt = j[img_number_1]["cam_t_w2c"];
        Eigen::Matrix4d T1_gt = Eigen::Matrix4d::Zero();
        T1_gt << R1_gt[0], R1_gt[1], R1_gt[2], double(t1_gt[0]) / depth_scale,
            R1_gt[3], R1_gt[4], R1_gt[5], double(t1_gt[1]) / depth_scale,
            R1_gt[6], R1_gt[7], R1_gt[8], double(t1_gt[2]) / depth_scale,
            0, 0, 0, 1;
        auto R2_gt = j[img_number_2]["cam_R_w2c"];
        auto t2_gt = j[img_number_2]["cam_t_w2c"];
        Eigen::Matrix4d T2_gt = Eigen::Matrix4d::Zero();
        T2_gt << R2_gt[0], R2_gt[1], R2_gt[2], double(t2_gt[0]) / depth_scale,
            R2_gt[3], R2_gt[4], R2_gt[5], double(t2_gt[1]) / depth_scale,
            R2_gt[6], R2_gt[7], R2_gt[8], double(t2_gt[2]) / depth_scale,
            0, 0, 0, 1;
        Eigen::Matrix4d T_gt = T1_gt * (T2_gt.inverse());
        fout << setprecision(6) << t << " "
             << setprecision(9) << T_gt(0, 3) << " " << T_gt(1, 3) << " " << T_gt(2, 3) << "\n";
    }
    fout.close();

    return 0;
}

void LoadImages(vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps)
{
    for (int i = 0; i < 1000; i += 10)
    {
        double t = i * 0.005;
        vTimestamps.emplace_back(t);
        string sindex = to_string(i);
        int zeroNum = 6 - sindex.size();
        string sRGB = "rgb/" + string(zeroNum, '0') + sindex + ".jpg";
        vstrImageFilenamesRGB.emplace_back(sRGB);
        string sD = "depth/" + string(zeroNum, '0') + sindex + ".png";
        vstrImageFilenamesD.emplace_back(sD);
    }
}
