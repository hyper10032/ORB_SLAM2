#include <iostream>
#include <fstream>
#include <algorithm>
#include <chrono>

#include <opencv2/core/core.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "System.h"
#include "mytest/json.hpp"

using namespace std;
using json = nlohmann::json;

int main(int argc, char **argv)
{
    if (argc != 4)
    {
        cerr << endl
             << "Usage: ./run_orb path_to_vocabulary path_to_settings path_to_sequence" << endl
             << "example: ./mytest/run_orb ./Vocabulary/ORBvoc.txt ./mytest/camera.yaml \
             ../backup/chandra_pose_estimation_posetxt/train_pbr/000000"
             << endl;
        return 1;
    }

    string myPath = string(argv[3]);
    double depth_scale = 1000.0;

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM(argv[1], argv[2], ORB_SLAM2::System::RGBD, true);

    // Vector for tracking time statistics
    vector<double> vTimesTrack;

    // save result
    ofstream fout_gt("ground_truth.txt");
    fout_gt << fixed;
    fout_gt << setprecision(6) << 0.0 << " " << setprecision(9) << 0.0 << " " << 0.0 << " " << 0.0 << " "
            << 0.0 << " " << 0.0 << " " << 0.0 << " " << 1.0 << "\n";
    ofstream fout_orb("orb.txt");
    fout_orb << fixed;
    fout_orb << setprecision(6) << 0.0 << " " << setprecision(9) << 0.0 << " " << 0.0 << " " << 0.0 << " "
             << 0.0 << " " << 0.0 << " " << 0.0 << " " << 1.0 << "\n";
    ofstream fout_orbIcp("orb_icp.txt");
    fout_orbIcp << fixed;
    fout_orbIcp << setprecision(6) << 0.0 << " " << setprecision(9) << 0.0 << " " << 0.0 << " " << 0.0 << " "
                << 0.0 << " " << 0.0 << " " << 0.0 << " " << 1.0 << "\n";

    // 第一帧ground truth
    ifstream fin;
    json j;
    fin.open(myPath + "/" + "scene_camera.json");
    fin >> j;
    fin.close();
    auto R1_gt = j["0"]["cam_R_w2c"];
    auto t1_gt = j["0"]["cam_t_w2c"];
    Eigen::Matrix4d T1_gt = Eigen::Matrix4d::Zero();
    T1_gt << R1_gt[0], R1_gt[1], R1_gt[2], double(t1_gt[0]) / depth_scale,
        R1_gt[3], R1_gt[4], R1_gt[5], double(t1_gt[1]) / depth_scale,
        R1_gt[6], R1_gt[7], R1_gt[8], double(t1_gt[2]) / depth_scale,
        0, 0, 0, 1;

    // Main loop
    double timeStamp = 0.0;
    cv::Mat imRGB, imD;
    int count = 0;
    int step = 5;
    while (true)
    {
        // Read image and depthmap from file
        string img_number_2 = to_string((count % 1000));
        int zero_num = 6 - img_number_2.size();
        string depth_path_2 = myPath + "/" + "depth/" + string(zero_num, '0') + img_number_2 + ".png";
        string rgb_path_2 = myPath + "/" + "rgb/" + string(zero_num, '0') + img_number_2 + ".jpg";
        imRGB = cv::imread(rgb_path_2, CV_LOAD_IMAGE_UNCHANGED);
        imD = cv::imread(depth_path_2, CV_LOAD_IMAGE_UNCHANGED);
        // cout << "current image: " << rgb_path_2 << endl;

        if (imRGB.empty() || imD.empty())
        {
            cerr << endl
                 << "Failed to load image at: "
                 << rgb_path_2 << endl;
            return 1;
        }

        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

        // Pass the image to the SLAM system
        // T_camera^world
        std::pair<cv::Mat, cv::Mat> trackResult = SLAM.TrackRGBD(imRGB, imD, timeStamp);
        cv::Mat Torb = trackResult.first;
        cv::Mat TorbIcp = trackResult.second;

        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

        double ttrack = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
        // cout << "time used: " << ttrack << endl;

        vTimesTrack.emplace_back(ttrack);

        // Wait to load the next frame
        double T = 0.05;

        if (ttrack < T)
            usleep((T - ttrack) * 1e6);

        if (count != 0)
        {
            // save result
            Eigen::Matrix3f temp;
            temp << Torb.at<float>(0, 0), Torb.at<float>(0, 1), Torb.at<float>(0, 2),
                Torb.at<float>(1, 0), Torb.at<float>(1, 1), Torb.at<float>(1, 2),
                Torb.at<float>(2, 0), Torb.at<float>(2, 1), Torb.at<float>(2, 2);
            Eigen::Quaternionf qorb(temp);
            fout_orb << setprecision(6) << timeStamp << " "
                     << setprecision(9) << Torb.at<float>(0, 3) << " " << Torb.at<float>(1, 3) << " " << Torb.at<float>(2, 3) << " "
                     << qorb.x() << " " << qorb.y() << " " << qorb.z() << " " << qorb.w() << "\n";

            temp << TorbIcp.at<float>(0, 0), TorbIcp.at<float>(0, 1), TorbIcp.at<float>(0, 2),
                TorbIcp.at<float>(1, 0), TorbIcp.at<float>(1, 1), TorbIcp.at<float>(1, 2),
                TorbIcp.at<float>(2, 0), TorbIcp.at<float>(2, 1), TorbIcp.at<float>(2, 2);
            Eigen::Quaternionf qorbIcp(temp);
            fout_orbIcp << setprecision(6) << timeStamp << " "
                        << setprecision(9) << TorbIcp.at<float>(0, 3) << " " << TorbIcp.at<float>(1, 3) << " " << TorbIcp.at<float>(2, 3) << " "
                        << qorbIcp.x() << " " << qorbIcp.y() << " " << qorbIcp.z() << " " << qorbIcp.w() << "\n";

            // ground truth
            auto R2_gt = j[img_number_2]["cam_R_w2c"];
            auto t2_gt = j[img_number_2]["cam_t_w2c"];
            Eigen::Matrix4d T2_gt = Eigen::Matrix4d::Zero();
            T2_gt << R2_gt[0], R2_gt[1], R2_gt[2], double(t2_gt[0]) / depth_scale,
                R2_gt[3], R2_gt[4], R2_gt[5], double(t2_gt[1]) / depth_scale,
                R2_gt[6], R2_gt[7], R2_gt[8], double(t2_gt[2]) / depth_scale,
                0, 0, 0, 1;
            Eigen::Matrix4d T_gt = T1_gt * (T2_gt.inverse());
            Eigen::Matrix3d R = T_gt.block(0, 0, 3, 3);
            Eigen::Quaterniond q(R);
            fout_gt << setprecision(6) << timeStamp << " "
                    << setprecision(9) << T_gt(0, 3) << " " << T_gt(1, 3) << " " << T_gt(2, 3) << " "
                    << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
        }

        count += step;
        timeStamp += 0.005 * step;
        // if (count == 1000)
        // {
        //     break;
        // }

        if (count == 1000)
        {
            myPath = "/home/hyper/code/backup/chandra_pose_estimation_posetxt/train_pbr/000001";
            fin.open(myPath + "/" + "scene_camera.json");
            fin >> j;
            fin.close();
        }
        else if (count == 2000)
        {
            myPath = "/home/hyper/code/backup/chandra_pose_estimation_posetxt/train_pbr/000002";
            fin.open(myPath + "/" + "scene_camera.json");
            fin >> j;
            fin.close();
        }
        else if (count == 3000)
        {
            myPath = "/home/hyper/code/backup/chandra_pose_estimation_posetxt/train_pbr/000003";
            fin.open(myPath + "/" + "scene_camera.json");
            fin >> j;
            fin.close();
        }
        else if (count == 4000)
        {
            myPath = "/home/hyper/code/backup/chandra_pose_estimation_posetxt/train_pbr/000004";
            fin.open(myPath + "/" + "scene_camera.json");
            fin >> j;
            fin.close();
        }
        else if (count == 5000)
        {
            myPath = "/home/hyper/code/backup/chandra_pose_estimation_posetxt/train_pbr/000005";
            fin.open(myPath + "/" + "scene_camera.json");
            fin >> j;
            fin.close();
        }
        else if (count == 6000)
        {
            myPath = "/home/hyper/code/backup/chandra_pose_estimation_posetxt/train_pbr/000006";
            fin.open(myPath + "/" + "scene_camera.json");
            fin >> j;
            fin.close();
        }
        else if (count == 7000)
        {
            myPath = "/home/hyper/code/backup/chandra_pose_estimation_posetxt/train_pbr/000007";
            fin.open(myPath + "/" + "scene_camera.json");
            fin >> j;
            fin.close();
        }
        else if (count == 8000)
        {
            break;
        }
    }
    fout_gt.close();
    fout_orb.close();
    fout_orbIcp.close();

    // Stop all threads
    SLAM.Shutdown();

    // Tracking time statistics
    int nImages = vTimesTrack.size();
    sort(vTimesTrack.begin(), vTimesTrack.end());
    double totaltime = accumulate(vTimesTrack.begin(), vTimesTrack.end(), 0.0);
    cout << "-------" << endl
         << endl;
    cout << "median tracking time: " << vTimesTrack[nImages / 2] << endl;
    cout << "mean tracking time: " << totaltime / nImages << endl;

    // Save camera trajectory
    SLAM.SaveTrajectoryTUM("CameraTrajectory.txt");
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

    return 0;
}
