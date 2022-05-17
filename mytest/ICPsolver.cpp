#include "mytest/ICPsolver.h"
#include <pcl/io/pcd_io.h>

namespace ORB_SLAM2
{
    ICPsolver::ICPsolver(Tracking *pTracking, const string &strSettingPath) : cloudSource(new pcl::PointCloud<pcl::PointXYZ>),
                                                                              cloudTarget(new pcl::PointCloud<pcl::PointXYZ>),
                                                                              cloudIcp(new pcl::PointCloud<pcl::PointXYZ>),
                                                                              cloudSourceFiltered(new pcl::PointCloud<pcl::PointXYZ>),
                                                                              mpTracker(pTracking), mbFinishRequested(false), mbFinished(false),
                                                                              depth_scale(1000.0), foutIcp("icp.txt")
    {
        Ticp = cv::Mat::eye(4, 4, CV_32F);
        Trect = cv::Mat::eye(4, 4, CV_32F);
        K = (cv::Mat_<double>(3, 3) << 300.0, 0, 512.0, 0, 300.0, 512.0, 0, 0, 1);
        foutIcp << fixed;
        foutIcp << setprecision(6) << 0.0 << " " << setprecision(9) << 0.0 << " " << 0.0 << " " << 0.0 << " "
                << 0.0 << " " << 0.0 << " " << 0.0 << " " << 1.0 << "\n";

        pcl::io::loadPCDFile<pcl::PointXYZ>("/home/hyper/code/backup/chandra_pc.pcd", *cloudTarget);
        // pcl::io::loadPCDFile<pcl::PointXYZ>("/home/hyper/code/backup/clementine_pc.pcd", *cloudTarget);
        Eigen::Matrix4f T1_gt = Eigen::Matrix4f::Zero();
        T1_gt << -1.0, 0.0, 0.0, -10000.0 / 1000.0,
            0.0, -1.0, 0.0, -10000.0 / 1000.0,
            0.0, 0.0, 1.0, 15000.0 / 1000.0,
            0.0, 0.0, 0.0, 1.0;
        // T1_gt << -1.0, 0.0, 0.0, 5000.0 / 1000.0,
        //     0.0, -1.0, 0.0, 5000.0 / 1000.0,
        //     0.0, 0.0, 1.0, 10000.0 / 1000.0,
        //     0.0, 0.0, 0.0, 1.0;

        // 把模型点云变换到第一帧点云的坐标系，用模型点云替换第一帧点云作为cloudTarget
        // T1_gt变换矩阵是从scene_gt.json文件里拿到的，这里为了方便没有采用读取文件的方式
        pcl::transformPointCloud(*cloudTarget, *cloudTarget, T1_gt);
        cout << "get point cloud model with size: " << cloudTarget->size() << endl;
    }

    void ICPsolver::run()
    {
        initCloudTarget();
        usleep(500000); // 不能用sleep(0.5);
        while (true)
        {
            chrono::steady_clock::time_point t0 = chrono::steady_clock::now();

            cv::Mat currentDepth;
            cv::Mat Torb;
            double timeStamp;
            std::tie(timeStamp, currentDepth, Torb) = mpTracker->GetCurrentDepth();
            // orbTransformation = orbTransformation.inverse();
            for (int v = 0; v < currentDepth.rows; v++)
            {
                for (int u = 0; u < currentDepth.cols; u++)
                {
                    ushort d2 = currentDepth.ptr<unsigned short>(v)[u];
                    if (d2 != 0 && d2 != 65535)
                    {
                        double z2 = double(d2) / depth_scale;
                        double x2 = (u - K.at<double>(0, 2)) / K.at<double>(0, 0) * z2;
                        double y2 = (v - K.at<double>(1, 2)) / K.at<double>(1, 1) * z2;
                        cloudSource->push_back(pcl::PointXYZ(x2, y2, z2));
                    }
                }
            }
            // cout << "get depth with cloud source size: " << cloudSource->size() << endl;

            // 下采样滤波
            pcl::VoxelGrid<pcl::PointXYZ> voxelGrid;
            voxelGrid.setLeafSize(0.1f, 0.1f, 0.1f);
            voxelGrid.setInputCloud(cloudSource);
            voxelGrid.filter(*cloudSourceFiltered);
            // cout << "down size cloud_source from " << cloudSource->size()
            //      << " to " << cloudSourceFiltered->size() << endl;

            // icp配准算法
            pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
            // kdTree加速搜索
            pcl::search::KdTree<pcl::PointXYZ>::Ptr tree1(new pcl::search::KdTree<pcl::PointXYZ>);
            tree1->setInputCloud(cloudSourceFiltered);
            pcl::search::KdTree<pcl::PointXYZ>::Ptr tree2(new pcl::search::KdTree<pcl::PointXYZ>);
            tree2->setInputCloud(cloudTarget);
            icp.setSearchMethodSource(tree1);
            icp.setSearchMethodTarget(tree2);
            icp.setInputSource(cloudSourceFiltered);
            icp.setInputTarget(cloudTarget);
            // icp.setMaxCorrespondenceDistance(0.5); //当两个点云相距较远时候，距离值要变大，所以一开始需要粗配准。
            // icp.setTransformationEpsilon(1e-10);   // svd奇异值分解，对icp时间影响不大
            // icp.setEuclideanFitnessEpsilon(0.01);  //前后两次误差大小，当误差值小于这个值停止迭代
            // icp.setMaximumIterations(100);         //最大迭代次数
            Eigen::Matrix4f temp;
            temp << Ticp.at<float>(0, 0), Ticp.at<float>(0, 1), Ticp.at<float>(0, 2), Ticp.at<float>(0, 3),
                Ticp.at<float>(1, 0), Ticp.at<float>(1, 1), Ticp.at<float>(1, 2), Ticp.at<float>(1, 3),
                Ticp.at<float>(2, 0), Ticp.at<float>(2, 1), Ticp.at<float>(2, 2), Ticp.at<float>(2, 3),
                Ticp.at<float>(3, 0), Ticp.at<float>(3, 1), Ticp.at<float>(3, 2), Ticp.at<float>(3, 3);
            icp.align(*cloudIcp, temp);
            // cout << "icp has converged: " << icp.hasConverged() << "  score: " << icp.getFitnessScore() << endl;
            Eigen::Matrix4f temp2 = icp.getFinalTransformation();
            // cout << "icp result:\n"
            //      << temp2 << endl;
            Ticp = (cv::Mat_<float>(4, 4) << temp2(0, 0), temp2(0, 1), temp2(0, 2), temp2(0, 3),
                    temp2(1, 0), temp2(1, 1), temp2(1, 2), temp2(1, 3),
                    temp2(2, 0), temp2(2, 1), temp2(2, 2), temp2(2, 3),
                    temp2(3, 0), temp2(3, 1), temp2(3, 2), temp2(3, 3));

            Eigen::Matrix3f RIcp = temp2.block(0, 0, 3, 3);
            Eigen::Quaternionf qIcp(RIcp);
            foutIcp << setprecision(6) << timeStamp << " "
                    << setprecision(9) << temp2(0, 3) << " " << temp2(1, 3) << " " << temp2(2, 3) << " "
                    << qIcp.x() << " " << qIcp.y() << " " << qIcp.z() << " " << qIcp.w() << "\n";

            cloudSource->clear();
            cloudIcp->clear();
            cloudSourceFiltered->clear();

            if (Torb.empty())
            {
                cout << "get empty Torb..." << endl;
            }
            else
            {
                unique_lock<mutex> lock(mMutexRect);
                Trect = Ticp * Torb;
            }

            chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
            double time_used = chrono::duration_cast<chrono::duration<double>>(t1 - t0).count();
            mvTimesTrack.emplace_back(time_used);
            // cout << "icp time: " << time_used << endl;
            double T = 0.5;
            if (time_used < T)
            {
                usleep((T - time_used) * 1e6);
            }

            if (CheckFinish())
                break;
        }

        foutIcp.close();

        // Tracking time statistics
        int nImages = mvTimesTrack.size();
        sort(mvTimesTrack.begin(), mvTimesTrack.end());
        double totaltime = accumulate(mvTimesTrack.begin(), mvTimesTrack.end(), 0.0);
        cout << "-------" << endl
             << endl;
        cout << "icp median tracking time: " << mvTimesTrack[nImages / 2] << endl;
        cout << "icp mean tracking time: " << totaltime / nImages << endl;

        SetFinish();
    }

    void ICPsolver::initCloudTarget()
    {
        cv::Mat firstDepth;
        while (firstDepth.empty())
        {
            usleep(100000);
            cout << "trying to get first depth" << endl;
            firstDepth = mpTracker->GetFirstDepth();
        }
        // for (int v = 0; v < firstDepth.rows; v++)
        // {
        //     for (int u = 0; u < firstDepth.cols; u++)
        //     {
        //         ushort d1 = firstDepth.ptr<unsigned short>(v)[u];
        //         if (d1 != 0 && d1 != 65535)
        //         {
        //             double z1 = double(d1) / depth_scale;
        //             double x1 = (u - K.at<double>(0, 2)) / K.at<double>(0, 0) * z1;
        //             double y1 = (v - K.at<double>(1, 2)) / K.at<double>(1, 1) * z1;
        //             cloudTarget->push_back(pcl::PointXYZ(x1, y1, z1));
        //         }
        //     }
        // }
        // cout << "get fisrt depth with size: " << cloudTarget->size() << endl;
    }

    cv::Mat ICPsolver::getRectTransformation()
    {
        unique_lock<mutex> lock(mMutexRect);
        return Trect;
    }

    void ICPsolver::RequestFinish()
    {
        unique_lock<mutex> lock(mMutexFinish);
        mbFinishRequested = true;
    }

    bool ICPsolver::isFinished()
    {
        unique_lock<mutex> lock(mMutexFinish);
        return mbFinished;
    }

    bool ICPsolver::CheckFinish()
    {
        unique_lock<mutex> lock(mMutexFinish);
        return mbFinishRequested;
    }

    void ICPsolver::SetFinish()
    {
        unique_lock<mutex> lock(mMutexFinish);
        mbFinished = true;
    }
} // namespace ORB_SLAM2
