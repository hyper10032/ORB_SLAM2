#ifndef ICPSOLVER_H
#define ICPSOLVER_H

#include "System.h"
#include <mutex>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/ndt.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter.h>

namespace ORB_SLAM2
{
    class ICPsolver
    {
    public:
        ICPsolver(Tracking *pTracking, const string &strSettingPath);
        void run();
        Eigen::Matrix4f getRectTransformation();

        void RequestFinish();
        bool isFinished();

    private:
        void initCloudTarget();

        bool CheckFinish();
        void SetFinish();

        ofstream foutIcp;

        cv::Mat K;
        const double depth_scale;
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloudSource;
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloudTarget;
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloudIcp;
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloudSourceFiltered;

        Eigen::Matrix4f icpTransformation;

        Eigen::Matrix4f rectTransformation;
        std::mutex mMutexRect;

        vector<double> mvTimesTrack;

        Tracking *mpTracker;

        bool mbFinishRequested;
        bool mbFinished;
        std::mutex mMutexFinish;
    };

} // namespace ORB_SLAM2
#endif // ICPSOLVER_H