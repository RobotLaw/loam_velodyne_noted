// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// This is an implementation of the algorithm described in the following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.



/******************************读前须知*****************************************/
/*imu为x轴向前,y轴向左,z轴向上的右手坐标系，
  velodyne lidar被安装为x轴向前,y轴向左,z轴向上的右手坐标系，
  scanRegistration会把两者通过交换坐标轴，都统一到z轴向前,x轴向左,y轴向上的右手坐标系
  ，这是J. Zhang的论文里面使用的坐标系
  交换后：R = Ry(yaw)*Rx(pitch)*Rz(roll)
*******************************************************************************/

#include <cmath>
#include <vector>

#include <loam_velodyne/common.h>
#include <opencv/cv.h>
#include <nav_msgs/Odometry.h>
#include <opencv/cv.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

using std::sin;
using std::cos;
using std::atan2;

//扫描周期, velodyne频率10Hz，周期0.1s
const double scanPeriod = 0.1;

//初始化控制变量
const int systemDelay = 20;//弃用前20帧初始数据
int systemInitCount = 0;
bool systemInited = false;

//激光雷达线数
const int N_SCANS = 16;

//点云曲率, 40000为一帧点云(sweep)中点的最大数量
float cloudCurvature[40000];
//曲率点对应点云中的序号，最终是按照大小排序过的曲率点的id，里面又按照不同scan不同区域进行了大小排序的
int cloudSortInd[40000];
//点是否筛选过标志：0-未筛选过，1-筛选过
int cloudNeighborPicked[40000];
//点分类标号:2-代表曲率很大，1-代表曲率比较大,-1-代表曲率很小，0-曲率比较小(其中1包含了2,0包含了1,0和1构成了点云全部的点)
int cloudLabel[40000];

//imu时间戳大于当前点云时间戳的位置
int imuPointerFront = 0;
//imu最新收到的点在数组中的位置
int imuPointerLast = -1;
//imu循环队列长度
const int imuQueLength = 200;

// 与点云数据第一个点对应的imu的位移/速度/欧拉角(在世界坐标系下)
float imuRollStart = 0, imuPitchStart = 0, imuYawStart = 0;
// 与点云数据中的点对应的imu的位移/速度/欧拉角(在世界坐标系下)
float imuRollCur = 0, imuPitchCur = 0, imuYawCur = 0;

float imuVeloXStart = 0, imuVeloYStart = 0, imuVeloZStart = 0;
float imuShiftXStart = 0, imuShiftYStart = 0, imuShiftZStart = 0;

//当前点的速度，位移信息
float imuVeloXCur = 0, imuVeloYCur = 0, imuVeloZCur = 0;
float imuShiftXCur = 0, imuShiftYCur = 0, imuShiftZCur = 0;

//每次点云数据当前点相对于开始第一个点的畸变位移，速度(点云第一个点对应的imu坐标系下)
float imuShiftFromStartXCur = 0, imuShiftFromStartYCur = 0, imuShiftFromStartZCur = 0;
float imuVeloFromStartXCur = 0, imuVeloFromStartYCur = 0, imuVeloFromStartZCur = 0;

//IMU信息
double imuTime[imuQueLength] = {0};
float imuRoll[imuQueLength] = {0};
float imuPitch[imuQueLength] = {0};
float imuYaw[imuQueLength] = {0};

float imuAccX[imuQueLength] = {0};
float imuAccY[imuQueLength] = {0};
float imuAccZ[imuQueLength] = {0};
//imu在世界坐标系下的速度，只在接收到imu回调时更新
float imuVeloX[imuQueLength] = {0};
float imuVeloY[imuQueLength] = {0};
float imuVeloZ[imuQueLength] = {0};
//imu在世界坐标系下的坐标，只在接收到imu回调时更新
float imuShiftX[imuQueLength] = {0};
float imuShiftY[imuQueLength] = {0};
float imuShiftZ[imuQueLength] = {0};

ros::Publisher pubLaserCloud;
ros::Publisher pubCornerPointsSharp;
ros::Publisher pubCornerPointsLessSharp;
ros::Publisher pubSurfPointsFlat;
ros::Publisher pubSurfPointsLessFlat;
ros::Publisher pubImuTrans;

//计算局部坐标系下点云中的点相对第一个开始点的由于加减速运动产生的位移畸变
void ShiftToStartIMU(float pointTime)
{
  //计算相对于第一个点由于加减速产生的畸变位移(全局坐标系下畸变位移量delta_Tg)
  /*  为什么这样计算???这里假设点云在整个出点(即一个sweep)的过程中，机器人都是匀速运动的，即以第一个点的速度做匀速运动，
      所以后面每个点对应的imu世界坐标理想情况下应该为： sn = s0 + v0 * pointTime;
      所以实际对应的imu世界坐标减去理想的坐标就得到了运动的畸变，即 sn' - sn
  */
  //imuShiftFromStartCur = imuShiftCur - (imuShiftStart + imuVeloStart * pointTime)
  imuShiftFromStartXCur = imuShiftXCur - imuShiftXStart - imuVeloXStart * pointTime;
  imuShiftFromStartYCur = imuShiftYCur - imuShiftYStart - imuVeloYStart * pointTime;
  imuShiftFromStartZCur = imuShiftZCur - imuShiftZStart - imuVeloZStart * pointTime;

  /********************************************************************************
  R_b_n = R_ZYX = Rx(roll)*Ry(pitch)*Rz(yaw) 这个是未交换轴前的顺序
  Rz(pitch).inverse * Rx(pitch).inverse * Ry(yaw).inverse * delta_Tg
  transfrom from the global frame to the local frame，接下来要将这个畸变从世界坐标转换到imu坐标
  *********************************************************************************/

  //绕y轴旋转(-imuYawStart)，即Ry(yaw).inverse
  float x1 = cos(imuYawStart) * imuShiftFromStartXCur - sin(imuYawStart) * imuShiftFromStartZCur;
  float y1 = imuShiftFromStartYCur;
  float z1 = sin(imuYawStart) * imuShiftFromStartXCur + cos(imuYawStart) * imuShiftFromStartZCur;

  //绕x轴旋转(-imuPitchStart)，即Rx(pitch).inverse
  float x2 = x1;
  float y2 = cos(imuPitchStart) * y1 + sin(imuPitchStart) * z1;
  float z2 = -sin(imuPitchStart) * y1 + cos(imuPitchStart) * z1;

  //绕z轴旋转(-imuRollStart)，即Rz(pitch).inverse
  imuShiftFromStartXCur = cos(imuRollStart) * x2 + sin(imuRollStart) * y2;
  imuShiftFromStartYCur = -sin(imuRollStart) * x2 + cos(imuRollStart) * y2;
  imuShiftFromStartZCur = z2;
}

//计算局部坐标系下点云中的点相对第一个开始点由于加减速产生的的速度畸变（增量）
void VeloToStartIMU()
{
  //计算相对于第一个点由于加减速产生的畸变速度(全局坐标系下畸变速度增量delta_Vg)
  // 同理，假设一个sweep中机器人做匀速运动，其速度为第一个点的速度
  imuVeloFromStartXCur = imuVeloXCur - imuVeloXStart;
  imuVeloFromStartYCur = imuVeloYCur - imuVeloYStart;
  imuVeloFromStartZCur = imuVeloZCur - imuVeloZStart;

  /********************************************************************************
    Rz(pitch).inverse * Rx(pitch).inverse * Ry(yaw).inverse * delta_Vg
    transfrom from the global frame to the local frame
  *********************************************************************************/
  
  //绕y轴旋转(-imuYawStart)，即Ry(yaw).inverse
  float x1 = cos(imuYawStart) * imuVeloFromStartXCur - sin(imuYawStart) * imuVeloFromStartZCur;
  float y1 = imuVeloFromStartYCur;
  float z1 = sin(imuYawStart) * imuVeloFromStartXCur + cos(imuYawStart) * imuVeloFromStartZCur;

  //绕x轴旋转(-imuPitchStart)，即Rx(pitch).inverse
  float x2 = x1;
  float y2 = cos(imuPitchStart) * y1 + sin(imuPitchStart) * z1;
  float z2 = -sin(imuPitchStart) * y1 + cos(imuPitchStart) * z1;

  //绕z轴旋转(-imuRollStart)，即Rz(pitch).inverse
  imuVeloFromStartXCur = cos(imuRollStart) * x2 + sin(imuRollStart) * y2;
  imuVeloFromStartYCur = -sin(imuRollStart) * x2 + cos(imuRollStart) * y2;
  imuVeloFromStartZCur = z2;
}

//去除点云加减速产生的位移畸变
void TransformToStartIMU(PointType *p)
{
  /********************************************************************************
    (变换后的轴)Ry(yaw)*Rx(pitch)*Rz(roll)*Pl,原始R_n_b = Rz(yaw)*Ry(pitch)*Rx(roll), 
    transform point to the global frame，将激光雷达坐标下的点转换到世界坐标!!!
  *********************************************************************************/
  //绕z轴旋转(imuRollCur)
  float x1 = cos(imuRollCur) * p->x - sin(imuRollCur) * p->y;
  float y1 = sin(imuRollCur) * p->x + cos(imuRollCur) * p->y;
  float z1 = p->z;

  //绕x轴旋转(imuPitchCur)
  float x2 = x1;
  float y2 = cos(imuPitchCur) * y1 - sin(imuPitchCur) * z1;
  float z2 = sin(imuPitchCur) * y1 + cos(imuPitchCur) * z1;

  //绕y轴旋转(imuYawCur)
  float x3 = cos(imuYawCur) * x2 + sin(imuYawCur) * z2;
  float y3 = y2;
  float z3 = -sin(imuYawCur) * x2 + cos(imuYawCur) * z2;

  /********************************************************************************
    Rz(pitch).inverse * Rx(pitch).inverse * Ry(yaw).inverse * Pg
    transfrom global points to the local frame，将世界坐标下的激光点数据转换到第一个点对应的imu坐标系下
  *********************************************************************************/
  
  //绕y轴旋转(-imuYawStart)
  float x4 = cos(imuYawStart) * x3 - sin(imuYawStart) * z3;
  float y4 = y3;
  float z4 = sin(imuYawStart) * x3 + cos(imuYawStart) * z3;

  //绕x轴旋转(-imuPitchStart)
  float x5 = x4;
  float y5 = cos(imuPitchStart) * y4 + sin(imuPitchStart) * z4;
  float z5 = -sin(imuPitchStart) * y4 + cos(imuPitchStart) * z4;

  //绕z轴旋转(-imuRollStart)，然后叠加平移量
  p->x = cos(imuRollStart) * x5 + sin(imuRollStart) * y5 + imuShiftFromStartXCur;
  p->y = -sin(imuRollStart) * x5 + cos(imuRollStart) * y5 + imuShiftFromStartYCur;
  p->z = z5 + imuShiftFromStartZCur;
}

//世界坐标系下积分速度与位移
void AccumulateIMUShift()
{
  float roll = imuRoll[imuPointerLast];   //世界坐标下的姿态，最新imu数据
  float pitch = imuPitch[imuPointerLast];
  float yaw = imuYaw[imuPointerLast];
  float accX = imuAccX[imuPointerLast]; // 交换后的左上前imu坐标系(局部)下的加速度
  float accY = imuAccY[imuPointerLast];
  float accZ = imuAccZ[imuPointerLast];
  // a_g = R_g_i * a_i 
  //将当前时刻的加速度值(交换过的)绕交换过的ZXY固定轴（原XYZ）分别旋转(roll, pitch, yaw)角，转换得到世界坐标系下的加速度值(right hand rule)
  // 因为交换过后RPY对应fixed axes ZXY(RPY---ZXY)，Now R_n_b = R_XYZ = Ry(yaw)*Rx(pitch)*Rz(roll).
  // 而且因为现在加速度是交换过后的值，所以要求世界坐标下的值的话按需要照交换前的XYZ顺序来进行转换处理

  //绕x轴旋转(roll)
  float x1 = cos(roll) * accX - sin(roll) * accY;
  float y1 = sin(roll) * accX + cos(roll) * accY;
  float z1 = accZ;
  //绕y轴旋转(pitch)
  float x2 = x1;
  float y2 = cos(pitch) * y1 - sin(pitch) * z1;
  float z2 = sin(pitch) * y1 + cos(pitch) * z1;
  //绕z轴旋转(yaw)
  accX = cos(yaw) * x2 + sin(yaw) * z2;
  accY = y2;
  accZ = -sin(yaw) * x2 + cos(yaw) * z2;

  //上一个imu点
  int imuPointerBack = (imuPointerLast + imuQueLength - 1) % imuQueLength;///为什么要加上imuQueLength???防止imuPointerLast为0时产生误解
  //上一个点到当前点所经历的时间，即计算imu测量周期
  double timeDiff = imuTime[imuPointerLast] - imuTime[imuPointerBack];
  //要求imu的频率至少比lidar高，这样的imu信息才使用，后面校正也才有意义
  if (timeDiff < scanPeriod) {//（隐含从静止开始运动）
    //求每个imu时间点的位移(相对于世界坐标原点)与速度(均在世界坐标下),两点之间视为匀/减加速直线运动，类似于惯性导航的样子!!!
    // IMU第一帧数据时，上一点(虚假的上一点，并不是真的上一个点)的速度，位移都为0
    imuShiftX[imuPointerLast] = imuShiftX[imuPointerBack] + imuVeloX[imuPointerBack] * timeDiff 
                              + accX * timeDiff * timeDiff / 2;
    imuShiftY[imuPointerLast] = imuShiftY[imuPointerBack] + imuVeloY[imuPointerBack] * timeDiff 
                              + accY * timeDiff * timeDiff / 2;
    imuShiftZ[imuPointerLast] = imuShiftZ[imuPointerBack] + imuVeloZ[imuPointerBack] * timeDiff 
                              + accZ * timeDiff * timeDiff / 2;
    //当前imu在世界坐标下的速度
    imuVeloX[imuPointerLast] = imuVeloX[imuPointerBack] + accX * timeDiff;
    imuVeloY[imuPointerLast] = imuVeloY[imuPointerBack] + accY * timeDiff;
    imuVeloZ[imuPointerLast] = imuVeloZ[imuPointerBack] + accZ * timeDiff;
  }
}

//接收点云数据，velodyne雷达坐标系安装为x轴向前，y轴向左，z轴向上的右手坐标系(与IMU安装的坐标系一样)
// 该函数的主要功能是对接收到的原始点云进行预处理，完成分类。具体分类内容为：一是将原始点云划入不同scan线中存储；二是对其进行特征分类。
void laserCloudHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg)
{
  if (!systemInited) {//丢弃前20个点云数据
    systemInitCount++;
    if (systemInitCount >= systemDelay) { // systemDelay有延时作用，保证有imu数据后在调用laserCloudHandler
      systemInited = true;
    }
    return;
  }

  //记录每个scan有曲率的点的开始和结束索引，这个索引是基于点云的点序的
  std::vector<int> scanStartInd(N_SCANS, 0);//N_SCANS(N_SCANS表示激光雷达线数)个重复的元素，每个元素的值都是0
  std::vector<int> scanEndInd(N_SCANS, 0);
  
  //当前点云时间戳，这个时间戳记录的是点云中发射出第一点的时间戳
  double timeScanCur = laserCloudMsg->header.stamp.toSec();
  pcl::PointCloud<pcl::PointXYZ> laserCloudIn;
  //ros点云消息转换成pcl数据存放
  pcl::fromROSMsg(*laserCloudMsg, laserCloudIn);
  std::vector<int> indices;
  //移除空点
  pcl::removeNaNFromPointCloud(laserCloudIn, laserCloudIn, indices);
  //点云中点的数量
  int cloudSize = laserCloudIn.points.size();
  //lidar scan开始点的旋转角,atan2范围(-pi,+pi],计算旋转角时取负号是因为velodyne是顺时针旋转
  float startOri = -atan2(laserCloudIn.points[0].y, laserCloudIn.points[0].x); // 此时还是安装好的前左上坐标系
  //lidar scan结束点的旋转角，加2*pi使点云旋转周期为2*pi ////如果结束点在0度附近，
  float endOri = -atan2(laserCloudIn.points[cloudSize - 1].y,
                        laserCloudIn.points[cloudSize - 1].x) + 2 * M_PI;//todo 个人理解是相当于求解与-X轴的夹角

  //结束方位角与开始方位角差值控制在(PI,3*PI)范围，允许lidar不是一个圆周扫描
  //正常情况下在这个范围内：pi < endOri - startOri < 3*pi，异常则修正
  if (endOri - startOri > 3 * M_PI) {
    endOri -= 2 * M_PI;
  } else if (endOri - startOri < M_PI) {
    endOri += 2 * M_PI;
  }
  //lidar扫描线是否旋转过半
  bool halfPassed = false;
  int count = cloudSize;//点云点的数量
  PointType point;
  // 遍历所有点，根据其垂直方向的角度计算结果将其划分为不同的scan线：计算角度-->计算起始和终止位置-->插入IMU数据-->将点插入容器中
  std::vector<pcl::PointCloud<PointType> > laserCloudScans(N_SCANS);
  for (int i = 0; i < cloudSize; i++) {
    //坐标轴交换，velodyne lidar的坐标系(前左上)也转换到z轴向前，x轴向左，y轴向上(左上前)的右手坐标系
    point.x = laserCloudIn.points[i].y;
    point.y = laserCloudIn.points[i].z;
    point.z = laserCloudIn.points[i].x;

    //在转换后的(左上前)坐标系中计算点的仰角(根据lidar文档垂直角计算公式),根据仰角排列激光线号，velodyne每两个scan之间间隔2度
    float angle = atan(point.y / sqrt(point.x * point.x + point.z * point.z)) * 180 / M_PI;
    int scanID;
    /* 这里简单介绍一个多线激光的原理
      30°FOV    ----------------------->顺时针发射激光(水平)
         /|\  ID    角度/°                获取到的激光点云数据
          |   15    15      ------------------------------------------------    (一次scan)
      发  |   13    13      ------------------------------------------------
      射  |   :      :                            :
      顺  |   3      3      ------------------------------------------------
      序  |   1      1      ------------------------------------------------
      |   |-.-----.-----.------.-------.---.----.-----.---.----.----.------.---.----    0°(垂直方向)
      垂  |   14    -1      ------------------------------------------------
      直  |   12    -3      ------------------------------------------------
          |   :     :                             :
          |   2     -13     ------------------------------------------------
          |   0     -15     ------------------------------------------------

          这里，所有的多线激光数据称作一次sweep，也就是一帧
    */
    //仰角四舍五入(加减0.5截断效果等于四舍五入)
    int roundedAngle = int(angle + (angle<0.0?-0.5:+0.5)); 
    if (roundedAngle > 0){
      scanID = roundedAngle;
    }
    else {
      scanID = roundedAngle + (N_SCANS - 1);
    }
    //过滤点，只挑选[-15度，+15度]范围内的点,scanID属于[0,15]，因为这里用的是十六线激光雷达
    if (scanID > (N_SCANS - 1) || scanID < 0 ){
      count--;
      continue;
    }

    //该点的旋转角
    // 因为转一圈可能会超过2pi， 故角度a可能对应a或者2pi + a
    // 如何确定是a还是2pi+a呢， half_passed 利用点的顺序与时间先后相关这一点解决了这个问题
    // 见https://zhuanlan.zhihu.com/p/57351961 解析
    float ori = -atan2(point.x, point.z);// atan2的角度，值位于-pi到pi之间
    if (!halfPassed) {//根据扫描线是否旋转过半选择与起始位置还是终止位置进行差值计算，从而进行补偿
        //确保-pi/2 < ori - startOri < 3*pi/2
      if (ori < startOri - M_PI / 2) {
        ori += 2 * M_PI;
      } else if (ori > startOri + M_PI * 3 / 2) {
        ori -= 2 * M_PI;
      }
      //半个周期
      if (ori - startOri > M_PI) {
        halfPassed = true;
      }
    } else {
      ori += 2 * M_PI;

      //确保-3*pi/2 < ori - endOri < pi/2
      if (ori < endOri - M_PI * 3 / 2) {
        ori += 2 * M_PI;
      } else if (ori > endOri + M_PI / 2) {
        ori -= 2 * M_PI;
      } 
    }

    //-0.5 < relTime < 1.5（点旋转的角度与整个周期旋转角度的比率, 即点云中点的相对时间）// 根据水平角度可以得到获取每个点时相对于开始点的时间relTime
    float relTime = (ori - startOri) / (endOri - startOri);
    // 对点云数据格式信息的利用达到最大化：把pcl::PointXYZI中的intensity设置为 对应线束id + 相对开始点的时间，一个数据包含了两个信息！！
    //点强度=线号+点相对时间（即一个整数+一个小数，整数部分是线号，小数部分是该点的相对时间）,匀速扫描：根据当前扫描的角度和扫描周期计算相对扫描起始位置的时间
    point.intensity = scanID + scanPeriod * relTime;

    //相对时间relTime用来和IMU数据一起近似去除激光的非匀速运动，构建匀速运动模型。
    //点时间=点云时间+周期时间 // imuPointerLast 是imu当前点，变量只在imu中改变，设为t时刻
    if (imuPointerLast >= 0) {//如果收到IMU数据,使用IMU矫正点云畸变
      float pointTime = relTime * scanPeriod;//计算点的周期时间(其实就是当前点相对于点云起始点的时间)
      //寻找是否有点云的时间戳刚好小于IMU的时间戳的IMU位置:imuPointerFront
      while (imuPointerFront != imuPointerLast) {
        if (timeScanCur + pointTime < imuTime[imuPointerFront]) {//找到ti(点云中该点的时间)后的最近一个imu时刻
          break;
        }
        imuPointerFront = (imuPointerFront + 1) % imuQueLength;
      }
      //timeScanCur + pointTime是ti时刻(第i个点扫描的时间;imuPointerFront是ti后一时刻的imu时间,imuPointerBack是ti前一时刻的imu时间
      if (timeScanCur + pointTime > imuTime[imuPointerFront]) {//没找到,此时imuPointerFront==imtPointerLast,只能以当前收到的最新的IMU的速度，位移，欧拉角作为当前点的速度，位移，欧拉角使用
        imuRollCur = imuRoll[imuPointerFront]; // 当前时刻imu的世界姿态，这里的姿态还是基于未进行坐标变换的imu
        imuPitchCur = imuPitch[imuPointerFront];
        imuYawCur = imuYaw[imuPointerFront];

        imuVeloXCur = imuVeloX[imuPointerFront]; // 当前时刻imu在世界坐标下的速度
        imuVeloYCur = imuVeloY[imuPointerFront];
        imuVeloZCur = imuVeloZ[imuPointerFront];

        imuShiftXCur = imuShiftX[imuPointerFront]; // 当前时刻imu在世界坐标下的位移
        imuShiftYCur = imuShiftY[imuPointerFront];
        imuShiftZCur = imuShiftZ[imuPointerFront];
      } else {//找到了点云时间戳小于IMU时间戳的IMU位置,则该点必处于imuPointerBack和imuPointerFront之间，据此线性插值，计算点云点的速度，位移和欧拉角
        int imuPointerBack = (imuPointerFront + imuQueLength - 1) % imuQueLength;
        //按时间距离计算权重分配比率,也即线性插值
        float ratioFront = (timeScanCur + pointTime - imuTime[imuPointerBack]) 
                         / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
        float ratioBack = (imuTime[imuPointerFront] - timeScanCur - pointTime) 
                        / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);

        imuRollCur = imuRoll[imuPointerFront] * ratioFront + imuRoll[imuPointerBack] * ratioBack;//用ti时间前后的两个imu数据进行插值
        imuPitchCur = imuPitch[imuPointerFront] * ratioFront + imuPitch[imuPointerBack] * ratioBack;
        if (imuYaw[imuPointerFront] - imuYaw[imuPointerBack] > M_PI) {// 个人理解这里是对imu两个时间跨越正负pi的角度作处理，因为偏航角范围为[-pi,+pi]
          imuYawCur = imuYaw[imuPointerFront] * ratioFront + (imuYaw[imuPointerBack] + 2 * M_PI) * ratioBack;
        } else if (imuYaw[imuPointerFront] - imuYaw[imuPointerBack] < -M_PI) {
          imuYawCur = imuYaw[imuPointerFront] * ratioFront + (imuYaw[imuPointerBack] - 2 * M_PI) * ratioBack;
        } else {
          imuYawCur = imuYaw[imuPointerFront] * ratioFront + imuYaw[imuPointerBack] * ratioBack;
        }

        //本质:imuVeloXCur = imuVeloX[imuPointerback] + (imuVelX[imuPointerFront]-imuVelX[imuPoniterBack])*ratioFront
        imuVeloXCur = imuVeloX[imuPointerFront] * ratioFront + imuVeloX[imuPointerBack] * ratioBack;
        imuVeloYCur = imuVeloY[imuPointerFront] * ratioFront + imuVeloY[imuPointerBack] * ratioBack;
        imuVeloZCur = imuVeloZ[imuPointerFront] * ratioFront + imuVeloZ[imuPointerBack] * ratioBack;

        imuShiftXCur = imuShiftX[imuPointerFront] * ratioFront + imuShiftX[imuPointerBack] * ratioBack;
        imuShiftYCur = imuShiftY[imuPointerFront] * ratioFront + imuShiftY[imuPointerBack] * ratioBack;
        imuShiftZCur = imuShiftZ[imuPointerFront] * ratioFront + imuShiftZ[imuPointerBack] * ratioBack;
      }

      if (i == 0) {//如果是第一个点,记住点云起始位置对应的imu的速度，位移，欧拉角
        imuRollStart = imuRollCur;
        imuPitchStart = imuPitchCur;
        imuYawStart = imuYawCur;

        imuVeloXStart = imuVeloXCur;
        imuVeloYStart = imuVeloYCur;
        imuVeloZStart = imuVeloZCur;

        imuShiftXStart = imuShiftXCur;
        imuShiftYStart = imuShiftYCur;
        imuShiftZStart = imuShiftZCur;
      } else {//计算之后每个点相对于第一个点的由于加减速非匀速运动产生的位移速度畸变，并对点云中的每个点位置信息重新补偿矫正
        ShiftToStartIMU(pointTime);// 将Lidar位移转到IMU起始坐标系下//计算局部坐标系下点云中的点相对第一个开始点的由于加减速运动产生的位移畸变
        VeloToStartIMU();// 将Lidar运动速度转到IMU起始坐标系下//计算局部坐标系下点云中的点相对第一个开始点由于加减速产生的的速度畸变（增量）
        TransformToStartIMU(&point);// 将点坐标转到起始IMU坐标系(与点云中第一个点对应)下
      }
    }
    laserCloudScans[scanID].push_back(point);//将点按照每一层线，分类压入16个数组中，将每个补偿矫正的点放入对应线号的容器
  }

  //获得有效范围内的点的数量
  cloudSize = count;

  pcl::PointCloud<PointType>::Ptr laserCloud(new pcl::PointCloud<PointType>());
  for (int i = 0; i < N_SCANS; i++) {//将所有点存入laserCloud中，点按(线序)进行排列
    *laserCloud += laserCloudScans[i];
  }
  int scanCount = -1;//scanCount 可以表明当前遍历到多少层scan了
  /////Q 按照这种算法的话，其实除了第一层开始跟最后一层结束五个点没有参与计算，其他每一层前后五个点都计算进去了，这里感觉可以重新设计一下，减少计算量。
  for (int i = 5; i < cloudSize - 5; i++) {//使用每个点的前后五个点计算曲率，因此前五个与最后五个点跳过
    float diffX = laserCloud->points[i - 5].x + laserCloud->points[i - 4].x 
                + laserCloud->points[i - 3].x + laserCloud->points[i - 2].x 
                + laserCloud->points[i - 1].x - 10 * laserCloud->points[i].x 
                + laserCloud->points[i + 1].x + laserCloud->points[i + 2].x
                + laserCloud->points[i + 3].x + laserCloud->points[i + 4].x
                + laserCloud->points[i + 5].x;
    float diffY = laserCloud->points[i - 5].y + laserCloud->points[i - 4].y 
                + laserCloud->points[i - 3].y + laserCloud->points[i - 2].y 
                + laserCloud->points[i - 1].y - 10 * laserCloud->points[i].y 
                + laserCloud->points[i + 1].y + laserCloud->points[i + 2].y
                + laserCloud->points[i + 3].y + laserCloud->points[i + 4].y
                + laserCloud->points[i + 5].y;
    float diffZ = laserCloud->points[i - 5].z + laserCloud->points[i - 4].z 
                + laserCloud->points[i - 3].z + laserCloud->points[i - 2].z 
                + laserCloud->points[i - 1].z - 10 * laserCloud->points[i].z 
                + laserCloud->points[i + 1].z + laserCloud->points[i + 2].z
                + laserCloud->points[i + 3].z + laserCloud->points[i + 4].z
                + laserCloud->points[i + 5].z;
    //曲率计算，对应论文公式(1)，此处没有做normalize处理
    cloudCurvature[i] = diffX * diffX + diffY * diffY + diffZ * diffZ;
    //记录曲率点在点云中的索引
    cloudSortInd[i] = i;
    //初始时，点全未筛选过
    cloudNeighborPicked[i] = 0;
    //初始化为less flat次平面点
    cloudLabel[i] = 0;

    //每个scan，只有第一个符合的点会进来，因为每个scan的点都在一起存放
    if (int(laserCloud->points[i].intensity) != scanCount) {//scanCount 可以表明当前遍历到多少层scan了
      scanCount = int(laserCloud->points[i].intensity);//控制每个scan只进入第一个点

      //曲率只取同一个scan计算出来的，跨scan计算的曲率非法，排除，也即排除每个scan的前后五个点
      if (scanCount > 0 && scanCount < N_SCANS) {//注意scanCount大于0，刚开始第一层scan时不走这个逻辑，所以下面会对第一个scan的开始索引进行设置
        //每个scan有曲率的点的开始和结束索引，在这里，他其实记录的是当前层scan第一个有曲率的点的所在的scan层的开始索引与上一层scan的结束索引。
        scanStartInd[scanCount] = i + 5;//因为曲率计算需要有前后各五个点来计算，所以每一层scan开始和结束的五个点都被抛弃了
        scanEndInd[scanCount - 1] = i - 5;
      }
    }
  }
  //第一个scan曲率点有效点序从第5个开始，最后一个激光线结束点序size-5(上面的if语句中只能设置最后一个激光线scan的开始索引，所以要手动设置一个结束索引)
  scanStartInd[0] = 5;
  scanEndInd.back() = cloudSize - 5;

  //挑选点，排除容易被斜面挡住的点以及离群点，有些点容易被斜面挡住，而离群点可能出现带有偶然性，这些情况都可能导致前后两次扫描不能被同时看到
  for (int i = 5; i < cloudSize - 6; i++) {//与后一个点差值，所以减6
    float diffX = laserCloud->points[i + 1].x - laserCloud->points[i].x;
    float diffY = laserCloud->points[i + 1].y - laserCloud->points[i].y;
    float diffZ = laserCloud->points[i + 1].z - laserCloud->points[i].z;
    //计算有效曲率点与后一个点之间的距离平方和
    float diff = diffX * diffX + diffY * diffY + diffZ * diffZ;

    if (diff > 0.1) {// 两个点之间距离大于0.1，可能出现被遮挡的情况

      //当前点的深度
      float depth1 = sqrt(laserCloud->points[i].x * laserCloud->points[i].x + 
                     laserCloud->points[i].y * laserCloud->points[i].y +
                     laserCloud->points[i].z * laserCloud->points[i].z);

      //后一个点的深度
      float depth2 = sqrt(laserCloud->points[i + 1].x * laserCloud->points[i + 1].x + 
                     laserCloud->points[i + 1].y * laserCloud->points[i + 1].y +
                     laserCloud->points[i + 1].z * laserCloud->points[i + 1].z);
      /*— 针对论文的(b)情况，两向量夹角小于某阈值b时（夹角小就可能存在遮挡），将其一侧的临近6个点设为不可标记为特征点的点 —*/
      /*— 构建了一个等腰三角形的底向量，根据等腰三角形性质，判断X[i]向量与X[i+1]的夹角小于5.732度(threshold=0.1) —*/
      /*— depth1>depth2 X[i+1]距离更近，远侧点标记不特征；depth1<depth2 X[i]距离更近，远侧点标记不特征 —*/

      //按照两点的深度的比例，将深度较大的点拉回后计算距离(构建等腰三角形) // 深度较大的点附近有被遮挡的风险
      if (depth1 > depth2) {
        diffX = laserCloud->points[i + 1].x - laserCloud->points[i].x * depth2 / depth1;
        diffY = laserCloud->points[i + 1].y - laserCloud->points[i].y * depth2 / depth1;
        diffZ = laserCloud->points[i + 1].z - laserCloud->points[i].z * depth2 / depth1;

        //边长比也即是弧度值，若小于0.1，说明夹角比较小，斜面比较陡峭,点深度变化比较剧烈 // 这句话感觉可以去掉  ---点处在近似与激光束平行的斜面上---
        if (sqrt(diffX * diffX + diffY * diffY + diffZ * diffZ) / depth2 < 0.1) {//排除容易被斜面挡住的点
          //该点及前面五个点（大致都在斜面上）全部置为筛选过
          // 当某点及其后点间的距离平方大于某阈值a（说明这两点有一定距离），且两向量夹角小于某阈值b时（夹角小就可能存在遮挡）   
          // 将其一侧的临近6个点设为不可标记为特征点的点 /////Q是否可以试试两边都标记一下来排除掉异常??
          cloudNeighborPicked[i - 5] = 1;
          cloudNeighborPicked[i - 4] = 1;
          cloudNeighborPicked[i - 3] = 1;
          cloudNeighborPicked[i - 2] = 1;
          cloudNeighborPicked[i - 1] = 1;
          cloudNeighborPicked[i] = 1;
          // 因为处理过后的点是按照逆时针的顺序发射的，而且该点i的深度也是比另一个点大的，存在遮挡的可能，所以这里是除了标记五个点还要标记自身
        }
      } else {
        diffX = laserCloud->points[i + 1].x * depth1 / depth2 - laserCloud->points[i].x;
        diffY = laserCloud->points[i + 1].y * depth1 / depth2 - laserCloud->points[i].y;
        diffZ = laserCloud->points[i + 1].z * depth1 / depth2 - laserCloud->points[i].z;

        if (sqrt(diffX * diffX + diffY * diffY + diffZ * diffZ) / depth1 < 0.1) {
          cloudNeighborPicked[i + 1] = 1; // 此时点i的深度较浅，不存在遮挡的可能，故无需标记点i，只需要标记其他几个点
          cloudNeighborPicked[i + 2] = 1;
          cloudNeighborPicked[i + 3] = 1;
          cloudNeighborPicked[i + 4] = 1;
          cloudNeighborPicked[i + 5] = 1;
          cloudNeighborPicked[i + 6] = 1;
        }
      }
    }

    /*— 针对论文的(a)情况，当某点及其后点间的距离平方大于某阈值a（说明这两点有一定距离） ———*/
    /*— 若某点到其前后两点的距离均大于c倍的该点深度，则该点判定为不可标记特征点的点 ———————*/
    /*—（入射角越小，点间距越大，即激光发射方向与投射到的平面越近似水平） ———————————————*/
    float diffX2 = laserCloud->points[i].x - laserCloud->points[i - 1].x;
    float diffY2 = laserCloud->points[i].y - laserCloud->points[i - 1].y;
    float diffZ2 = laserCloud->points[i].z - laserCloud->points[i - 1].z;
    //与前一个点的距离平方和
    float diff2 = diffX2 * diffX2 + diffY2 * diffY2 + diffZ2 * diffZ2;

    //点深度的平方和
    float dis = laserCloud->points[i].x * laserCloud->points[i].x
              + laserCloud->points[i].y * laserCloud->points[i].y
              + laserCloud->points[i].z * laserCloud->points[i].z;

    //与前后点的平方和都大于深度平方和的万分之二，这些点视为离群点，包括陡斜面上的点，强烈凸凹点和空旷区域中的某些点，置为筛选过，弃用
    if (diff > 0.0002 * dis && diff2 > 0.0002 * dis) {
      cloudNeighborPicked[i] = 1;
    }
  }


  pcl::PointCloud<PointType> cornerPointsSharp;     // 边缘点
  pcl::PointCloud<PointType> cornerPointsLessSharp; // 次边缘点
  pcl::PointCloud<PointType> surfPointsFlat;        // 平面点
  pcl::PointCloud<PointType> surfPointsLessFlat;    // 次平面点(滤波后)

  //将每条scan线上的点分入相应的类别：边缘点和平面点
  for (int i = 0; i < N_SCANS; i++) {
    pcl::PointCloud<PointType>::Ptr surfPointsLessFlatScan(new pcl::PointCloud<PointType>);//未滤波的次平面点
    //将每个scan的曲率点分成6等份处理,确保周围都有点被选作特征点
    for (int j = 0; j < 6; j++) {
      //六等份起点：sp = scanStartInd + (scanEndInd - scanStartInd)*j/6
      int sp = (scanStartInd[i] * (6 - j)  + scanEndInd[i] * j) / 6;
      //六等份终点：ep = scanStartInd - 1 + (scanEndInd - scanStartInd)*(j+1)/6
      int ep = (scanStartInd[i] * (5 - j)  + scanEndInd[i] * (j + 1)) / 6 - 1;

      //按曲率从小到大冒泡排序
      for (int k = sp + 1; k <= ep; k++) {
        for (int l = k; l >= sp + 1; l--) {
          //如果后面曲率点小于前面，则交换
          if (cloudCurvature[cloudSortInd[l]] < cloudCurvature[cloudSortInd[l - 1]]) {
            int temp = cloudSortInd[l - 1];
            cloudSortInd[l - 1] = cloudSortInd[l];
            cloudSortInd[l] = temp;// 此时cloudSortInd里面的数据就是按照曲率从小到大排列好的曲率点的id了(注意是分区域的，即每一层scan有六个区域)
          }
        }
      }

      //挑选每个分段的曲率很大和比较大的点
      int largestPickedNum = 0;
      for (int k = ep; k >= sp; k--) {
        int ind = cloudSortInd[k];  //最开始进入那就是曲率最大点的点序(id)
        // 因为前面用了从小到大的冒泡排序，曲率最大的点的id肯定保存在cloudSortInd[ep]处，相反曲率最小的点的id肯定保存在cloudSortInd[sp]处

        //如果曲率大的点，曲率的确比较大，并且未被筛选过滤掉
        if (cloudNeighborPicked[ind] == 0 &&
            cloudCurvature[ind] > 0.1) {
        
          largestPickedNum++;  /*—— 筛选特征角点 Corner: label=2; LessCorner: label=1 ————*/
          if (largestPickedNum <= 2) {//挑选曲率最大的前2个点放入sharp边缘点集合
            //点分类标号:2-代表曲率很大，1-代表曲率比较大,-1-代表曲率很小，0-曲率比较小(其中1包含了2,0包含了1,0和1构成了点云全部的点)
            cloudLabel[ind] = 2;//2代表点曲率很大
            cornerPointsSharp.push_back(laserCloud->points[ind]); // 边缘点点云
            cornerPointsLessSharp.push_back(laserCloud->points[ind]);//cornerPointsLessSharp包含了ind为2和ind为1的点
          } else if (largestPickedNum <= 20) {//挑选曲率最大的前20个点放入less sharp点集合
            cloudLabel[ind] = 1;//1代表点曲率比较大
            cornerPointsLessSharp.push_back(laserCloud->points[ind]); // 次边缘点点云
          } else {
            break;
          }

          cloudNeighborPicked[ind] = 1;//筛选标志置位

          //将曲率比较大的点的前后各5个连续距离比较近的点筛选出去，防止特征点聚集，使得特征点在每个方向上尽量分布均匀
          for (int l = 1; l <= 5; l++) {
            float diffX = laserCloud->points[ind + l].x 
                        - laserCloud->points[ind + l - 1].x;
            float diffY = laserCloud->points[ind + l].y 
                        - laserCloud->points[ind + l - 1].y;
            float diffZ = laserCloud->points[ind + l].z 
                        - laserCloud->points[ind + l - 1].z;
            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05) { //点与点之间距离较近，筛出去
              break;
            }

            cloudNeighborPicked[ind + l] = 1;
          }
          for (int l = -1; l >= -5; l--) {
            float diffX = laserCloud->points[ind + l].x 
                        - laserCloud->points[ind + l + 1].x;
            float diffY = laserCloud->points[ind + l].y 
                        - laserCloud->points[ind + l + 1].y;
            float diffZ = laserCloud->points[ind + l].z 
                        - laserCloud->points[ind + l + 1].z;
            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05) {
              break;
            }

            cloudNeighborPicked[ind + l] = 1;
          }
        }
      }

      //挑选每个分段的曲率很小比较小的点
      int smallestPickedNum = 0;
      for (int k = sp; k <= ep; k++) {
        int ind = cloudSortInd[k];//刚开始进入时为曲率最小点的点序(id)

        //如果曲率的确比较小，并且未被筛选出
        if (cloudNeighborPicked[ind] == 0 &&
            cloudCurvature[ind] < 0.1) {

          cloudLabel[ind] = -1;//-1代表曲率很小的点
          surfPointsFlat.push_back(laserCloud->points[ind]); // 平面点点云

          smallestPickedNum++;
          if (smallestPickedNum >= 4) {//只选最小的四个，剩下的Label==0,就都是曲率比较小的
            break;
          }

          cloudNeighborPicked[ind] = 1;
          for (int l = 1; l <= 5; l++) {//同样防止特征点聚集
            float diffX = laserCloud->points[ind + l].x 
                        - laserCloud->points[ind + l - 1].x;
            float diffY = laserCloud->points[ind + l].y 
                        - laserCloud->points[ind + l - 1].y;
            float diffZ = laserCloud->points[ind + l].z 
                        - laserCloud->points[ind + l - 1].z;
            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05) {
              break;
            }

            cloudNeighborPicked[ind + l] = 1;
          }
          for (int l = -1; l >= -5; l--) {
            float diffX = laserCloud->points[ind + l].x 
                        - laserCloud->points[ind + l + 1].x;
            float diffY = laserCloud->points[ind + l].y 
                        - laserCloud->points[ind + l + 1].y;
            float diffZ = laserCloud->points[ind + l].z 
                        - laserCloud->points[ind + l + 1].z;
            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05) {
              break;
            }

            cloudNeighborPicked[ind + l] = 1;
          }
        }
      }

      //将剩余的点（包括之前被排除的点）全部归入次平面点中less flat类别中
      for (int k = sp; k <= ep; k++) {
        if (cloudLabel[k] <= 0) {
          surfPointsLessFlatScan->push_back(laserCloud->points[k]); // 平面点+次平面点点云
        }
      }
    } // 对每层scan中的每个区域的处理

    //由于less flat点最多，对每个分段less flat的点进行体素栅格滤波
    pcl::PointCloud<PointType> surfPointsLessFlatScanDS;
    pcl::VoxelGrid<PointType> downSizeFilter;
    downSizeFilter.setInputCloud(surfPointsLessFlatScan);
    downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
    downSizeFilter.filter(surfPointsLessFlatScanDS);

    //less flat 平面点+次平面点汇总
    surfPointsLessFlat += surfPointsLessFlatScanDS;
  }

  //publich消除非匀速运动畸变后的所有的点
  sensor_msgs::PointCloud2 laserCloudOutMsg;
  pcl::toROSMsg(*laserCloud, laserCloudOutMsg);
  laserCloudOutMsg.header.stamp = laserCloudMsg->header.stamp;
  laserCloudOutMsg.header.frame_id = "/camera";
  pubLaserCloud.publish(laserCloudOutMsg);

  //publich消除非匀速运动畸变后的边缘点
  sensor_msgs::PointCloud2 cornerPointsSharpMsg;
  pcl::toROSMsg(cornerPointsSharp, cornerPointsSharpMsg);
  cornerPointsSharpMsg.header.stamp = laserCloudMsg->header.stamp;
  cornerPointsSharpMsg.header.frame_id = "/camera";
  pubCornerPointsSharp.publish(cornerPointsSharpMsg);
  //publich消除非匀速运动畸变后的次边缘点
  sensor_msgs::PointCloud2 cornerPointsLessSharpMsg;
  pcl::toROSMsg(cornerPointsLessSharp, cornerPointsLessSharpMsg);
  cornerPointsLessSharpMsg.header.stamp = laserCloudMsg->header.stamp;
  cornerPointsLessSharpMsg.header.frame_id = "/camera";
  pubCornerPointsLessSharp.publish(cornerPointsLessSharpMsg);
  //publich消除非匀速运动畸变后的平面点
  sensor_msgs::PointCloud2 surfPointsFlat2;
  pcl::toROSMsg(surfPointsFlat, surfPointsFlat2);
  surfPointsFlat2.header.stamp = laserCloudMsg->header.stamp;
  surfPointsFlat2.header.frame_id = "/camera";
  pubSurfPointsFlat.publish(surfPointsFlat2);
  //publich消除非匀速运动畸变后的平面点+次平面点(滤波后)
  sensor_msgs::PointCloud2 surfPointsLessFlat2;
  pcl::toROSMsg(surfPointsLessFlat, surfPointsLessFlat2);
  surfPointsLessFlat2.header.stamp = laserCloudMsg->header.stamp;
  surfPointsLessFlat2.header.frame_id = "/camera";
  pubSurfPointsLessFlat.publish(surfPointsLessFlat2);

  //publich IMU消息,由于循环到了最后，因此是Cur都是代表最后一个点，即最后一个点的欧拉角，畸变位移及一个点云周期增加的速度
  pcl::PointCloud<pcl::PointXYZ> imuTrans(4, 1);
  //起始点欧拉角，世界坐标系下
  imuTrans.points[0].x = imuPitchStart;
  imuTrans.points[0].y = imuYawStart;
  imuTrans.points[0].z = imuRollStart;

  //最后一个点的欧拉角，世界坐标系下
  imuTrans.points[1].x = imuPitchCur;
  imuTrans.points[1].y = imuYawCur;
  imuTrans.points[1].z = imuRollCur;

  //最后一个点相对于第一个点的畸变位移和速度，imu start系下
  imuTrans.points[2].x = imuShiftFromStartXCur;//imuShiftFromStartXCur是局部坐标系下(imu start系下 )的表示
  imuTrans.points[2].y = imuShiftFromStartYCur;
  imuTrans.points[2].z = imuShiftFromStartZCur;

  imuTrans.points[3].x = imuVeloFromStartXCur;
  imuTrans.points[3].y = imuVeloFromStartYCur;
  imuTrans.points[3].z = imuVeloFromStartZCur;

  sensor_msgs::PointCloud2 imuTransMsg;
  pcl::toROSMsg(imuTrans, imuTransMsg);
  imuTransMsg.header.stamp = laserCloudMsg->header.stamp;
  imuTransMsg.header.frame_id = "/camera";
  pubImuTrans.publish(imuTransMsg);
}

//接收imu消息，imu安装的坐标系为x轴向前，y轴向左，z轴向上的右手坐标系
void imuHandler(const sensor_msgs::Imu::ConstPtr& imuIn)
{
  ////Q有个疑问？这个姿态角是怎么算出来的？如果我坐标发生了交换，那么这个姿态又是怎么算的？？
  double roll, pitch, yaw; /////Q 应该增加一个数据检查的过程!!!
  tf::Quaternion orientation;
  //convert Quaternion msg to Quaternion
  tf::quaternionMsgToTF(imuIn->orientation, orientation); /////Q. 模块输出的姿态是否准确??? 加速度要不要做滤波???IMU初始化做不做???
  //This will get the roll pitch and yaw from the matrix about fixed axes X, Y, Z respectively. 
  //Here roll pitch yaw is in the global frame，全局坐标系下的姿态，这里IMU是用的一个现有的模块，可以直接输出角度信息。
  tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
  /*
    旋转矩阵小结：
    R_b_n = R_ZYX = Rx(roll)*Ry(pitch)*Rz(yaw) 表示从导航坐标系(世界坐标系)到本体坐标系(局部坐标系)的旋转矩阵，
    R_n_b = R_XYZ = Rz(yaw)*Ry(pitch)*Rx(roll) 表示从本体坐标系(局部坐标系)到导航坐标系(世界坐标系)的旋转矩阵，
  */
  // 这里，其实隐含了两个步骤：
  // (1) IMU的测量值是一个比力，有重力的影响，需要先减去这个重力因素的影响；a_true = a_measure - R_ZYX * g（R_ZYX：ZYX航空旋转次序，即R_b_n)
  // (2) 因为imu坐标系(前左上)与雷达坐标系(左上前)的方向不同，所以需要把他们的值做一个简单的交换即可。
  // 具体见知乎  https://zhuanlan.zhihu.com/p/263090394 ，这里R_ZYX的作用是把世界坐标的重力转换到当前的imu坐标下
  //减去重力的影响,求出xyz方向的加速度实际值，并进行坐标轴交换，统一到z轴向前,x轴向左，y轴向上的右手坐标系(imu局部坐标系), 
  //交换过后RPY对应fixed axes ZXY(RPY---ZXY)。
  float accX = imuIn->linear_acceleration.y - sin(roll) * cos(pitch) * 9.81;
  float accY = imuIn->linear_acceleration.z - cos(roll) * cos(pitch) * 9.81;
  float accZ = imuIn->linear_acceleration.x + sin(pitch) * 9.81;

  //循环移位效果，形成环形数组  范围0~200
  imuPointerLast = (imuPointerLast + 1) % imuQueLength; //imuQueLength = 200

  imuTime[imuPointerLast] = imuIn->header.stamp.toSec();
  imuRoll[imuPointerLast] = roll; // 当前时刻imu的全局姿态
  imuPitch[imuPointerLast] = pitch;
  imuYaw[imuPointerLast] = yaw;
  imuAccX[imuPointerLast] = accX; // 当前时刻imu的三轴加速度(交换后的左上前坐标系下，即局部坐标系，注意这里不是世界坐标系)
  imuAccY[imuPointerLast] = accY;
  imuAccZ[imuPointerLast] = accZ;

  AccumulateIMUShift();//世界坐标系下的积分速度与位移
}

/*
这一节点主要功能是：对点云和IMU数据进行预处理，用于特征点的配准。
所以这个节点实际上就是一个计算准备的过程，其实就做了一个工作：那就是根据点的曲率c来将点划分为不同的类别
*/
int main(int argc, char** argv)
{
  ros::init(argc, argv, "scanRegistration");
  ros::NodeHandle nh;
  // 接收原始点云数据
  ros::Subscriber subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2> 
                                  ("/velodyne_points", 2, laserCloudHandler);
  // 接收IMU数据
  ros::Subscriber subImu = nh.subscribe<sensor_msgs::Imu> ("/imu/data", 50, imuHandler);
  // 发布消除非匀速运动畸变后的所有的点
  pubLaserCloud = nh.advertise<sensor_msgs::PointCloud2>
                                 ("/velodyne_cloud_2", 2);
  // 发布消除非匀速运动畸变后的边缘点
  pubCornerPointsSharp = nh.advertise<sensor_msgs::PointCloud2>
                                        ("/laser_cloud_sharp", 2);
  // 发布消除非匀速运动畸变后的次边缘点
  pubCornerPointsLessSharp = nh.advertise<sensor_msgs::PointCloud2>
                                            ("/laser_cloud_less_sharp", 2);
  // 发布消除非匀速运动畸变后的平面点
  pubSurfPointsFlat = nh.advertise<sensor_msgs::PointCloud2>
                                       ("/laser_cloud_flat", 2);
  // 发布消除非匀速运动畸变后的次平面点
  pubSurfPointsLessFlat = nh.advertise<sensor_msgs::PointCloud2>
                                           ("/laser_cloud_less_flat", 2);
  // 发布一次sweep中最后一个点相对于第一个点的欧拉角，畸变位移及速度(imu start系下)
  pubImuTrans = nh.advertise<sensor_msgs::PointCloud2> ("/imu_trans", 5);

  ros::spin();

  return 0;
}

