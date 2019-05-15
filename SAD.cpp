#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

int winSize = 7;//匹配窗口的大小   
float sub_Sum;//存储匹配范围的视差和
int DSR = 30;//视差搜索范围 

int sub_kernel(Mat &kernel_left, Mat &kernel_right)
{
	Mat Dif;
	//计算两个数组差的绝对值
	absdiff(kernel_left, kernel_right, Dif);
	Scalar Add;
	Add = sum(Dif);
	sub_Sum = Add[0];
	return sub_Sum;//返回匹配窗像素相减之后的和
}

Mat getDisparity(Mat &left, Mat &right)
{
	int row = left.rows;
	int col = left.cols;

	Mat Kernel_L(Size(winSize, winSize), CV_8UC1, Scalar::all(0));
	Mat Kernel_R(Size(winSize, winSize), CV_8UC1, Scalar::all(0));
	Mat disparity(row, col, CV_8UC1, Scalar(0));//视差图


	for (int i = 0; i < row - winSize; i++)
	{
		for (int j = 0; j < col - winSize; j++)
		{
			//rect是以左下角为坐标原点的xy坐标系
			Kernel_L = left(Rect(j, i, winSize, winSize));
			Mat Temp(1, DSR, CV_32F, Scalar(0));
			
			for (int k = 0; k < DSR; k++)
			{
				int y = j - k;
				if (y >= 0)
				{
					Kernel_R = right(Rect(y, i, winSize, winSize));
					Temp.at<float>(k) = sub_kernel(Kernel_L, Kernel_R);
				}
			}
			//寻找最佳匹配点
			Point minLoc;
			//minMaxLoc寻找矩阵中最小值和最大值的位置. 
			minMaxLoc(Temp, NULL, NULL, &minLoc, NULL);

			int loc = minLoc.x;
			disparity.at<uchar>(i, j) = loc * 16;//得到最后显示的灰度值
		}
	}
	return disparity;
}

int main()
{
	Mat left = imread("left.png");
	Mat right = imread("right.png");
	//Mat left,right;
	//resize(LeftImg, left, left.size(), 0.4, 0.4);
	//resize(RightImg, right, right.size(), 0.4, 0.4);
	Mat Disparity;
	namedWindow("left", WINDOW_AUTOSIZE);
	namedWindow("right", WINDOW_AUTOSIZE);
	imshow("left", left);
	imshow("right", right);

	Disparity = getDisparity(left, right);
	namedWindow("Disparity", WINDOW_AUTOSIZE);
	imshow("Disparity", Disparity);
	waitKey(0);

	return 0;
}