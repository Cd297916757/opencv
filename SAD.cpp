#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

int winSize = 7;//ƥ�䴰�ڵĴ�С   
float sub_Sum;//�洢ƥ�䷶Χ���Ӳ��
int DSR = 30;//�Ӳ�������Χ 

int sub_kernel(Mat &kernel_left, Mat &kernel_right)
{
	Mat Dif;
	//�������������ľ���ֵ
	absdiff(kernel_left, kernel_right, Dif);
	Scalar Add;
	Add = sum(Dif);
	sub_Sum = Add[0];
	return sub_Sum;//����ƥ�䴰�������֮��ĺ�
}

Mat getDisparity(Mat &left, Mat &right)
{
	int row = left.rows;
	int col = left.cols;

	Mat Kernel_L(Size(winSize, winSize), CV_8UC1, Scalar::all(0));
	Mat Kernel_R(Size(winSize, winSize), CV_8UC1, Scalar::all(0));
	Mat disparity(row, col, CV_8UC1, Scalar(0));//�Ӳ�ͼ


	for (int i = 0; i < row - winSize; i++)
	{
		for (int j = 0; j < col - winSize; j++)
		{
			//rect�������½�Ϊ����ԭ���xy����ϵ
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
			//Ѱ�����ƥ���
			Point minLoc;
			//minMaxLocѰ�Ҿ�������Сֵ�����ֵ��λ��. 
			minMaxLoc(Temp, NULL, NULL, &minLoc, NULL);

			int loc = minLoc.x;
			disparity.at<uchar>(i, j) = loc * 16;//�õ������ʾ�ĻҶ�ֵ
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