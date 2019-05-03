#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

#define k_size 3//卷积核大小
#define sigma 1//sigma越大，平滑效果越明显
#define threshold 15//FAST的阈值

const double PI = 4.0*atan(1.0);//π
double gaus[k_size][k_size];

void GetgaussianKernel()
{
	int center = k_size / 2;
	double sum = 0;

	for (int i = 0; i < k_size; i++)
	{
		for (int j = 0; j < k_size; j++)
		{
			gaus[i][j] = exp(-((i - center)*(i - center) + (j - center)*(j - center)) / (2 * sigma*sigma));
			sum += gaus[i][j];
		}
	}
	for (int i = 0; i < k_size; i++)
		for (int j = 0; j < k_size; j++)
			gaus[i][j] /= sum;
}

Mat GaussianFilter(Mat src)
{
	Mat dst;
	GetgaussianKernel();//生成高斯卷积核矩阵
	Mat gaus_kern = Mat(k_size, k_size, CV_64F, gaus);
	filter2D(src, dst, src.depth(), gaus_kern);
	//namedWindow("gaussian_filter", WINDOW_AUTOSIZE);
	//imshow("gaussian_filter", dst);
	return dst;
}

//返回1可能为角点，返回0一定不是角点
int compare(uchar a, uchar b)
{
	if (a + threshold < b)
		return 1;
	else if (a - threshold >b)
		return 1;
	else
		return 0;
}

//Features from  Accelerated Segment Test
void FAST(Mat &src,Mat &area)
{
	int row = src.rows;
	int col = src.cols;
	//原图片扩展0，否则后面循环矩阵越界
	copyMakeBorder(src, src, 3, 3, 3, 3, BorderTypes::BORDER_CONSTANT);//可能最后一个参数需要考虑一下
	 
	int test[16];
	int result[16];

	for (int i = 3; i < row + 3; i++)
	{
		for (int j = 3; j < col + 3; j++)
		{	
			int sum = 0;
			//先检测1,5,9,13位置的像素点
			test[0] = src.at<uchar>(i, j - 3);
			test[4] = src.at<uchar>(i + 3, j);
			test[8] = src.at<uchar>(i, j + 3);
			test[12] = src.at<uchar>(i - 3, j);
			result[0] = compare(src.at<uchar>(i, j), test[0]);
			result[1] = compare(src.at<uchar>(i, j), test[4]);
			result[2] = compare(src.at<uchar>(i, j), test[8]);
			result[3] = compare(src.at<uchar>(i, j), test[12]);

			//判断是否是角点
			sum = result[0] + result[1] + result[2] + result[3];
			if (sum < 3)
				area.at<uchar>(i - 3, j - 3) = 0;
			else
			{
				test[1] = src.at<uchar>(i + 1, j - 3);
				test[2] = src.at<uchar>(i + 2, j - 2);
				test[3] = src.at<uchar>(i + 3, j - 1);
				test[5] = src.at<uchar>(i + 3, j + 1);
				test[6] = src.at<uchar>(i + 2, j + 2);
				test[7] = src.at<uchar>(i + 1, j + 3);
				test[9] = src.at<uchar>(i - 1, j + 3);
				test[10] = src.at<uchar>(i - 2, j + 2);
				test[11] = src.at<uchar>(i - 3, j + 1);
				test[13] = src.at<uchar>(i - 3, j - 1);
				test[14] = src.at<uchar>(i - 2, j - 2);
				test[15] = src.at<uchar>(i - 1, j - 3);
				for(int k = 0;k < 16;k++)
					result[k] = compare(src.at<uchar>(i, j), test[k]);

				//进一步判断是否连续12个角点的灰度值都大于当前点加阈值或小于当前点减阈值
				bool flag = false;
				int count = 0;
				sum = 0;
				for (int k = 0; k < 15; k++)
					sum += result[k];
				if (sum < 12)//不存在12个1
					area.at<uchar>(i - 3, j - 3) = 0;
				else
				{
					//应该是个环
					count = 0;
					for (int k = 0; k < 15; k++)
					{
						for (int step = 0; step < 12; step++)
						{
							if (result[(k + step) % 16 == 1])
								count++;
						}
						if (count >= 12)
							flag = true;
					}
					if (!flag)
						area.at<uchar>(i - 3, j - 3) = 0;
				}
			}
		}
	}
}

int main()
{
	Mat src1 = imread("1.jpg");
	Mat src2 = imread("2.jpg");
	//判断图像是否加载成功
	if (src1.data && src2.data)
		cout << "图像加载成功!" << endl;
	else
	{
		cout << "图像加载失败!" << endl;
		system("pause");
		return -1;
	}
	Mat src1_gray,src1_low;

	//灰度处理，高斯滤波
	cvtColor(src1, src1_gray, COLOR_BGR2GRAY);
	src1_low = GaussianFilter(src1_gray);

	Mat area1 = Mat::ones(src1_low.rows, src1_low.cols, CV_8UC1);
	area1 += Scalar::all(254);
	FAST(src1_low, area1);

	Mat dst1 = src1.clone();
	for (int i = 0; i < dst1.rows; i++)
	{
		for (int j = 0; j < dst1.cols; j++)
		{
			if (area1.at<uchar>(i, j) == 255)
			{
				dst1.at<Vec3b>(i, j) = 0;
				dst1.at<Vec3b>(i, j)[2] = 255;
			}
		}
	}
	imshow("dst1", dst1);

	waitKey(0);
	return 0;
}