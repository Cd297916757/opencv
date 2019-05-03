#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

int compare(uchar a, uchar b)
{
	if (a > b)
		return 1;
	else
		return 0;
}

//Features from  Accelerated Segment Test
void FAST(Mat &src,Mat &flag)
{
	//flag默认全1，确定不是角点的置为0

	//原图片扩展0，否则后面循环矩阵越界
	copyMakeBorder(src, src, 3, 3, 3, 3, BorderTypes::BORDER_CONSTANT);//可能最后一个参数需要考虑一下
	int row = src.rows;
	int col = src.cols;

	uchar test[16];
	int result[16];
	int sum = 0;
	
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{	
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
			if (sum == 2)
				flag.at<uchar>(i, j) = 0;
			else//进一步判断是否连续n个角点的灰度值都大于或小于当前点
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
				for(int k = 0;k < 15;k++)
					result[k] = compare(src.at<uchar>(i, j), test[k]);
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


	return 0;
}