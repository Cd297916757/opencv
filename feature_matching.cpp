#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <random>
#include <vector>
#include <string>

using namespace std;
using namespace cv;

#define k_size 9//卷积核大小
#define sigma 2//sigma越大，平滑效果越明显
//#define Threshold 20//FAST的阈值
//=====BRIEF描述子部分=====
#define s 31//半径//论文
#define Time 512 //随机选取的点对
#define Min_match 150//最少多少个01码相等才认为匹配
#define Max 1000//一张图中最多取多少特征点
#define Min_distance 30//匹配的最小距离

const double PI = 4.0*atan(1.0);//π
double gaus[k_size][k_size];
int Threshold;//阈值

class point {
public:
	int x;
	int y;
	point(int x, int y)
	{
		this->x = x;
		this->y = y;
	}
	point()
	{
		this->x = 0;
		this->y = 0;
	}
};

class descriptor
{
public:
	point p;
	string code;
	bool matching;
	descriptor(int x, int y)
	{
		this->p = point(x, y);
		matching = false;
	}
	descriptor()
	{
		this->p = point();
		matching = false;
	}
};

class match
{
public:
	point p;
	point q;
	bool matching;
	float distance;
	match(int x, int y, int i, int j, float dis)
	{
		this->p = point(x, y);
		this->q = point(i, j);
		distance = dis;
	}
	match()
	{
		this->p = point();
		this->q = point();
		distance = 0;
	}
};

void GetGaussianKernel()
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
	GetGaussianKernel();//生成高斯卷积核矩阵
	Mat gaus_kern = Mat(k_size, k_size, CV_64F, gaus);
	filter2D(src, dst, src.depth(), gaus_kern);
	//namedWindow("gaussian_filter", WINDOW_AUTOSIZE);
	//imshow("gaussian_filter", dst);
	return dst;
}

//返回1和-1可能为角点，返回0一定不是角点
int compare(uchar a, uchar b)
{
	if (a + Threshold < b)
		return -1;
	else if (a > Threshold + b)
		return 1;
	else
		return 0;
}

//Features from  Accelerated Segment Test
void FAST(Mat src, Mat &area)
{
	int row = src.rows;
	int col = src.cols;
	//原图片复制扩展，否则后面循环矩阵越界
	copyMakeBorder(src, src, 3, 3, 3, 3, BorderTypes::BORDER_REFLECT_101);

	int test[16], result[16];
	KeyPoint my_keypoint;
	for (int i = 3; i < row + 3; i++)
	{
		for (int j = 3; j < col + 3; j++)
		{
			int sum = 0;
			//阈值一般为p点灰度值的20%
			Threshold = src.at<uchar>(i, j) *0.2;//自动类型转换
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
			if (sum >= -2 && sum <= 2)
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
				for (int k = 1; k < 16; k++)
					result[k] = compare(src.at<uchar>(i, j), test[k]);

				//进一步判断是否连续12个角点的灰度值都大于当前点加阈值或小于当前点减阈值
				bool flag = true;
				sum = 0;
				for (int k = 0; k < 16; k++)
					sum += result[k];
				if (sum < 12 && sum > -12)//不存在12个1或者-1
					area.at<uchar>(i - 3, j - 3) = 0;
				else
				{
					//起点
					for (int k = 0; k < 16; k++)
					{
						for (int step = 0; step < 12; step++)
						{
							if (result[(k + step) % 16] != result[k])
							{
								flag = false;
								break;
							}
						}
					}
					if (flag)
						area.at<uchar>(i - 3, j - 3) = 0;
				}
			}
		}
	}
}

double GaussRand()
{
	random_device rd;
	mt19937 gen(rd());
	//normal(0,1)中0为均值，1为方差
	//normal_distribution<double> normal(0, s*s / 25);
	normal_distribution<double> normal(0, s / 5);
	return round(normal(gen));
}

//p和q都符合(0,s*s/25)的高斯分布 特征点周围sxs的区域
void BRIEF(Mat src, descriptor &desc)
{
	//生成符合高斯分布的随机坐标
	point p[Time];// , q[Time];
	for (int i = 0; i < Time; i++)
	{
		p[i].x = GaussRand();
		p[i].y = GaussRand();
	}
	//for (int i = 0; i < Time; i++)
	//{
	//	q[i].x = GaussRand();
	//	q[i].y = GaussRand();
	//}
	int sum1 = 0, sum2 = 0;
	//原图片扩展0，否则后面循环矩阵越界
	copyMakeBorder(src, src, s, s, s, s, BorderTypes::BORDER_REFLECT_101);

	for (int i = 0; i < Time; i++)
	{
		//for (int x = -4; x <= 4; x++)
		//{
		//	for (int y = -4; y <= 4; y++)
		//	{
		//		sum1 += src.at<uchar>(s + desc.p.x + p[i].x + x, s + desc.p.y + p[i].y + y);
		//		sum2 += src.at<uchar>(s + desc.p.x + q[i].x + x, s + desc.p.y + q[i].y + y);
		//	}
		//}
		sum1 = src.at<uchar>(s + desc.p.x + p[i].x, s + desc.p.y + p[i].y);
		//sum2 = src.at<uchar>(s + desc.p.x, s + desc.p.y);
		desc.code += sum1;
		/*if (sum1 >= sum2)
			desc.code += '1';
		else
			desc.code += '0'*/;
	}
}

//生成特征描述子
void GetBRIFE(Mat src, Mat area, descriptor desc[Max])
{
	int temp = 0;
	for (int i = 0; i < area.rows; i++)
	{
		for (int j = 0; j < area.cols; j++)
		{
			if (area.at<uchar>(i, j) == 255)
			{
				desc[temp].p.x = i;
				desc[temp].p.y = j;
				BRIEF(src, desc[temp]);
				temp++;
			}
		}
	}
}

float getDistance(Point pointO, Point pointA)
{
	float distance;
	distance = powf((pointO.x - pointA.x), 2) + powf((pointO.y - pointA.y), 2);
	distance = sqrtf(distance);
	return distance;
}

//比较描述子的二进制码
void CmpDescriptor(Mat src, descriptor desc1[Max], descriptor desc2[Max], int num1, int num2, int offset)
{
	Point p0, p1;
	bool flag = true;
	int match_num, Max_match;
	float min_distance = 1000;
	match best_match[Max];
	int temp = 0,num = 0;
	for (int i = 0; i < num1; i++)
	{
		Max_match = Min_match;
		for (int j = 0; j < num2; j++)
		{
			//排除异常点
			if (desc1[i].p.x == 0 && desc1[i].p.y == 0)
				break;
			else if (desc2[j].p.x == 0 && desc2[j].p.y == 0)
				break;
			else if (desc2[j].matching)
				continue;
			match_num = 0;
			for (int k = 0; k < Time; k++)
			{
				//int threshold = src.at<uchar>(desc1[i].p.x, desc1[i].p.y) * 0.2;
				if (abs(desc1[i].code[k] - desc2[j].code[k]) < 10)
				//if (desc1[i].code[k] == desc2[j].code[k])
					match_num++;
			}
			//匹配个数超过阈值即可能匹配
			if (match_num > Max_match)
			{
				Max_match = match_num;
				num = j;
			}
		}
		if (desc2[num].p.x != 0 && desc2[num].p.y != 0 && (Max_match != Min_match))
		{
			best_match[temp].p = desc1[i].p;
			best_match[temp].q = desc2[num].p;
			desc2[num].matching = true;
			temp++;
		}
	}
	for (int x = 0; x < num2; x++)
	{
		if (desc2[x].matching)
			cout << "matching" << endl;
	}
	for (int k = 0; k < temp; k++)
	{
		if (best_match[k].q.x != 0 && best_match[k].q.y != 0)
		{
			//筛选
			p0.y = best_match[k].p.x;
			p0.x = best_match[k].p.y;
			p1.y = best_match[k].q.x;
			p1.x = best_match[k].q.y + offset;
			best_match[k].distance = getDistance(p0, p1);
			if (best_match[k].distance < min_distance)
				min_distance = best_match[k].distance;
		}
	}
	for (int k = 0; k < temp; k++)
	{
		//if (best_match[k].distance < 2 * min_distance && best_match[k].distance > 2 * Min_distance)
		//if (best_match[k].distance > 2 * Min_distance)
		//{
		p0.y = best_match[k].p.x;
		p0.x = best_match[k].p.y;
		p1.y = best_match[k].q.x;
		p1.x = best_match[k].q.y + offset;
		line(src, p0, p1, Scalar(0, 0, 255));
		//}
	}
}

int main()
{
	Mat src1 = imread("3.png");
	Mat src2 = imread("4.png");
	//判断图像是否加载成功
	if (src1.data && src2.data)
		cout << "图像加载成功!" << endl;
	else
	{
		cout << "图像加载失败!" << endl;
		system("pause");
		return -1;
	}

	Mat src1_gray, src1_low;
	Mat src2_gray, src2_low;
	//灰度处理，高斯滤波
	cvtColor(src1, src1_gray, COLOR_BGR2GRAY);
	cvtColor(src2, src2_gray, COLOR_BGR2GRAY);
	src1_low = GaussianFilter(src1_gray);
	src2_low = GaussianFilter(src2_gray);

	Mat area1 = Mat::ones(src1_low.rows, src1_low.cols, CV_8UC1);
	Mat area2 = Mat::ones(src2_low.rows, src2_low.cols, CV_8UC1);
	area1 += Scalar::all(254);
	area2 += Scalar::all(254);
	FAST(src1_low, area1);
	FAST(src2_low, area2);

	//绘制特征点
	Mat src1_fast = src1.clone();
	Mat src2_fast = src2.clone();
	int num1 = 0, num2 = 0;//特征点的数量
						   //使用vector存储keypoint
	vector <KeyPoint> keypoint_vector1, keypoint_vector2;
	KeyPoint my_keypoint;

	for (int i = 0; i < src1_fast.rows; i++)
	{
		for (int j = 0; j < src1_fast.cols; j++)
		{
			if (area1.at<uchar>(i, j) == 255)
			{
				my_keypoint = KeyPoint(j, i, 1);
				keypoint_vector1.push_back(my_keypoint);
				num1++;
			}
		}
	}
	drawKeypoints(src1, keypoint_vector1, src1_fast, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	imshow("src1特征点", src1_fast);
	//keypoint_vector.clear();
	for (int i = 0; i < src2_fast.rows; i++)
	{
		for (int j = 0; j < src2_fast.cols; j++)
		{
			if (area2.at<uchar>(i, j) == 255)
			{
				my_keypoint = KeyPoint(j, i, 1);
				keypoint_vector2.push_back(my_keypoint);
				num2++;
			}
		}
	}
	drawKeypoints(src2, keypoint_vector2, src2_fast, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	imshow("src2特征点", src2_fast);

	//将两张特征点图片拼接
	Mat dst(src1_fast.rows, src1_fast.cols + src2_fast.cols + 1, src1_fast.type());
	src1_fast.colRange(0, src1_fast.cols).copyTo(dst.colRange(0, src1_fast.cols));
	src2_fast.colRange(0, src2_fast.cols).copyTo(dst.colRange(src1_fast.cols + 1, dst.cols));

	descriptor desc1[Max], desc2[Max];
	GetBRIFE(src1_low, area1, desc1);
	GetBRIFE(src2_low, area2, desc2);

	CmpDescriptor(dst, desc1, desc2, num1, num2, src1.cols);
	imshow("特征点匹配", dst);

	waitKey(0);
	return 0;
}