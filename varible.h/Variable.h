#pragma once
#include <vector>
#include <opencv2/opencv.hpp>

#define SHOW(Name, img) namedWindow(Name, WINDOW_AUTOSIZE); imshow(Name, img)
#define BLUE Scalar(255,0,0)
#define GREEN Scalar(0,255,0)
#define RED Scalar(0,0,255)

//pixel
#define up_width 800
#define up_height 800
#define Reflective 240 //190

//smooth
#define Kpoint 3
#define DownSampleSize 2

#define Reflective_low Scalar(20, 0, 200)
#define Reflective_up Scalar(100, 255, 255)

//#define test888
//#define fittingtest

typedef unsigned char uchar;
typedef unsigned short int  uint16_t;
using namespace std;
using namespace cv;

const double pi = 3.1415926;


template<typename T>
vector<T> arange(T start, T stop, T step = 1) {
	vector<T> values;
	for (T value = start; value > stop; value -= step)
		values.push_back(value);
	return values;
}

template<typename T>
vector<T> absminus(vector<T> A, T B) {
	vector<T> values;
	for (uint16_t i = 0; i < A.size(); i++)
	{
		values.push_back(abs(A[i] - B));
	}
	return values;
}

template<typename T>
size_t argmin(vector<T> A) {

	return distance(A.begin(),min_element(A.begin(), A.end()));
}

template<typename T>
vector<Point> thresholdIndex(vector<vector<T> > A, uchar td) {
	vector<Point> values;
	Point axis;
	for (uint16_t i = 0; i < A.size(); i++)
	{
		for (uint16_t j = 0; j < A[0].size(); j++)
		{
			if (A[i][j] > td)
			{
				axis.x = j;
				axis.y = i;
				values.push_back(axis);
			}
			
		}
	}
	return values;
}

template<typename T>
uint16_t length(T A, T B)
{
	return sqrt(pow((B.x - A.x), 2) + pow(B.y - A.y, 2)) + 0.5;
}


template<typename T>
uint16_t Maxlength(vector<T>  &A) {
	
	uint16_t  value = 0;
	Point  tmp;
	uint16_t tmp_f = 0;
	uint16_t tmp_b = 0;
	for (uint16_t i = 0; i < A.size()-1; i++)
	{
		for (uint16_t j = i; j < A.size()-1; j++)
		{
			if (value < length(A[i], A[j + 1]))
			{
				value = length(A[i], A[j + 1]);
				tmp_f = i;
				tmp_b = j + 1;
			}
		}
	}
	tmp = A[0];
	A[0] = A[tmp_f];
	A[tmp_f] = tmp;

	tmp = A.back();
	A[A.size()-1] = A[tmp_b];
	A[tmp_b] = tmp;

	return value;
}
const uchar lowpassfilter[3][3] =
{
	{2,2,2},
	{1,5,1},
	{1,1,1}
};

  enum   GraphPixel 
{
	Line = 0,
	Cruve,
	Circle
};

typedef struct {
	float a;
	float b;
	float c;
	float d;
}coefficient;

typedef struct {
	uchar CompenSize;
	Rect targetAreas;
	vector<vector<uchar> > feature;
	vector<vector<Point> > MaxTarget;
}stReflective;

typedef struct  {
	uint16_t IMGROW;
	uint16_t IMGCOL;
	vector<vector<Point> > SmoothTarget;
	vector<float> slopearray;
	vector<vector<Point> > Line;
	vector<vector<Point> > Cruve;
}stLenCalssify;

typedef struct {
	double Radius;
	Point center;
	double Curvature;
	uint16_t CruveStart;
	uint16_t CruveEnd;
	double Length;
	Point LineStart;
	Point LineEnd;

}stCalParameter;

typedef struct {
	stReflective IMGReflective;
	stLenCalssify LenClassify;
	stCalParameter Parameter;
	coefficient cc;
}stLenAlgorithm;