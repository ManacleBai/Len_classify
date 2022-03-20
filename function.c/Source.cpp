#include "Header.h"
#include <core/mat.hpp>
#include <opencv2/opencv.hpp>
#include <math.h>




void MEMReflective(Mat img, stLenAlgorithm *LenAlgorithm)
{
	LenAlgorithm->LenClassify.IMGCOL = up_height;
	LenAlgorithm->LenClassify.IMGROW = up_width;
	//size_t DetectorEdge = 1;
	size_t tmpCol = 0;
	vector<uchar> tmp;
	vector<vector<uchar> > tmpEdgebuff;

	for (size_t nrow = 0; nrow < img.rows; nrow++)
	{
	
		for (size_t ncol = 0; ncol < img.cols; ncol++)
		{
			uchar val = (uchar)img.at<uchar>(nrow, ncol);
			if (val >= Reflective)
			{
				tmpCol = ncol;
				/*LenAlgorithm->IMGReflective.CurrentEdgeRow = nrow;
				LenAlgorithm->IMGReflective.CurrentEdgeCol = tmpCol;*/
				tmp.push_back(1);
			}
			else
			{
				tmp.push_back(0);
			}
		}
		tmpEdgebuff.push_back(tmp);
		tmp.clear();
	}

	LenAlgorithm->IMGReflective.feature = tmpEdgebuff; //帶改成copy
	DeExtraPixel(img, LenAlgorithm);
}

void DeExtraPixel(Mat img, stLenAlgorithm* LenAlgorithm)
{
	uchar tmpvalue = 0;
	uchar tmpPixel = 0;
	
	vector<vector<uchar> > tmp1 = LenAlgorithm->IMGReflective.feature;
	vector<vector<uchar> > tmp2 = LenAlgorithm->IMGReflective.feature;

	for (size_t nrow = 0; nrow < LenAlgorithm->LenClassify.IMGROW; nrow++)
	{
		for (size_t ncol = 0; ncol < LenAlgorithm->LenClassify.IMGCOL; ncol++)
		{
			tmpvalue = tmp1[nrow][ncol];
			int tmppp_left = nrow - LenAlgorithm->IMGReflective.CompenSize;
			int tmppp_up = ncol - LenAlgorithm->IMGReflective.CompenSize;
			int tmppp_right = nrow + LenAlgorithm->IMGReflective.CompenSize;
			int tmppp_down = ncol + LenAlgorithm->IMGReflective.CompenSize;

			if (tmppp_left < 0 || tmppp_up < 0 || tmppp_right > LenAlgorithm->LenClassify.IMGROW || tmppp_down > LenAlgorithm->LenClassify.IMGCOL)
			{
				continue;
			}

			if (tmpvalue == 1)
			{
				if (tmp1[nrow - 1][ncol] && tmp1[nrow][ncol - 1] && tmp1[nrow + 1][ncol] && tmp1[nrow][ncol + 1])// 上右下左
				{
					tmp2[nrow][ncol] = 1;
					//cout << "0";
				}
				else 
				{
					//tmpPixel = img.at<uchar>(nrow,ncol - LenAlgorithm->IMGReflective.CompenSize );

					if (tmp1[nrow - 1][ncol]) // upward
					{

						for (uchar i = 0; i < LenAlgorithm->IMGReflective.CompenSize; i++)
						{
							//img.at<uchar>(nrow - i, ncol) = tmpPixel;
							tmp2[nrow - i][ncol] = 1;
						}
					}
					if (tmp1[nrow + 1][ncol]) // downward
					{
						for (uchar i = 0; i < LenAlgorithm->IMGReflective.CompenSize; i++)
						{
							//img.at<uchar>(nrow + i, ncol) = tmpPixel;
							tmp2[nrow + i][ncol] = 1;
						}
					}
					if (tmp1[nrow][ncol - 1]) // leftdownward
					{
						for (uchar i = 0; i < LenAlgorithm->IMGReflective.CompenSize; i++)
						{
							//img.at<uchar>(nrow , ncol - i) = tmpPixel;
							tmp2[nrow ][ncol -i] = 1;
						}
					}
					if (tmp1[nrow][ncol + 1]) // rightward
					{
						for (uchar i = 0; i < LenAlgorithm->IMGReflective.CompenSize; i++)
						{
							//img.at<uchar>(nrow , ncol + i) = tmpPixel;
							tmp2[nrow][ncol + i] = 1;
						}
					}
				}

			}
		}
	}

	LenAlgorithm->IMGReflective.feature.assign(tmp2.begin(),tmp2.end());
}

void ProcessReflective(Mat img, stLenAlgorithm* LenAlgorithm)
{
	vector<vector<uchar> > tmp1 = LenAlgorithm->IMGReflective.feature;
	uchar tmpvalue = 0;
	Mat tmp ;
	img.copyTo(tmp);
	
	for (size_t nrow = 0; nrow < LenAlgorithm->LenClassify.IMGROW; nrow++)
	{
		for (size_t ncol = 0; ncol < LenAlgorithm->LenClassify.IMGCOL; ncol++)
		{
			tmpvalue = tmp1[nrow][ncol];
			if (tmpvalue)
			{
				tmp.at<uchar>(nrow, ncol) = 255;
			}
			else
			{
				tmp.at<uchar>(nrow, ncol) = 0;
			}
			
		}
	}

	dilate(tmp,tmp, Mat::ones(14,14, CV_8U));
	//SHOW("pro", tmp);
	
	

	for (size_t nrow = 0; nrow < LenAlgorithm->LenClassify.IMGROW; nrow++)
	{
		for (size_t ncol = 0; ncol < LenAlgorithm->LenClassify.IMGCOL; ncol++)
		{
			tmpvalue = tmp.at<uchar>(nrow,ncol);
			//tmpvalue = 0;
			//tmpvalue = tmp1[nrow][ncol];
			if (tmpvalue)
			{
				img.at<uchar>(nrow, ncol) = (uchar)0;
			}
		}
	}
}


void SignContours(vector<vector<Point> > contours, stLenAlgorithm* LenAlgorithm)
{
	Rect targetArea;
	double tmpTargetArea = 0, tmpIndex = 0;
	vector<vector<Point> > contoursTmp;
	vector<Point> tmpvector;

	for (int i = 0; i < contours.size(); i++)
	{
		double a = contourArea(contours[i], false);
		if (a > tmpTargetArea)
		{
			tmpTargetArea = a;
			tmpIndex = i;
			targetArea = boundingRect(contours[i]); //帶修改
		}
	}

	tmpvector.assign(contours[tmpIndex].begin(), contours[tmpIndex].end());

	contoursTmp.push_back(tmpvector);

	LenAlgorithm->IMGReflective.targetAreas = targetArea;
	LenAlgorithm->IMGReflective.MaxTarget = contoursTmp;


}

void SmoothGraph(stLenAlgorithm* LenAlgorithm, uchar KpointSize)
{
	uint16_t Times = (uint16_t)LenAlgorithm->IMGReflective.MaxTarget[0].size();
	static uchar Downflag = 1;
	uint16_t TamX = 0, TamY = 0;

	vector<Point> pointer;
	pointer = LenAlgorithm->IMGReflective.MaxTarget[0];
	vector<Point>::iterator iter = pointer.begin();

	vector<Point> tmpvector;
	Point axis;

	if (Downflag)
	{
		LenAlgorithm->LenClassify.SmoothTarget.push_back(tmpvector);
	}
	

	for (uint16_t i = 0; i < Times; i++)
	{
		if (i < KpointSize)
		{
			for (uint16_t j = 0; j < (4+i); j++) 
			{
				TamX += (iter + j)->x;
				TamY += (iter + j)->y;
			}
			for (uint16_t k = (Times-1); k > (Times-4+i); k--)
			{
				TamX += (iter + k )->x;
				TamY += (iter + k )->y;
			}
		}
		else if (i < Times - KpointSize)
		{
			for (uint16_t j = i- KpointSize; j < (KpointSize * 2 + 1 + (i- KpointSize)); j++)
			{
				TamX += (iter + j)->x;
				TamY += (iter + j)->y;
			}
		}
		else
		{
			for (uint16_t j = i - KpointSize; j < Times; j++)
			{
				TamX += (iter + j)->x;
				TamY += (iter + j)->y;
			}

			for (uint16_t k = 0; k < (4 - (Times - i)); k++) 
			{
				//Point axis;
				TamX += (iter + k)->x;
				TamY += (iter + k)->y;
			}
		}

		TamX /= (KpointSize * 2 + 1);
		TamY /= (KpointSize * 2 + 1);
		axis.x = TamX;
		axis.y = TamY;
		TamX = 0;TamY = 0;
		tmpvector.push_back(axis);
	}
	 
	if (Downflag)
	{
		uint16_t m = 0;
		Downflag = 0;
		LenAlgorithm->IMGReflective.MaxTarget[0].resize(Times/ DownSampleSize);
		for (uint16_t k = 0; k < LenAlgorithm->IMGReflective.MaxTarget[0].size(); k++)
		{
			LenAlgorithm->IMGReflective.MaxTarget[0][k] = tmpvector[m];
			m = m + DownSampleSize;
			if (m >= Times) break;
		}
		SmoothGraph(LenAlgorithm, Kpoint);
	}
	else
	{
		LenAlgorithm->LenClassify.SmoothTarget[0].assign(tmpvector.begin(),tmpvector.end());
	}
}

void Classify(stLenAlgorithm* LenAlgorithm)
{
	vector<double> Slope;
	double TmpX1, TmpY1, TmpX2, TmpY2;
	double slope;
	
	vector<Point> pointer;
	pointer = LenAlgorithm->LenClassify.SmoothTarget[0];
	vector<Point>::iterator iter = pointer.begin();

	for (uint16_t i = 0; i < pointer.size(); i++) //順時針切線角度
	{
		if (i == pointer.size()-1)
		{
			TmpX1 = (iter + i)->x;
			TmpY1 = (iter + i)->y;
			TmpX2 = (iter + 0)->x;
			TmpY2 = (iter + 0)->y;
		}
		else
		{
			TmpX1 = (iter + i)->x;
			TmpY1 = (iter + i)->y;
			TmpX2 = (iter + i + 1)->x;
			TmpY2 = (iter + i + 1)->y;
		}
	
		slope = atan2(TmpY2 - TmpY1, TmpX2 - TmpX1);
		LenAlgorithm->LenClassify.slopearray.push_back(slope*180/pi);
	}
	AddFeature(LenAlgorithm, pointer);

}


void AddFeature(stLenAlgorithm* LAM, vector<Point> Graphy)
{
	double lineflag1; double lineflag2;
	uchar EndPoint = (uchar)LAM->LenClassify.slopearray.size() - 1;
	uchar StartPoint = 0;
	uchar IsFirstTime = 1;
	uchar tmp1 = 0;
	vector<double> slopArray;
	vector<Point> Cruve;
	vector<Point> Line;
	slopArray.assign(LAM->LenClassify.slopearray.begin(), LAM->LenClassify.slopearray.end());

	 for (uint16_t j = 0; j < LAM->LenClassify.slopearray.size(); j++)
	 {
		 if (StartPoint >= EndPoint) break;
		 lineflag1 = abs(slopArray[StartPoint] - slopArray[StartPoint + 1]);
		 lineflag2 = abs(slopArray[StartPoint] - slopArray[StartPoint +1 + 1]);

		if (lineflag1 < 2 && lineflag2 < 2) //if this session is Line
		{
			for (uint16_t i =0; i <= EndPoint + 1; i++)
			{
				switch (EndPoint - i) //Edge jugment
				{
				case 0:
					lineflag1 = 1;
					lineflag1 = 1;
					break;
				case 1:
					lineflag1 = abs(slopArray[StartPoint + i] - slopArray[StartPoint + 1 + i]);
					lineflag2 = 1;
					break;
				default:
					lineflag1 = abs(slopArray[StartPoint + i] - slopArray[StartPoint + 1 + i]);
					lineflag2 = abs(slopArray[StartPoint + i + 1] - slopArray[StartPoint + 1 + 1 + i]);
					break;
				}

				if (lineflag1 > 2 || lineflag2 > 2)
				{
					//CurrentPoint += i;
					StartPoint += i;
					break;
				}
				else
				{
					if (EndPoint - i == 0)
					{
						Line.push_back(Graphy[StartPoint + i]);
						StartPoint = i;
					}
					else
					{
						Line.push_back(Graphy[StartPoint + i]);
					}
				}
				
			}
			

			if (IsFirstTime) //backward to grow vecotor;
			{
				tmp1 = EndPoint;
				IsFirstTime = 0;
				for (uchar i = tmp1; i > StartPoint; i--)
				{
					lineflag1 = abs(slopArray[StartPoint] - slopArray[EndPoint - i]);
					lineflag2 = abs(slopArray[StartPoint] - slopArray[EndPoint - 1 - i]);
					if (lineflag1 > 2 || lineflag2 > 2)
					{
						EndPoint = i;
						break;
					}
					Line.insert(Line.begin(), Graphy[0]); 
				}
				LAM->LenClassify.Line.push_back(Line);
			}
			else
			{
				LAM->LenClassify.Line.push_back(Line);
			}
		}
		else //if this session is cruve
		{
			for (uint16_t i = 0; i <= EndPoint; i++)
			{

				switch (EndPoint - i)
				{
				case 0:
					lineflag1 = 3;
					lineflag1 = 3;
					break;
				case 1:
					lineflag1 = abs(slopArray[StartPoint + i] - slopArray[StartPoint + 1 + i]);
					lineflag2 = 3;
					break;
				default:
					lineflag1 = abs(slopArray[StartPoint + i] - slopArray[StartPoint + 1 + i]);
					lineflag2 = abs(slopArray[StartPoint + i + 1] - slopArray[StartPoint + 1 + 1 + i]);
					break;
				}


				if (lineflag1 < 2 && lineflag2 < 2)
				{
					//CurrentPoint += i;
					StartPoint += i ;
					break;
				}
				else
				{
					if (EndPoint - i == 0)
					{
						Cruve.push_back(Graphy[StartPoint + i]);
						StartPoint = i;
					}
					else
					{
						Cruve.push_back(Graphy[StartPoint + i]);
					}
				}
				//Cruve.push_back(Graphy[StartPoint + i]);
			}


			if (IsFirstTime) //backward to grow vecotor;
			{
				tmp1 = EndPoint;
				//IsFirstTime = 0;
				for (uchar i = tmp1; i > StartPoint; i--)
				{
					if (IsFirstTime)
					{
						lineflag1 = abs(slopArray[StartPoint] - slopArray[EndPoint]);
						lineflag2 = abs(slopArray[EndPoint] - slopArray[EndPoint -1]);
						IsFirstTime = 0;
					}
					else
					{
						lineflag1 = abs(slopArray[i] - slopArray[i - 1]);
						lineflag2 = abs(slopArray[i - 1] - slopArray[i - 2]);
					}

					if (lineflag1 <2 && lineflag2 < 2)
					{
						EndPoint = i;
						break;
					}
					Cruve.insert(Cruve.begin(), Graphy[0]);
				}
				LAM->LenClassify.Cruve.push_back(Cruve);
			}
			else
			{
				LAM->LenClassify.Cruve.push_back(Cruve);
			}
		}
	 }
}




void OneD_Hough(stLenAlgorithm * LAM, uint16_t r, uchar theta, uchar vote, uint16_t minLength, uchar jumpNumber)
{
	//transfar hough space
	float d = sqrt(pow(up_height, 2) + pow(up_width, 2));
	float drho = (2 * d) / r;
	vector<float> rhos = arange(d, (-1 * d), drho);
	float tmpi = pi / theta;

	vector<uint16_t> tmp(theta);
	vector<vector<uint16_t> >  accumulator(r, tmp);
	vector<Point> axis;
	axis.assign(LAM->LenClassify.SmoothTarget[0].begin(),LAM->LenClassify.SmoothTarget[0].end());
	vector<vector<uint16_t> >  table(axis.size(), tmp);


	uint16_t r_idx = 0;
	uint16_t tmx = 0;
	uint16_t tmy = 0;
	float dr = 0;
	for (size_t i = 0; i < axis.size(); i++)
	{
		tmx = axis[i].x;
		tmy = axis[i].y;
		for (size_t j = 0; j < theta; j++)
		{
			dr = tmx* cos(tmpi * j) + tmy * sin(tmpi * j);
			r_idx = argmin(absminus(rhos, dr));
			accumulator[r_idx][j]++;
			table[i][j] = r_idx;
		}
	}
	
	vector<Point> Maxindex = thresholdIndex(accumulator, vote);
	vector<Point> tmpLine;
	vector<uint16_t> tmprecordLine;
	vector<vector<uint16_t> > recordLine;
	vector<vector<Point> >realLine;
	//Hough space to x-y axis;
	for (uint16_t i = 0; i < Maxindex.size(); i++)
	{
		uint16_t goal = Maxindex[i].y;
		uint16_t locktheta = Maxindex[i].x;
		for (uint16_t j = 0; j < table.size(); j++)
		{
			if (goal == table[j][locktheta] )
			{
				Point tmpdot;
				tmpdot.x = axis[j].x;
				tmpdot.y = axis[j].y;
				tmprecordLine.push_back(j);
				tmpLine.push_back(tmpdot);
			}
		}
		recordLine.push_back(tmprecordLine);
		realLine.push_back(tmpLine);
		tmpLine.clear();
		tmprecordLine.clear();
	}

	//flag 1
	vector<vector<Point> >OutputLine;
	vector<vector<uint16_t> > OutputRecord;
	for (uchar i = 0; i < realLine.size(); i++)
	{
		if (Maxlength(realLine[i]) < minLength)
		{
			continue;
		}
		OutputRecord.push_back(recordLine[i]);
		OutputLine.push_back(realLine[i]);
	}
	realLine.clear();
	realLine = OutputLine;
	OutputLine.clear();
	tmpLine.clear();

	//flag 2==================
// max - (尾 + 頭)
	for (uchar i = 0; i < realLine.size(); i++)
	{
		uint16_t tmp = 0;
		uint16_t MaxJump = 0;
		for (uchar j = 0; j < OutputRecord[i].size() - 1; j++)
		{
			if (!OutputRecord[i][0] && OutputRecord[i][j + 1] == axis.size() - 1) continue;
			tmp = OutputRecord[i][j + 1] - OutputRecord[i][j];
			MaxJump = (tmp > MaxJump) ? tmp : MaxJump;
		}

		if (MaxJump < jumpNumber)
		{
			tmpLine.push_back(realLine[i][0]);
			tmpLine.push_back(realLine[i].back());
			OutputLine.push_back(tmpLine);
			tmpLine.clear();
		}
		MaxJump = 0;
	}
	//==============================================
	realLine.clear();
	realLine = OutputLine;
	OutputLine.clear();
	tmpLine.clear();
	if (realLine.size() > 0)
	{
		for (uchar i = 0; i < realLine.size(); i++)
		{
			vector<Point> tmp;
			tmp.push_back(realLine[i][0]);
			tmp.push_back(realLine[i][1]);
			LAM->LenClassify.Line.push_back(tmp);
		}
	}

	//flag 2==================
	 //max - (尾 + 頭)
	//vector<vector<Point> > Finally;
	//for (uchar i = 0; i < OutputLine.size(); i++)
	//{
	//	uint16_t tmp = 0;
	//	uint16_t MaxJump = 0;
	//	vector<Point> tmpOutput ;
	//	for (uchar j = 0; j < OutputRecord[i].size()-1; j++)
	//	{
	//		if(!OutputRecord[i][0] && OutputRecord[i][j + 1] == axis.size() - 1) continue;
	//		tmp = OutputRecord[i][j + 1] - OutputRecord[i][j];
	//		MaxJump = (tmp > MaxJump) ? tmp : MaxJump;
	//	}

	//	if (MaxJump < jumpNumber)
	//	{
	//		tmpOutput.push_back(OutputLine[i][0]);
	//		tmpOutput.push_back(OutputLine[i].back());
	//		Finally.push_back(tmpOutput);
	//	}
	//	MaxJump = 0;
	//}
	//==============================================

}



void ClassifyGraphy(stLenAlgorithm* LAM)
{
	uint16_t buff_index[50] = { 0 };

	//Mat Tmp = Mat::zeros(up_width, up_height, CV_8U);
	//for (uint16_t i = 0; i < LAM->LenClassify.SmoothTarget[0].size(); i++)
	//{
	//	uint16_t x = LAM->LenClassify.SmoothTarget[0][i].x;
	//	uint16_t y = LAM->LenClassify.SmoothTarget[0][i].y;
	//	Tmp.at<uchar>(y, x) = 255;
	//}
	//vector<Vec4f> lines;
	//HoughLinesP(Tmp, lines, 1, CV_PI / 180, 15, 60, 70); //15,80,70
	//if (lines.size() > 0)
	//{
	//	for (uchar i = 0; i < lines.size(); i++)
	//	{
	//		Vec4f hline = lines[i];
	//		vector<Point> tmp;
	//		tmp.push_back(Point(hline[0], hline[1]));
	//		tmp.push_back(Point(hline[2], hline[3]));
	//		LAM->LenClassify.Line.push_back(tmp);
	//	}
	//}

	vector<Point> pointer;
	pointer = LAM->LenClassify.SmoothTarget[0];
	vector<Point>::iterator iter = pointer.begin();
	uchar count = 0;
	for (uchar i = 0; i < LAM->LenClassify.Line.size(); i++)
	{
		int16_t tmpIndex_first = -1;
		int16_t tmpIndex_second = -1;

		for (uint16_t j = 0; j < LAM->LenClassify.SmoothTarget[0].size(); j++)
		{
			if ((iter + j)->x == LAM->LenClassify.Line[i][0].x && (iter + j)->y == LAM->LenClassify.Line[i][0].y)
			{
				tmpIndex_first = j;
			}
			if ((iter + j)->x == LAM->LenClassify.Line[i][1].x && (iter + j)->y == LAM->LenClassify.Line[i][1].y)
			{
				tmpIndex_second = j;
			}

			if (tmpIndex_first >= 0 && tmpIndex_second >= 0)
			{
				break;
			}
		}
		if (tmpIndex_first > tmpIndex_second) //swap
		{
			uint16_t tmp = tmpIndex_first;
			tmpIndex_first = tmpIndex_second;
			tmpIndex_second = tmp;
		}
		buff_index[count++] = tmpIndex_first;
		buff_index[count++] = tmpIndex_second;
		
	}
	count = 0;
	for (uchar i = 0; i < LAM->LenClassify.Line.size(); i++)
	{
		vector<Point> CruveTmp;
		bool phase;
		uchar flag = 0;
		uchar index_back = (i + 1) * 2;
		index_back = (index_back == LAM->LenClassify.Line.size() * 2) ? 0 : index_back;
		int16_t thrould = (int16_t) (buff_index[i*2 + 1] - buff_index[index_back]);
		if (thrould > 100)
		{
			for (uint16_t j = buff_index[i * 2 + 1]; j < LAM->LenClassify.SmoothTarget[0].size(); j++)
			{
				Point tmpaxis = *(iter + j);
				CruveTmp.push_back(tmpaxis);
			}

			for (uint16_t j = 0; j < buff_index[index_back]+1; j++)
			{
				Point tmpaxis = *(iter + j);
				CruveTmp.push_back(tmpaxis);
			}

			LAM->LenClassify.Cruve.push_back(CruveTmp);
		}
		else
		{
			phase = (buff_index[i * 2 + 1] > buff_index[index_back]);

			switch (phase)
			{
			case 0:
				for (uint16_t j = buff_index[i * 2 + 1]; j < buff_index[index_back] + 1; j++)
				{
					Point tmpaxis = *(iter + j);
					CruveTmp.push_back(tmpaxis);
				}
				LAM->LenClassify.Cruve.push_back(CruveTmp);
				break;
			case 1:
				for (uint16_t j = buff_index[i * 2 + 1]; j > buff_index[index_back] + 1; j--)
				{
					Point tmpaxis = *(iter + j);
					CruveTmp.push_back(tmpaxis);
				}
				LAM->LenClassify.Cruve.push_back(CruveTmp);
				break;
			default:
				break;
			}

		}

	}
}


void FitMatrix(vector<vector<float> > &OutputA,vector<vector<float> > &OutputY, vector<float> X, vector<float> Y, GraphPixel Type)
{
	vector<float> tmp1;
	uchar col = static_cast<int>(Type) + 2;

	switch (Type)
	{
	case Line:
	case Cruve:

		tmp1.resize((col), 0);
		OutputA.resize(X.size(), tmp1);
		tmp1.resize(1, 0);
		OutputY.resize(Y.size(), tmp1);
		for (uchar i = 0; i < X.size(); i++)
		{
			for (uchar j = 0; j < Type + 2; j++)
			{
				OutputA[i][j] = pow((X.at(i)), j);
			}
		}
		for (uchar i = 0; i < Y.size(); i++)
		{
			OutputY[i][0] = Y.at(i);
		}
		break;
	case Circle:
		uint8_t N = (uchar)X.size();
		float Mx2 = 0, My2 = 0;
		float Mx = 0, My = 0;
		float Mxy = 0;
		float Mxx2y2 = 0, Myx2y2 = 0, Mx2y2 = 0;
		tmp1.resize(3, 0);
		OutputA.resize(3, tmp1);
		tmp1.resize(1, 0);
		OutputY.resize(3, tmp1);
		//==========================================
		// calculotor X^2, XY, X, Y^2, Y, x(x^2+y^2),
		// y(X^2+Y^2), (X^2 + Y^2)
		// 
		// [ x2, xy, x] [A]   [ x(x2+y2)]
		// [ xy, y2, y] [B] = [ y(x2+y2)]
		// [ x,  y,  n] [C]   [ (x2+y2) ]
		//  n = Input.size;
		// 
		//==========================================
		for (uchar i = 0; i < N; i++)
		{
			Mx2 += (X[i] * X[i]);
			My2 += (Y[i] * Y[i]);
			Mx += (X[i]);
			My += (Y[i]);
			Mxy += (X[i] * Y[i]);
			Mx2y2 += (X[i] * X[i] + Y[i] * Y[i]);
			Mxx2y2 += (X[i]) * (X[i] * X[i] + Y[i] * Y[i]);
			Myx2y2 += (Y[i]) * (X[i] * X[i] + Y[i] * Y[i]);
		}
		OutputA[0][0] = Mx2;
		OutputA[0][1] = Mxy;
		OutputA[0][2] = Mx;
		OutputA[1][0] = Mxy;
		OutputA[1][1] = My2;
		OutputA[1][2] = My;
		OutputA[2][0] = Mx;
		OutputA[2][1] = My;
		OutputA[2][2] = N;

		OutputY[0][0] = Mxx2y2;
		OutputY[1][0] = Myx2y2;
		OutputY[2][0] = Mx2y2;
		break;
	}
}

void axistoInput(vector<float>  &X, vector<float>  &Y,vector<Point> axis)
{
	uchar N = axis.size();
	for (uchar i = 0; i < N; i++)
	{
		X.push_back((float)axis[i].x / 1000 );
		Y.push_back((float)axis[i].y / 1000 );
	}
}




void LS_Fit(stLenAlgorithm* LAM, vector<Point> axis, GraphPixel Type)
{
	GraphPixel Type1;
	vector <vector<float> > InputX;
	vector <vector<float> > InputY;
	vector<float>  Y;
	vector<float>  X;

	vector <vector<float> > multiple;
	vector <vector<float> > transpose;
	vector <vector<float> > inverse;

#ifdef fittingtest
	if (Type == Line)
	{
		 X = { 1, 2, 3 };
		 Y = { 1, 3, 4 };
	}
	else if(Type == Cruve)
	{

	}
	else
	{
		X = { 0,0.02,0.01,0.07 };
		Y = { 0.01,0.05,0.06,0.06 };
	}
#else
#endif



	axistoInput(X, Y, axis);
	FitMatrix(InputX, InputY, X, Y, Type); //Ax = Y; (A^t * A)^-1 * A^t * Y
	transposeMatrix(transpose, InputX);
	multipleMatrix(multiple, transpose, InputX);
	inverseMatrix(inverse, multiple);
	multipleMatrix(multiple, inverse, transpose);
	multipleMatrix(multiple, multiple, InputY);

	if (Type == Line) // y = ax +b; 
	{
		//LAM->LenClassify.Line
		float distance;
		LAM->cc.a = multiple[0][0];
		LAM->cc.b = multiple[1][0];
		distance = pow((axis.front().x - axis.back().x), 2) + pow((axis.front().y - axis.back().y), 2);
		LAM->Parameter.Length = pow(distance, 0.5);
		LAM->Parameter.LineStart = axis.front();
		LAM->Parameter.LineEnd = axis.back();

	/*	distance = pow((axis.front().x - axis.back().x), 2) + pow((Y.back() - Y.front()), 2);
		LAM->Parameter.Length = pow(distance, 0.5);*/

	}
	else if(Type == Cruve) // y = ax^2 + bx + c;
	{
		//======Curvature cal=========
		// y = ax2+bx+c;
		// y' = 2ax+b;
		// y'' = 2a;
		// K = |2a| / [1+(2ax+b)^2]^3/2
		// (2ax+b) = 0; K is maxValue;
		//============================
		LAM->cc.a = multiple[0][0];
		LAM->cc.b = multiple[1][0];
		LAM->cc.c = multiple[2][0];
		LAM->Parameter.Curvature = abs(2 * LAM->cc.a);
		LAM->Parameter.CruveStart = X.front();
		LAM->Parameter.CruveEnd = X.back();
	}
	// reference to URL (https://mangoroom.cn/opencv/least-square-circle-fit.html)
	else if (Type == Circle) 
	{
		LAM->cc.a = multiple[0][0] / 2 *1000 + 0.5;
		LAM->cc.b = multiple[1][0] / 2 *1000 + 0.5;
		LAM->cc.c = (multiple[0][0] * multiple[0][0])+( multiple[1][0] * multiple[1][0])+ (4*(multiple[2][0])) ;
		LAM->cc.c = (pow(LAM->cc.c, 0.5) / 2 *1000 ) + 0.5;
		LAM->Parameter.center.x = LAM->cc.a;
		LAM->Parameter.center.y = LAM->cc.b;
		LAM->Parameter.Radius = LAM->cc.c;
	}
}

void FindBoundary(stLenAlgorithm* LAM)
{
	vector<Point> axis;
	if (LAM->LenClassify.Line.empty())
	{
		axis.assign(LAM->LenClassify.Cruve[0].begin(), LAM->LenClassify.Cruve[0].end());
		LS_Fit(LAM, axis,Circle);
	}
	else
	{
		double tmpTargetArea = 0, tmpIndex = 0;
		vector<vector<Point> > contoursTmp;
		vector<Point> tmpvector;

		for (int i = 0; i < LAM->LenClassify.Line.size(); i++)
		{
			double a = contourArea(LAM->LenClassify.Line[i], false);
			if (a > tmpTargetArea)
			{
				tmpTargetArea = a;
				tmpIndex = i;
			}
		}
		axis.assign(LAM->LenClassify.Line[tmpIndex].begin(), LAM->LenClassify.Line[tmpIndex].end());
		LS_Fit(LAM, axis, Line);

		tmpIndex = 0; tmpTargetArea = 0;
		for (int i = 0; i < LAM->LenClassify.Cruve.size(); i++)
		{
			double a = contourArea(LAM->LenClassify.Cruve[i], false);
			if (a > tmpTargetArea)
			{
				tmpTargetArea = a;
				tmpIndex = i;
			}
		}
		axis.assign(LAM->LenClassify.Cruve[tmpIndex].begin(), LAM->LenClassify.Cruve[tmpIndex].end());
		LS_Fit(LAM, axis, Cruve);
	}
}

void Draw(Mat img, stLenAlgorithm* LAM)
{
	if (LAM->LenClassify.Line.empty())
	{
		circle(img, LAM->Parameter.center, LAM->cc.c, GREEN ,2);
		circle(img, LAM->Parameter.center, 10, RED , -1);
		String con1 = "Circle center : (x : %d, y : %a )";
		String x = to_string(LAM->Parameter.center.x);
		String y = to_string(LAM->Parameter.center.y);
		String Output = con1.replace(con1.find("%d"), x.size(),x);
		Output = Output.replace(Output.find("%a"), y.size(), y);
		putText(img, Output, Point(10, 30), FONT_HERSHEY_COMPLEX, 0.7, BLUE);
		String con2 = "Circle radius : ";
		String radius = to_string((uint16_t)(LAM->Parameter.Radius+0.5));
		con2 += radius;
		putText(img, con2,Point(10,70),FONT_HERSHEY_COMPLEX, 0.7, BLUE);

	}
	else
	{
		for (uchar j = 0; j < LAM->LenClassify.Line.size(); j++)
		{
			line(img, LAM->LenClassify.Line[j][0], LAM->LenClassify.Line[j][1], Scalar(255, 0, 0), 4);
		}
		//line(img, LAM->Parameter.LineStart, LAM->Parameter.LineEnd, Scalar(255, 0, 0),4);

	/*	double tmpTargetArea = 0, tmpIndex = 0;
		vector<vector<Point> > contoursTmp;
		vector<Point> tmpvector;*/

		//for (int i = 0; i < LAM->LenClassify.Cruve.size(); i++)
		//{
		//	double a = contourArea(LAM->LenClassify.Cruve[i], false);
		//	if (a > tmpTargetArea)
		//	{
		//		tmpTargetArea = a;
		//		tmpIndex = i;
		//	}
		//}
		//tmpvector.assign(LAM->LenClassify.Cruve[tmpIndex].begin(), LAM->LenClassify.Cruve[tmpIndex].end());
		//contoursTmp.push_back(tmpvector);

		for (uchar j = 0; j < LAM->LenClassify.Cruve.size(); j++)
		{
			for (uint16_t i = 0; i < LAM->LenClassify.Cruve[j].size() - 1; i++)
			{
				line(img, LAM->LenClassify.Cruve[j][i], LAM->LenClassify.Cruve[j][i + 1], Scalar(0, 255, 0), 4);
			}
		}

		String con1 = "Cruve curvature : ";
		String curvature = to_string((float)(LAM->Parameter.Curvature + 0.5));
		con1 += curvature;
		putText(img, con1, Point(10, 30), FONT_HERSHEY_COMPLEX, 0.7, BLUE);
		String con2 = "Max line length : ";
		String length = to_string((uint16_t)(LAM->Parameter.Length+0.5));
		con2 += length;
		putText(img, con2, Point(10, 70), FONT_HERSHEY_COMPLEX, 0.7, BLUE);
		//drawContours(img, contoursTmp, 0, Scalar(0, 255, 0), 2);
	}


}


void transposeMatrix(vector<vector<float> >& Output, vector<vector<float> > Input)
{
	Output.clear();
	vector<float> tmp1;
	tmp1.resize(Input.size(), 0);
	Output.resize(Input[0].size(), tmp1);

	for (uint8_t i = 0; i < Input.size(); i++)
	{
		for (uint16_t j = 0; j < Input[0].size(); j++)
		{
			Output[j][i] = Input[i][j];
		}
	}
}

void multipleMatrix(vector<vector<float> >& Output, vector<vector<float> > IutputA, vector<vector<float> > IutputB)
{
	// Output = A X B ;
	Output.clear();
	uint8_t A_row = (uint8_t)IutputA.size();
	uint8_t A_col = (uint8_t)IutputA[0].size();
	uint8_t B_row = (uint8_t)IutputB.size();
	uint8_t B_col = (uint8_t)IutputB[0].size();
	//double tmp;
	vector<float> tmp1;
	tmp1.resize(B_col, 0);
	Output.resize(A_row, tmp1);

	for (uint8_t i = 0; i < A_row; i++)
	{
		for (uint8_t j = 0; j < A_col; j++)
		{
			for (uint8_t k = 0; k < B_col; k++)
			{
				Output[i][k] += IutputA[i][j] * IutputB[j][k];
			}

		}

	}

}

float det(vector<vector<float> > Input)
{
	int i; float sum = 0;
	int h = 0, l = 0;
	uchar n = (uchar)Input.size();
	//float b[MAX][MAX];
	vector<float> tmp1;
	vector<vector< float> >Ntmp;
	tmp1.resize(Input.size()-1, 1);
	Ntmp.resize(Input.size()-1, tmp1);

	if (n == 1)
	{
		return Input[0][0];
	}
	else if (n == 2)
	{
		return (Input[0][0] * Input[1][1] - Input[0][1] * Input[1][0]);
	}
	else {

		for (i = 0; i < n; i++)
		{
			for (uchar j = 1; j < Input.size(); j++) //求分解式
			{
				for (uchar k = 0; k < Input.size(); k++)
				{
					if (l == k) continue;
					Ntmp[l][h] = Input[j][k];
					h++;
					if (h == Input.size() - 1)
					{
						h = 0;
						l++;
					}
				}
			}

			sum = (float)(sum + Input[0][i] * pow(-1, i) * det(Ntmp));	//  sum = det matrix
		}
	}
	return sum;
}

void inverseMatrix(vector<vector<float > > &Output, vector<vector<float> > Input)
{
	Output.clear();
	vector<float> tmp1;
	float tmp;
	uchar N = (uchar)Input.size();
	tmp1.resize(Input.size(), 0);
	Output.resize(Input.size(), tmp1);  

	for (uchar i = 0; i < Input.size(); i++)
	{
		Output[i][i] = 1;
	}

	/* 1 0 0 */
	/* 0 1 0 */
	/* 0 0 1 */ /* 高斯消去法 */

	int x,y, k;

	for (k = 0; k < N; k++)
	{
		if (Input[k][k] == 0)
		{
			for (x = k + 1; x < N; x++)  //判斷是否對角元素是否為0
			{
				for (y = 0; y < N; y++) //swap chang line
				{
					tmp = Input[k][y];
					Input[k][y] = Input[x][y];
					Input[x][y] = tmp;
					tmp = Output[k][y];
					Output[k][y] = Output[x][y];
					Output[x][y] = tmp;
				}

				if (Input[k][k] != 0)
					break;
			}

			if (Input[k][k] == 0)
			{
				// 無反矩陣
				return;
			}
		}


		tmp = 1 / Input[k][k];
		for (y = 0; y < N; y++)
		{
			Output[k][y] *= tmp;
			Input[k][y] *= tmp;
		}

		for (x = 0; x < N; x++)
		{
			if (x == k)
				continue;

			tmp = Input[x][k] / Input[k][k];
			for (y = 0; y < N; y++)
			{
				Input[x][y] -= tmp * Input[k][y];
				Output[x][y] -= tmp * Output[k][y];
			}
		}

		//temp = 1 / m[k][k];

		//for (y = 0; y < N; y++)
		//{
		//	m[k][y] *= temp;
		//	r[k][y] *= temp;
		//}

	}


}


