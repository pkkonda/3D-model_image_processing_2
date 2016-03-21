/* THIS PROJECT TAKES THE FRONT VIEW, TOP VIEW AND SIDE VIEW OF A SIMPLE OBJECT TO GENERATE ITS 3-D VIEW*/
/*"test_fv.jpg, test_sv.jpg, test_tv.jpg" ARE FRONT, SIDE AND TOP VIEWS RESPECTIVELY*/




#include "stdafx.h"
#include<iostream>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
//#include<opencv2/imgproc/imgproc.hpp>                            //INCLUDED FOR CANNY



using namespace std;
using namespace cv;
Mat binary(Mat img);     //THIS CONVERTS GRAYSCALE IMAGE TO BINARYY
int trans_i_sv(int i, int l);        //THIS FUNC GIVES TRANSFORM OF A POINT l DIST FROM MEAN LINE IN SV
int trans_j_sv(int j, int l);        //THIS FUNC GIVES TRANSFORM OF A POINT l DIST FROM MEAN LINE IN SV
Mat proj_fv_rbr(Mat img, int row);   //THIS FUNC TAKES INPUT AS FV OF IMG AND GIVES CORRESPONDING ROW'S PROJ IN 45 DEGREES
Mat proj_sv_rbr(Mat img,int row);    //THIS FUNC TAKES INPUT AS SV OF IMG AND GIVES CORRESPONDING ROW'S PROJ IN 45 DEGREES
Mat proj_fv_cbc(Mat img, int col);
Mat proj_tv_cbc(Mat img, int col);    //THIS TAKES INPUT AS TV IMAGE AND GIVES PROJ OF CORRESPONDING COLUMN 
Mat out_fvsv_union(Mat img_sv, Mat img_fv);   //THIS FUNC FINDS UNION OF SV AND FV PROJECTIONS
Mat make_white(Mat img);           //THIS FUNC TAKES AN IMG AND MAKES OUTPUTS COMPLETE WHITE IMG OF SAME DIMENSIONS.
int upper_limit_sv(Mat img);
int lower_limit_sv(Mat img);
int upper_limit_fv(Mat img);
int lower_limit_fv(Mat img);
int right_limit_tv(Mat img);
int left_limit_tv(Mat img);
Mat out_fvtv_union(Mat img_fv, Mat img_tv);

int main()
{
	Mat image_fv,image_sv,image_tv;
	image_fv = imread("E:\\test_fv.jpg");
	image_sv = imread("E:\\test_sv.jpg");
	image_tv = imread("E:\\test_tv.jpg");
	Mat img_bnry_fv_in(image_fv.rows, image_fv.cols, CV_8UC1, Scalar(255));     //BINARY OF INPUT IMAGES
	Mat img_bnry_sv_in(image_sv.rows, image_sv.cols, CV_8UC1, Scalar(255));
	Mat img_bnry_tv_in(image_tv.rows, image_tv.cols, CV_8UC1, Scalar(255));

	Mat img_sv_orth(image_sv.rows, image_sv.cols, CV_8UC1, Scalar(255));        //ORTHOGONAL VIEW OF SV &TV
	Mat img_tv_orth(image_tv.rows, image_tv.cols, CV_8UC1, Scalar(255));
  

	img_bnry_fv_in = binary(image_fv);            //THESE ARE  IMAGES IN BINARY
	img_bnry_sv_in = binary(image_sv);
	img_bnry_tv_in = binary(image_tv);

	Mat img_fv_proj_temp(image_fv.rows, image_fv.cols, CV_8UC1, Scalar(255));
	Mat img_sv_proj_temp(image_sv.rows, image_sv.cols, CV_8UC1, Scalar(255));

	//int g = upper_limit_sv(img_bnry_sv_in);
	//cout << g;
	//int f = lower_limit_sv(img_bnry_sv_in);
	//cout << f;
	Mat out_fvtv(image_fv.rows, image_fv.cols, CV_8UC1, Scalar(255));
	out_fvtv = out_fvtv_union(img_bnry_fv_in, img_bnry_tv_in);
	Mat out_fvsv(image_fv.rows, image_fv.cols, CV_8UC1, Scalar(255));
	out_fvsv = out_fvsv_union(img_bnry_sv_in, img_bnry_fv_in);

namedWindow("Frame", WINDOW_AUTOSIZE);
imshow("Frame", out_fvtv);
//imshow("Frame", proj_tv_cbc(img_bnry_tv_in, 350));
namedWindow("Frame2", WINDOW_AUTOSIZE);
//imshow("Frame", out);
imshow("Frame2", out_fvsv);
	waitKey(0);
}

Mat binary(Mat img)
{
	Mat img_bnry(img.rows, img.cols, CV_8UC1, Scalar(255));
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			if (0.33*(img.at<Vec3b>(i, j)[0] + img.at<Vec3b>(i, j)[1] + img.at<Vec3b>(i, j)[2])>128)
				img_bnry.at<uchar>(i, j) = 255;
			else
				img_bnry.at<uchar>(i, j) = 0;
		}
	}
	
	return img_bnry;
}

int trans_i_sv(int i,int l)
{
	int p = i - 0.707*l;
	return p;
}

int trans_j_sv(int j, int l)
{
	int q = j - 0.293*l;
	return q;
}

Mat proj_fv_rbr(Mat img, int row)     // PROJECTION OF A ROW FROM FV BY 45 DEGREES (BACKWARDS)
{
	Mat img_out_rbr(img.rows, img.cols, CV_8UC1, Scalar(255));
	
	int p = 0;

	
		for (int j = 0; j < img.cols; j++)
		{
			if (img.at<uchar>(row, j) == 0)     //CHECKING THE BLACK POINTS IN INPUT IMAGE
			{
				//img.at<uchar>(i, j) = 128;
				for (p = 0; ((row - 0.707*p) >= 1) && ((j + 0.707*p) <= img.cols - 1); p++)  //MID STATEMENT IS TO IGNORE ARRAY OUT OF BOUNDS
				{
					img_out_rbr.at<uchar>(row - 0.707*p, j + 0.707*p) = 128;   //TURNING CORRESPONDING ROW PROJ TO BRIGHTNESS 128.
				}


			}
		}
	

		return img_out_rbr;
}



Mat proj_sv_rbr(Mat img,int row)      // PROJECTION OF A ROW FROM SV BY 45 DEGREES (RIGHT SIDEWARDS)
{
	int a = 0;       // 'a' IS TO FIND LEFTMOST POINT OF INPUT SV IMG, SO THAT IMG CAN BE ROTATED AROUND THIS COLUMN 'a' TO GET SV
	Mat img_sv_out(img.rows, img.cols, CV_8UC1, Scalar(255));             //SIDE_VIEW OUTPUT
	for (int j = 0; j < img.cols; j++)
	{
		for (int i = 0; i < img.rows; i++)
		{
			if (img.at<uchar>(i, j) == 0)
			{
				a = j;
				break;
			}
		}
		if (a != 0)
			break;

	}

	//cout << a;

	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			int l = j - a;  // 'l' IS DIST OF ANY POINT FROM MEAN AXIS('a'), TO BE ROTATED
			if (img.at<uchar>(i, j) == 0)
			{
				if (((i - 0.707*l) >= 0) && ((j - 0.293*l) >= 0))
				{
					img_sv_out.at<uchar>((i - 0.707*l), (j - 0.293*l)) = 0;  // BLACKENING THE POINTS IN SV OUT IMG FOR ORTH PROJ IN SV
				}
			}

		}
	}

	int d;

	for (int p = 0; p < img.rows; p++)
	{
		for (int q = 0; q < img.cols; q++)
		{
			
			 if(img_sv_out.at<uchar>(p, q) == 0)   //ITERATING TO FIND A BLACK POINT IN ORTH PROJECTED SV
			{
				for (int r = 0; r < img.cols; r++)   //
				{
					 d = r - a;
					if (img.at<uchar>(row, r) == 0)  //ITERATING IN CORRESPONDING ROW OF INPUT IMG TO FIND BLACK POINT
					{
						if (p == trans_i_sv(row, d))
						{                               //CHECKING IF THIS IS THE POINT (p,q) IS ACTUALLY THE POINT CORRESPONDING TO ROW TO BE TRANSFORMED 
							if (q == trans_j_sv(r, d))
							{
								for (int m = q; m < img.cols; m++)
									img_sv_out.at<uchar>(p, m) = 100;   //MAKING THE COMPLETE CORRESPONDING ROW GRAY FROM RIGHT.
							}
						}
					}
				}
			}
		}
	}



	return img_sv_out;
}

Mat out_fvsv_union(Mat img_sv, Mat img_fv)
{
	Mat output(img_fv.rows, img_sv.cols, CV_8UC1, Scalar(255));
	Mat fv_temp_rbr(img_fv.rows, img_sv.cols, CV_8UC1, Scalar(255));
	Mat sv_temp_rbr(img_fv.rows, img_sv.cols, CV_8UC1, Scalar(255));    //TAKING TEMP IMGS TO SHOW ROW BY ROW PROJ'S, ONE ROW AFTER OTHER IN SAME IMG
	int p = upper_limit_fv(img_fv), q = lower_limit_fv(img_fv), r = upper_limit_sv(img_sv),s=lower_limit_sv(img_sv);
	
	for (int row = p; row < q; row++)
	{
		fv_temp_rbr = proj_fv_rbr(img_fv, row);   //LOADING A CORRESPONDING ROW'S PROJ INTO TEMP IMG FOR FINDING INTERSECTION
		sv_temp_rbr = proj_sv_rbr(img_sv, row);

		for (int i = r; i < s; i++)
		{
			for (int j = 0; j < img_fv.cols; j++)
			{
				if (fv_temp_rbr.at<uchar>(i, j) == 128)
				{
					if (sv_temp_rbr.at<uchar>(i, j) == 100)   //TAKING INTERSECTION FROM SV & FV PROJ RBR's
					{
						output.at<uchar>(i, j) = 80;
					}
				}
			}
		}

		make_white(fv_temp_rbr);    //MAKING TEMP IMGS WHITE TO BE USED FOR NEXT ITERATION (NEXT ROW)
		make_white(sv_temp_rbr);
	}


	

	


	return output;
}


Mat make_white(Mat img)
{
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			img.at<uchar>(i, j) = 255;
		}
	}
	return img;
}

int upper_limit_sv(Mat img)
{
	Mat img2 = proj_sv_rbr(img, 0);
	int flag = 0;
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			if (img2.at<uchar>(i, j) == 0)
			{
				flag=i;
				break;
			}

		}
		if (flag != 0)
		{
			break;
		}
	}

	return flag;
}

int lower_limit_sv(Mat img)
{
	Mat img2 = proj_sv_rbr(img, 0);
	int flag = 0;
	for (int i = img.rows-1; i >=0; i--)
	{
		for (int j = 0; j < img.cols; j++)
		{
			if (img2.at<uchar>(i, j) == 0)
			{
				flag = i;
				break;
			}

		}
		if (flag != 0)
		{
			break;
		}
	}

	return flag;
}

int lower_limit_fv(Mat img)
{
	//Mat img2 = proj_sv_rbr(img, 0);
	int flag = 0;
	for (int i = img.rows - 1; i >= 0; i--)
	{
		for (int j = 0; j < img.cols; j++)
		{
			if (img.at<uchar>(i, j) == 0)
			{
				flag = i;
				break;
			}

		}
		if (flag != 0)
		{
			break;
		}
	}

	return flag;
}

int upper_limit_fv(Mat img)
{
	//Mat img2 = proj_sv_rbr(img, 0);
	int flag = 0;
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			if (img.at<uchar>(i, j) == 0)
			{
				flag = i;
				break;
			}

		}
		if (flag != 0)
		{
			break;
		}
	}

	return flag;
}

Mat proj_tv_cbc(Mat img, int col)
{
	int a = 0;
	Mat out(img.rows, img.cols, CV_8UC1, Scalar(255));
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			{
				if (img.at<uchar>(i, j) == 0)
				{
					a = i;
					break;
				}
			}
		}
		if (a != 0)
			break;
	}

	int d;

	for (int p = 0; p < img.rows; p++)
	{
		for (int q = 0; q < img.cols; q++)
		{
			if (img.at<uchar>(p, q) == 0)
			{
				d = p - a;
				if (((p - 1.707*d) >= 0) && ((col + 0.707*d) < img.cols))
				{
					out.at<uchar>(p - 1.707*d, col + 0.707*d) = 0;
				}
			}

		}
	}
	int h;
	for (int n = 0; n < img.cols; n++)
	{
		for (int m = 0; m < img.rows; m++)
		{
			if (out.at<uchar>(m, n) == 0)
			{
				for (h = m; h < img.rows; h++)
				{
					out.at<uchar>(h, n) = 128;
				}
			}
		}
	}

	return out;
}

Mat proj_fv_cbc(Mat img, int col)
{
	Mat img_out_rbr(img.rows, img.cols, CV_8UC1, Scalar(255));

	int p = 0;


	for (int i = 0; i < img.rows; i++)
	{
		if (img.at<uchar>(i,col) == 0)     //CHECKING THE BLACK POINTS IN INPUT IMAGE
		{
			//img.at<uchar>(i, j) = 128;
			for (p = 0; ((i - 0.707*p) >= 1) && ((col + 0.707*p) <= img.cols - 1); p++)  //MID STATEMENT IS TO IGNORE ARRAY OUT OF BOUNDS
			{
				img_out_rbr.at<uchar>(i - 0.707*p, col + 0.707*p) = 128;   //TURNING CORRESPONDING ROW PROJ TO BRIGHTNESS 128.
			}


		}
	}


	return img_out_rbr;
}

int left_limit_tv(Mat img)
{
	int l = 0;
	for (int j = 0; j < img.cols; j++)
	{
		for (int i = 0; i < img.rows; i++)
		{
			if (img.at<uchar>(i, j) != 0)
			{
				l = j;
				break;
			}
		}
		if (l != 0)
		{
			break;
		}
	}
	return l;
}

int right_limit_tv(Mat img)
{
	int l = 0;
	for (int j = img.cols-1; j >=0; j--)
	{
		for (int i = 0; i < img.rows; i++)
		{
			if (img.at<uchar>(i, j) != 0)
			{
				l = j;
				break;
			}
		}
		if (l != 0)
		{
			break;
		}
	}
	return l;
}

Mat out_fvtv_union(Mat img_fv, Mat img_tv)
{
	Mat temp_fv_cbc(img_fv.rows, img_fv.cols, CV_8UC1, Scalar(255));
	Mat temp_tv_cbc(img_fv.rows, img_fv.cols, CV_8UC1, Scalar(255));
	Mat out(img_fv.rows, img_fv.cols, CV_8UC1, Scalar(255));

	int i, j;
	for (int col = 0; col < img_fv.cols; col++)
	{
		temp_fv_cbc = proj_fv_cbc(img_fv, col);
		temp_tv_cbc = proj_tv_cbc(img_tv, col);

		for (j = 0; j < img_fv.cols; j++)
		{
			for (int i = 0; i < img_fv.rows; i++)
			{
				if (temp_fv_cbc.at<uchar>(i, j) != 255 && temp_tv_cbc.at<uchar>(i, j) != 255)
				{
					out.at<uchar>(i, j) = 128;
				}

			}
		}
	}
		return out;
}