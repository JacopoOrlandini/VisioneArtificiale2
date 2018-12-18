//OpneCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"

//std:
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <random>
#include <iterator>

//FUNZIONI
void constantStreching(cv::Mat& image);
float getValue(cv::Mat&input, int index);

struct ArgumentList {
	std::string image_name;		    //!< image file name
};



bool ParseInputs(ArgumentList& args, int argc, char **argv) {

	if(argc<3 || (argc==2 && std::string(argv[1]) == "--help") || (argc==2 && std::string(argv[1]) == "-h") || (argc==2 && std::string(argv[1]) == "-help"))
	{
		std::cout<<"usage: simple -i <image_name>"<<std::endl;
		std::cout<<"exit:  type q"<<std::endl<<std::endl;
		std::cout<<"Allowed options:"<<std::endl<<
				"   -h	                     produce help message"<<std::endl<<
				"   -i arg                   image name. Use %0xd format for multiple images."<<std::endl;
		return false;
	}

	int i = 1;
	while(i<argc)
	{
		if(std::string(argv[i]) == "-i") {
			args.image_name = std::string(argv[++i]);
		}
		++i;
	}

	return true;
}

template <class T>
T bilinearAt(const cv::Mat& image, float r, float c)
{
    float s = c - std::floor(c);
    float t = r - std::floor(r);

    int floor_r = (int) std::floor(r);
    int floor_c = (int) std::floor(c);

    int ceil_r = (int) std::ceil(r);
    int ceil_c = (int) std::ceil(c);

    T value = (1 - s) * (1 - t) * image.at<T>(floor_r, floor_c)
                + s * (1 - t) * image.at<T>(floor_r, ceil_c)
                + (1 - s) * t * image.at<T>(ceil_r, floor_c)
                + s * t * image.at<T>(ceil_r, ceil_c);

    return value;
}

void harrisCornerDetector(const cv::Mat image, std::vector<cv::KeyPoint> & keypoints0, float alpha, float harrisTh)
{

//Parametri Sobel
int scale = 1;
int delta = 0;
int ddepth = CV_32F;

//Derivativi x e y
cv::Mat grad_x, grad_y;

//Gaussian image
cv::Mat gausImg;

//Moltiplicazione per mat
cv::Mat Ixx, Iyy, Ixy;


//Applicazione filtri derivativi
Sobel( image, grad_x, ddepth, 1, 0, 1, scale, delta, cv::BORDER_DEFAULT );
Sobel( image, grad_y, ddepth, 0, 1, 1, scale, delta, cv::BORDER_DEFAULT );

//Adattamento del livello di colore per tipo CV_32F
constantStreching(grad_x);
constantStreching(grad_y);

//Creazione Matrici prodotto di derivativi
cv::multiply(grad_x,grad_x,Ixx);
cv::multiply(grad_y,grad_y,Iyy);
cv::multiply(grad_x,grad_y,Ixy);

//Applicazione di filtro gaussiano di smoothing
cv::GaussianBlur( Ixx, Ixx, cv::Size(3,3), 0);
cv::GaussianBlur( Iyy, Iyy, cv::Size(3,3), 0);
cv::GaussianBlur( Ixy, Ixy, cv::Size(3,3), 0);
cv::imshow("grad:x",grad_x);
cv::imshow("Ixx",Ixx);
cv::imshow("Iyy",Iyy);
cv::imshow("Ixy",Ixy);

//Non si usa M ma calcolo con la formula semplificata.
cv::Mat M , row1, row2;
cv::hconcat(Ixx, Ixy, row1);
cv::hconcat(Ixy, Iyy, row2);
cv::vconcat(row1,row2,M);
cv::imshow("M", M);

//Calcolo di theta
cv::Mat theta;
cv::Mat par1,par2,tmppar3, par3;

cv::multiply(Ixx,Iyy,par1);
cv::multiply(Ixy,Ixy,par2);
cv::Mat sumPar3 = Ixx + Iyy;
cv::multiply(alpha,Ixx + Iyy,tmppar3);
cv::multiply(tmppar3,Ixx + Iyy,par3);

theta = par1 - par2 - par3;

//Applico un adattamento per utilita.
constantStreching(theta);

//Visualizzazione di theta
cv::Mat adjMap;
cv::Mat falseColorsMap;
double minr,maxr;
cv::minMaxLoc(theta, &minr, &maxr);
cv::convertScaleAbs(theta, adjMap, 255 / (maxr-minr));
cv::applyColorMap(adjMap, falseColorsMap, cv::COLORMAP_RAINBOW);
cv::namedWindow("response1", cv::WINDOW_NORMAL);
cv::imshow("response1", falseColorsMap);


//Non-maximum suppression
	for (int r = 1; r < theta.rows-1; r++) {
		for (int c = 1; c < theta.cols-1; c++) {
			if (theta.at<float>(r,c) > harrisTh) {
				if(theta.at<float>(r,c) > theta.at<float>(r-1,c-1) &&
					theta.at<float>(r,c) > theta.at<float>(r-1,c) &&
					theta.at<float>(r,c) > theta.at<float>(r-1,c+1) &&
					theta.at<float>(r,c) > theta.at<float>(r,c-1) &&
					theta.at<float>(r,c) > theta.at<float>(r,c+1) &&
					theta.at<float>(r,c) > theta.at<float>(r+1,c-1) &&
					theta.at<float>(r,c) > theta.at<float>(r+1,c) &&
					theta.at<float>(r,c) > theta.at<float>(r+1,c+1)){
					float diameter = 3 ;
					keypoints0.push_back(cv::KeyPoint((float)c, (float)r, diameter));
				}
			}
		}
	}
}



void findHomographyRansac(const std::vector<cv::Point2f> & points1, const std::vector<cv::Point2f> & points0, int N, float epsilon, int sample_size, cv::Mat & H, std::vector<cv::Point2f> & inliers_best0, std::vector<cv::Point2f> & inliers_best1)
{

	float normEuclidea = 0;
	int numBestInlier = 0;
	std::vector<cv::Point2f> sample0(sample_size); //inliers
	std::vector<cv::Point2f> sample1(sample_size); //inliers
	srand ((unsigned) time(NULL));

	for (int k = 0; k < N; k++) 
	{
		int tmpNum = 0;
		std::vector<cv::Point2f> v0;	//vettori di appoggio per inliers_best.
		std::vector<cv::Point2f> v1;

		for (int i = 0; i < sample_size; i++)
		{
			 int randomChoose = rand() % points0.size();
			 sample0[i] = points0[randomChoose];
			 sample1[i] = points1[randomChoose];
		}

		//Modello di omografia approssimato a 4 punti random del mio set
		H = cv::findHomography(cv::Mat(sample1), cv::Mat(sample0), 0);


		//calcolo la versione omogenea e la versione euclidea del punto calcolato con omografia.
		for (unsigned int j = 0; j < points0.size(); j++)
		{
			//attenzione che findHomography restituisce un tipo DOUBLE.
			double omogX = H.at<double>(0,0)*points1[j].x + H.at<double>(0,1)*points1[j].y + H.at<double>(0,2);
			double omogY = H.at<double>(1,0)*points1[j].x + H.at<double>(1,1)*points1[j].y + H.at<double>(1,2);
			double omogZ = H.at<double>(2,0)*points1[j].x + H.at<double>(2,1)*points1[j].y + H.at<double>(2,2);
			double euclX = omogX / omogZ;
			double euclY = omogY / omogZ;
			cv::Point2f omographPoint(euclX,euclY);

			//Norma euclidea e discriminazione sull'errore
			normEuclidea = sqrt(pow( (points0[j].x - omographPoint.x ) , 2) + pow( (points0[j].y - omographPoint.y ) , 2) );
			if ( normEuclidea < epsilon )
			{
				tmpNum++;
				v0.push_back(points0[j]);
				v1.push_back(points1[j]);
			 }
			}
			if (tmpNum >= numBestInlier)
			{
				numBestInlier = tmpNum;
				inliers_best0.clear();
			 	inliers_best0 = v0;
				inliers_best1.clear();
				inliers_best1 = v1;
			}
		}
	//Ottenuto il migliore set di keypoint che soddisfano la minimizzazione dell'errore calcolo l'omografia definitiva.
	H = cv::findHomography(cv::Mat(inliers_best1), cv::Mat(inliers_best0), 0);

}

int main(int argc, char **argv)
{
	int frame_number = 0;
	char frame_name[256];
	bool exit_loop = false;

	//vettore delle immagini di input
	std::vector<cv::Mat> imageRGB_v;
	//vettore delle immagini di input grey scale
	std::vector<cv::Mat> image_v;

	std::cout<<"Simple image stitching program."<<std::endl;

	//////////////////////
	//parse argument list:
	//////////////////////
	ArgumentList args;
	if(!ParseInputs(args, argc, argv)) {
		return 1;
	}

	while(!exit_loop)
	{

		if(args.image_name.find('%') != std::string::npos)
			sprintf(frame_name,(const char*)(args.image_name.c_str()),frame_number);

		cv::Mat im = cv::imread(frame_name);
		if(im.empty())
		{
			break;
		}

		//opening file
		std::cout<<"Opening "<<frame_name<<std::endl;

		//save RGB image
		imageRGB_v.push_back(im);

		//save grey scale image for processing
		cv::Mat im_grey(im.rows, im.cols, CV_8UC1);
		cvtColor(im, im_grey, CV_RGB2GRAY);
		image_v.push_back(im_grey);

		frame_number++;
	}

	if(image_v.size()<2)
	{
		std::cout<<"At least 2 images are required. Exiting."<<std::endl;
		return 1;
	}

	int image_width = image_v[0].cols;
	int image_height = image_v[0].rows;

	////////////////////////////////////////////////////////
	/// HARRIS CORNER
	//
	float alpha = 0.05;

	float harrisTh = 0.37;    //da impostare in base alla propria implementazione

	std::vector<cv::KeyPoint> keypoints0, keypoints1;

	harrisCornerDetector(image_v[0], keypoints0, alpha, harrisTh);
	harrisCornerDetector(image_v[1], keypoints1, alpha, harrisTh);
	////////////////////////////////////////////////////////


	////////////////////////////////////////////////////////
    /// CALCOLO DESCRITTORI E MATCHES
	//
    int briThreshl=30;
    int briOctaves = 3;
    int briPatternScales = 1.0;
	cv::Mat descriptors0, descriptors1;

	//dichiariamo un estrattore di features di tipo BRISK
    cv::Ptr<cv::DescriptorExtractor> extractor = cv::BRISK::create(briThreshl, briOctaves, briPatternScales);
    //calcoliamo il descrittore di ogni keypoint
    extractor->compute(image_v[0], keypoints0, descriptors0);
    extractor->compute(image_v[1], keypoints1, descriptors1);

    //associamo i descrittori tra me due immagini
    std::vector<std::vector<cv::DMatch> > matches;
	cv::BFMatcher matcher = cv::BFMatcher(cv::NORM_HAMMING);
	matcher.radiusMatch(descriptors0, descriptors1, matches, image_v[0].cols*0.2);

    std::vector<cv::Point2f> points[2];
    for(unsigned int i=0; i<matches.size(); ++i)
      {
        if(!matches.at(i).empty())
          {
                points[0].push_back(keypoints0.at(matches.at(i).at(0).queryIdx).pt);
                points[1].push_back(keypoints1.at(matches.at(i).at(0).trainIdx).pt);
          }
      }
	////////////////////////////////////////////////////////

	////////////////////////////////////////////////////////
    // CALCOLO OMOGRAFIA
    //
    //
    // E' obbligatorio implementare RANSAC.
    //
    // Per testare i corner di Harris inizialmente potete utilizzare findHomography di opencv, che include gia' RANSAC
    //
    // Una volta che avete verificato che i corner funzionano, passate alla vostra implementazione di RANSAC
    //
    //
    cv::Mat H;            //omografia finale
		std::vector<cv::Point2f> inliers_best[2]; //inliers

		if(points[1].size()>=4)
    {
    	int N=200;            //numero di iterazioni di RANSAC
    	float epsilon = 0.5;  //distanza per il calcolo degli inliers
    	int sample_size = 4;  //dimensione del sample

    	findHomographyRansac(points[1], points[0], N, epsilon, sample_size, H, inliers_best[0], inliers_best[1]);

    	std::cout<<std::endl<<"Risultati Ransac: "<<std::endl;
    	std::cout<<"Num inliers / match totali "<<inliers_best[0].size()<<" / "<<points[0].size()<<std::endl;

    	//H = cv::findHomography( cv::Mat(points[1]), cv::Mat(points[0]), CV_RANSAC );
		}

    else
    {
    	std::cout<<"Non abbastanza matches per calcolare H!"<<std::endl;
    	H = (cv::Mat_<double>(3, 3 )<< 1.0, 0.0, 0.0,
    		                           0.0, 1.0, 0.0,
			                           0.0, 0.0, 1.0);
    }

    std::cout<<"H"<<std::endl<<H<<std::endl;
    cv::Mat H_inv = H.inv();///H.at<double>(2,2);
    std::cout<<"H_inverse "<<std::endl<<H_inv<<std::endl;
	////////////////////////////////////////////////////////


	////////////////////////////////////////////////////////
    /// CALCOLO DELLA DIMENSIONE DELL'IMMAGINE FINALE
    //
    cv::Mat p = (cv::Mat_<double>(3, 1) << 0, 0, 1);
    cv::Mat tl = H*p;
    tl/=tl.at<double>(2,0);
    p = (cv::Mat_<double>(3, 1) << image_width-1, image_height-1, 1);
    cv::Mat br = H*p;
    br/=br.at<double>(2,0);
    p = (cv::Mat_<double>(3, 1) << 0, image_height-1, 1);
    cv::Mat bl = H*p;
    bl/=bl.at<double>(2,0);
    p = (cv::Mat_<double>(3, 1) << image_width-1, 0, 1);
    cv::Mat tr = H*p;
    tr/=tr.at<double>(2,0);

    int min_warped_r = std::min(std::min(tl.at<double>(1,0), bl.at<double>(1,0)),std::min(tr.at<double>(1,0), br.at<double>(1,0)));
    int min_warped_c = std::min(std::min(tl.at<double>(0,0), bl.at<double>(0,0)),std::min(tr.at<double>(0,0), br.at<double>(0,0)));

    int max_warped_r = std::max(std::max(tl.at<double>(1,0), bl.at<double>(1,0)),std::max(tr.at<double>(1,0), br.at<double>(1,0)));
    int max_warped_c = std::max(std::max(tl.at<double>(0,0), bl.at<double>(0,0)),std::max(tr.at<double>(0,0), br.at<double>(0,0)));

    int min_final_r = std::min(min_warped_r,0);
    int min_final_c = std::min(min_warped_c,0);

    int max_final_r = std::max(max_warped_r,image_height-1);
    int max_final_c = std::max(max_warped_c,image_width-1);

    int width_final = max_final_c-min_final_c+1;
    int height_final = max_final_r-min_final_r+1;

    std::cout<<"width_final "<<width_final<<" height_final "<<height_final<<std::endl;
	////////////////////////////////////////////////////////


	////////////////////////////////////////////////////////
    /// CALCOLO IMMAGINE FINALE
    //
    cv::Mat outwarp(height_final, width_final, CV_8UC3, cv::Scalar(0,0,0));

    //copio l'immagine 0 sul nuovo piano immagine, e' solo uno shift
    imageRGB_v[0].copyTo(outwarp(cv::Rect(std::max(0,-min_warped_c), std::max(0,-min_warped_r), image_width, image_height)));

    //copio l'immagine 1 nel piano finale
    //in questo caso uso la trasformazione prospettica
    for(int r=0;r<height_final;++r)
    {
        for(int c=0;c<width_final;++c)
        {
        	cv::Mat p = (cv::Mat_<double>(3, 1) << c+std::min(0,min_warped_c), r+std::min(0,min_warped_r), 1);
        	cv::Mat pi = H_inv*p;
        	pi/=pi.at<double>(2,0);

        	if(int(pi.at<double>(1,0))>=1 && int(pi.at<double>(1,0))<image_height-1 && int(pi.at<double>(0,0))>=1 && int(pi.at<double>(0,0))<image_width-1)
        	{
        		cv::Vec3b pick = bilinearAt<cv::Vec3b>(imageRGB_v[1], pi.at<double>(1,0), pi.at<double>(0,0));

        		//media
        		if(outwarp.at<cv::Vec3b>(r,c) != cv::Vec3b(0.0))
        			outwarp.at<cv::Vec3b>(r,c) =  (outwarp.at<cv::Vec3b>(r,c)*0.5 + pick*0.5);
        		else
        			outwarp.at<cv::Vec3b>(r,c) = pick;
        	}
        }
    }
	////////////////////////////////////////////////////////

	////////////////////////////
	//WINDOWS
	//
    for(unsigned int i = 0;i<keypoints0.size();++i)
    	cv::circle(imageRGB_v[0], cv::Point(keypoints0[i].pt.x , keypoints0[i].pt.y ), 5,  cv::Scalar(0), 2, 8, 0 );

    for(unsigned int i = 0;i<keypoints1.size();++i){
    	cv::circle(imageRGB_v[1], cv::Point(keypoints1[i].pt.x , keypoints1[i].pt.y ), 5,  cv::Scalar(0), 2, 8, 0 );
		}
	cv::namedWindow("KeyPoints0", cv::WINDOW_AUTOSIZE);
	cv::imshow("KeyPoints0", imageRGB_v[0]);

	cv::namedWindow("KeyPoints1", cv::WINDOW_AUTOSIZE);
	cv::imshow("KeyPoints1", imageRGB_v[1]);

    cv::Mat matchsOutput(image_height, image_width*2, CV_8UC3);
    imageRGB_v[0].copyTo(matchsOutput(cv::Rect(0, 0, image_width, image_height)));
    imageRGB_v[1].copyTo(matchsOutput(cv::Rect(image_width, 0, image_width, image_height)));
    for(unsigned int i=0; i<points[0].size(); ++i)
    {
    	cv::Point2f p2shift = points[1][i];
    	p2shift.x+=imageRGB_v[0].cols;
    	cv::circle(matchsOutput, points[0][i], 3, cv::Scalar(0,0,255));
    	cv::circle(matchsOutput, p2shift, 3, cv::Scalar(0,0,255));
    	cv::line(matchsOutput, points[0][i], p2shift, cv::Scalar(255,0,0));
    }
	cv::namedWindow("Matches", cv::WINDOW_NORMAL);
	cv::imshow("Matches", matchsOutput);

    cv::Mat matchsOutputIn(image_height, image_width*2, CV_8UC3);
    imageRGB_v[0].copyTo(matchsOutputIn(cv::Rect(0, 0, image_width, image_height)));
    imageRGB_v[1].copyTo(matchsOutputIn(cv::Rect(image_width, 0, image_width, image_height)));
    for(unsigned int i=0; i<inliers_best[0].size(); ++i)
    {
    	cv::Point2f p2shift = inliers_best[1][i];
    	p2shift.x+=image_width;
    	cv::circle(matchsOutputIn, inliers_best[0][i], 3, cv::Scalar(0,0,255));
    	cv::circle(matchsOutputIn, p2shift, 3, cv::Scalar(0,0,255));
    	cv::line(matchsOutputIn, inliers_best[0][i], p2shift, cv::Scalar(255,0,0));
    }
	cv::namedWindow("Matches Inliers", cv::WINDOW_NORMAL);
	cv::imshow("Matches Inliers", matchsOutputIn);

	cv::namedWindow("Outwarp", cv::WINDOW_AUTOSIZE);
	cv::imshow("Outwarp", outwarp);

	cv::waitKey(0);
	////////////////////////////

	return 0;
}




/*************	IMPLEMENTAZIONE FUNZIONI****************/


void constantStreching(cv::Mat&image){
	typedef unsigned char uchar;
		const uchar OUT_MIN  = 0;
		const uchar OUT_MAX  = 1;
		float min = 30, max = -30;


		for( int y = 0; y < image.rows; y++ )
		{
		   for( int x = 0; x < image.cols; x++ ){
					int index = x+y*image.cols;
					float intensity = (float) getValue(image,index);
		      min = min < intensity ? min : intensity;
		      max = max > intensity ? max : intensity;
		   }
		}

		for(int y = 0; y < image.rows; y++)
		{
		   for(int x = 0; x < image.cols; x++)
		   {
				 int index = x+y*image.cols;
				 float intensity = (float) getValue(image,index);
		     float r = OUT_MAX*( intensity  - min ) / (max - min + OUT_MIN) ;
				 if(image.type() == CV_8UC1){
					 uchar* out = (uchar* ) image.data;
					 out[index] = (int) r;
				 }
				 if(image.type() == CV_32FC1){
					 float* out = (float*) image.data;
					 out[index] = (float) r;
				 }
		   }
		}
}
float getValue(cv::Mat&input, int index){
	float res;
	if(input.type() == CV_8UC1 ){
		res = (float)input.data[index];
	}
	if(input.type() == CV_8UC3 ){
		res = (float)input.data[index];
	}
	if (input.type() == CV_32FC1) {
		float* out = (float*) input.data;
		res = (float) out[index];
	}
	return res;
}
