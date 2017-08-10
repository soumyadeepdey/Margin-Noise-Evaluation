
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include <sys/stat.h>
#include <iostream>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;

#include <fstream>
#include <iostream>
#include <iomanip>

using namespace std;

#include "folder.h"

float HammingfDist(Mat gt, Mat clr);

float NoiseRatio(Mat gt, Mat clr);

float Gtremoval(Mat gt, Mat clr);

void Evaluate(Mat gt, Mat clr);


float HammingfDist(Mat gt, Mat clr)
{
  int dist = 0;
  int num = 0; 
  for(int i=0;i<gt.rows;i++)
  {
    for(int j=0;j<gt.cols;j++)
    {
      if(gt.at<uchar>(i,j)!=255)
      {
	num++;
	if(gt.at<uchar>(i,j)==0 && clr.at<uchar>(i,j)!=0 )
	  dist++;
	if(gt.at<uchar>(i,j)!=0 && clr.at<uchar>(i,j)==0 )
	  dist++;
      }
    }
  }
  float HD = dist * 1.0;
  HD = HD/num;
  return HD;

}

float NoiseRatio(Mat gt, Mat clr)
{
  int npb = 0;
  int np = 0; 
  
  for(int i=0;i<gt.rows;i++)
  {
    for(int j=0;j<gt.cols;j++)
    {
      if(gt.at<uchar>(i,j)!=255)
      {
	if(gt.at<uchar>(i,j)==0 && clr.at<uchar>(i,j)==0)
	  np++;
	else if(clr.at<uchar>(i,j)!=255 && gt.at<uchar>(i,j)!=0)
	  npb++;
      }
    }
  }
  
  
  float nr = (npb*1.0)/(np*1.0);
  return nr;
}

float Gtremoval(Mat gt, Mat clr)
{
  int np = 0;
  int nc = 0;
  
  for(int i=0;i<gt.rows;i++)
  {
    for(int j=0;j<gt.cols;j++)
    {
      if(gt.at<uchar>(i,j)==0)
      {
	np++;
	if(clr.at<uchar>(i,j)==255)
	  nc++;
      }
    }
  }
  
  //float gtr = (np-nc)*1.0;
  //gtr=gtr/np;
  float gtr = nc * 1.0;
  gtr = gtr/(np*1.0);
  return gtr;
}

void Evaluate(Mat gt, Mat clr)
{
  
  float HD = HammingfDist(gt,clr);
  float NR = NoiseRatio(gt,clr);
  float GR = Gtremoval(gt,clr);
  
  int tp,fp,tn,fn;
  tp=0;fp=0;tn=0;fn=0;
  
  for(int i=0;i<gt.rows;i++)
  {
    for(int j=0;j<gt.cols;j++)
    {
      if(gt.at<uchar>(i,j)!=255)
      {
	if(gt.at<uchar>(i,j)==0 && clr.at<uchar>(i,j)==0)
	  tp++;
	if(gt.at<uchar>(i,j)==0 && clr.at<uchar>(i,j)==255)
	  fn++;
	if(gt.at<uchar>(i,j)!=0 && clr.at<uchar>(i,j)==0)
	  fp++;
	if(gt.at<uchar>(i,j)!=0 && clr.at<uchar>(i,j)==255)
	  tn++;
      }
    }
  }
  
  float precision = tp*1.0;
  if(precision>0.0)
    precision = precision/(tp+fp);
  float recall = tp*1.0;
  if(recall > 0.0)
    recall = recall/(tp+fn);
  float specificity = tn*1.0;
  if(specificity>0.0)
    specificity = specificity/(tn+fp);
  float accuracy = (tp + tn)*1.0;
  if(accuracy > 0.0)
    accuracy = accuracy/(tp+fp+fn+tn);
  
  float f1score;
  f1score = 2*precision*recall;
  if(f1score > 0.0 )
    f1score = f1score/(precision+recall);
  
  float negPredVal=tn*1.0;
  if(negPredVal>0.0)
    negPredVal=negPredVal/(tn+fn);
  
  float balacc = (recall+specificity)/2;
  
  float Fscore_spec_sen = 2*recall*specificity;
  if(Fscore_spec_sen > 0.0)
    Fscore_spec_sen = Fscore_spec_sen/(recall+specificity);
  
  float Fscore_neg = 2*negPredVal*specificity;
  if(Fscore_neg > 0.0)
   Fscore_neg = Fscore_neg/(negPredVal+specificity);
  
  double MCC;
  MCC = (tp*tn*1.0)-(fp*fn*1.0);
  if(MCC > 0.0)
  {
    double deno = (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn);
    deno = sqrt(deno);
    MCC = MCC/deno;
  }
  
  FILE *fp1;
  fp1 = fopen("MarginResult.xls","a");
  fprintf(fp1,"%f\t%f\t%f\t",HD,NR,GR);
  fprintf(fp1,"%d\t%d\t%d\t%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t",tp,fp,tn,fn,precision,recall,f1score, specificity,negPredVal, Fscore_spec_sen,accuracy,balacc,Fscore_neg);
  fclose(fp1);  

}



int main(int argc, char* argv[])
{
	char *input_image_name,*name;
	
	input_image_name = (char *)malloc(2001*sizeof(char));
	
	strcpy(input_image_name,argv[1]);
	
	char *dirname;
	dirname = (char *)malloc(2001*sizeof(char));
	
	dirname = IITkgp_functions::input_image_name_cut(argv[1]);
	
	name = (char *)malloc(2001*sizeof(char));
	
	FILE *fp;
	
	
	name = IITkgp_functions::CreateNameIntoFolder(dirname,"LabelAllInOne.png");
	
	Mat GT = imread(name,IMREAD_GRAYSCALE);
	fp = fopen("MarginResult.xls","a");
	
	fprintf(fp,"Original\t");
	fclose(fp);

	Evaluate(GT,GT);
	
	
	name = (char *)malloc(2001*sizeof(char));
	
	name = IITkgp_functions::CreateNameIntoFolder("Unpaper",input_image_name);
	
	
	Mat UnpaperOut = imread(name,IMREAD_GRAYSCALE);
	
	
	fp = fopen("MarginResult.xls","a");
	
	fprintf(fp,"Unpaper\t");
	fclose(fp);

	Evaluate(GT,UnpaperOut);
	
	UnpaperOut.release();
	
	name = (char *)malloc(2001*sizeof(char));
	
	name = IITkgp_functions::CreateNameIntoFolder("DAR12",input_image_name);
	
	
	Mat DAROut = imread(name,IMREAD_GRAYSCALE);
	
	fp = fopen("MarginResult.xls","a");
	
	fprintf(fp,"DAR12\t");
	fclose(fp);

	Evaluate(GT,DAROut);
	
	DAROut.release();
	
	name = (char *)malloc(2001*sizeof(char));
	
	name = IITkgp_functions::CreateNameIntoFolder("DAR",input_image_name);
	
	
	Mat DAR = imread(name,IMREAD_GRAYSCALE);
	
	fp = fopen("MarginResult.xls","a");
	
	fprintf(fp,"DAR\t");
	fclose(fp);

	Evaluate(GT,DAR);
	
	DAR.release();
	
	
	name = (char *)malloc(2001*sizeof(char));
	
	name = IITkgp_functions::CreateNameIntoFolder("PageFrame",input_image_name);
		
	Mat PageFrame = imread(name,IMREAD_GRAYSCALE);
	
	fp = fopen("MarginResult.xls","a");
	
	fprintf(fp,"PageFrame\t");
	fclose(fp);

	Evaluate(GT,PageFrame);
	
	
	
	name = (char *)malloc(2001*sizeof(char));
	
	name = IITkgp_functions::CreateNameIntoFolder("NoiseFilter",input_image_name);
	
	
	Mat NoiseFilter = imread(name,IMREAD_GRAYSCALE);
	
	
	fp = fopen("MarginResult.xls","a");
	
	fprintf(fp,"NoiseFilter\t");
	fclose(fp);

	Evaluate(GT,NoiseFilter);
	
	fp = fopen("MarginResult.xls","a");
	
	fprintf(fp,"\n");
	
	fclose(fp);
	
	
	return 0;
  
}