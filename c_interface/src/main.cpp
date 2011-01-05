#include <stdio.h>
#include <stdlib.h>
#include "cv.h"
#include "cvaux.h"
#include "highgui.h"
#include <ml.h>
#include <ctype.h>
#include "conio.h"
#include "CvFastBgGMM.h"

#define TIMER_CUDA	TRUE

#define GROUNDTRUTH_TEST FALSE
#define FG_FRACTION FALSE
#define IMG_SEQUENCE FALSE

#if(GROUNDTRUTH_TEST)
#define GROUNDTRUTH_IMG_SEQUENCE TRUE
#if(GROUNDTRUTH_IMG_SEQUENCE)
	#define GROUNDTRUTH_SQN_TEMPLATE "H:\\Dataset\\VSSN06\\Testvideos\\video8_groundTruth\\img%05d.png"
#else
	#define GROUNDTRUTH_VIDEO "H:\\Dataset\\VSSN06\\Testvideos\\video4_groundTruth.avi"
#endif
	void CompareGT(IplImage* imRet, IplImage* imGT, float& precision, float& recall, float& tp, float& fp);
#endif

#if(FG_FRACTION)
#define FG_FRACTION_FILE "H:\\foregrndFrac.gpu.txt"
	float CountFG(IplImage* imgRet);
#endif

#if(IMG_SEQUENCE)
	#define IMG_SEQUENCE_PARAMETER TRUE
	#if(!IMG_SEQUENCE_PARAMETER)
		#define IMG_SQN_TEMPLATE "H:\\Dataset\\VSSN06\\Testvideos\\video2_matlab\\img%05d.png"
	#endif
#else
	#define TEST_VIDEO "D:\\WIN\\Documents\\Visual Studio 2008\\Projects\\CUDA\\cudaGMM_publish\\data\\vid\\video4.avi"  //"H:\\Dataset\\VSSN06\\Testvideos\\video2.avi"
#endif

void TestCUDAMem(int iSize);

int main(int argc, char** argv)
{
	//if(argc == 2)
	//{
	//	TestCUDAMem(atoi(argv[1]));
	//	return 0;
	//}
#if(IMG_SEQUENCE_PARAMETER)
	if(argc != 2)
		return 0;
#endif

	int key = -1;
	float fTotalTime = 0;
	unsigned int iFrameCnt, iFrameStart;
	iFrameStart = iFrameCnt = 0;
	

#if(IMG_SEQUENCE)
	char arrFileName[MAX_PATH];
#else
	CvCapture* capture = 0;
	capture = cvCaptureFromAVI(TEST_VIDEO);
	if( !capture )
	{
		fprintf(stderr,"Could not initialize...\n");
		return -1;
	}
#endif

#if(GROUNDTRUTH_TEST)
	IplImage* frameGT = NULL;

#if(GROUNDTRUTH_IMG_SEQUENCE)
	char sFrameGTPath[MAX_PATH];
	memset(sFrameGTPath, 0, sizeof(char)*MAX_PATH);
	sprintf_s(sFrameGTPath, MAX_PATH*sizeof(char), GROUNDTRUTH_SQN_TEMPLATE, iFrameCnt);
	//frameGT = cvLoadImage(sFrameGTPath);
#else
	CvCapture* captureGT = cvCaptureFromAVI(GROUNDTRUTH_VIDEO);
	if( !captureGT )
	{
		fprintf(stderr, "Could not initialize...\n");
		return -1;
	}

	//frameGT = cvQueryFrame(captureGT);
#endif

	float precision, recall, precisionAll, recallAll, tp, fp, tpAll, fpAll;
	precisionAll = recallAll = tpAll = fpAll = 0.0f;
#endif

#if(FG_FRACTION)
	FILE* fFraction = fopen(FG_FRACTION_FILE, "w");
	_ASSERT(fFraction);
#endif

	IplImage* videoFrame = NULL;

#if(IMG_SEQUENCE)
	memset(arrFileName, 0, sizeof(char)*MAX_PATH);
#if(IMG_SEQUENCE_PARAMETER)
	sprintf_s(arrFileName, MAX_PATH*sizeof(char), argv[1], iFrameCnt);
#else
	sprintf_s(arrFileName, MAX_PATH*sizeof(char), IMG_SQN_TEMPLATE, iFrameCnt);
#endif
	videoFrame = cvLoadImage(arrFileName);
#else
	videoFrame = cvQueryFrame(capture);
#endif

	if(!videoFrame)
	{
		printf("Bad frame \n");
		exit(0);
	}

	cvNamedWindow("BG", 1);
	cvNamedWindow("FG", 1);
#if(GROUNDTRUTH_TEST)
	cvNamedWindow("GT", 1);
#endif

	CvFastBgGMMParams* pGMMParams = 0;
	CvFastBgGMM* pGMM = NULL;
	pGMMParams = cvCreateFastBgGMMParams(videoFrame->width, videoFrame->height);

	// modify other params in pGMMParams
	//pGMMParams->fAlphaT = 0.0001f;
	pGMMParams->fAlphaT = 0.008f;

	// officially create the parameter (on device memory)
#if(CUDAGMM_VERSION >= 4)
	IplImage* frame1 = NULL, *frame2 = NULL;
	#if(IMG_SEQUENCE)
		memset(arrFileName, 0, sizeof(char)*MAX_PATH);
#if(IMG_SEQUENCE_PARAMETER)
		sprintf_s(arrFileName, MAX_PATH*sizeof(char), argv[1], iFrameCnt+1);
#else
		sprintf_s(arrFileName, MAX_PATH*sizeof(char), IMG_SQN_TEMPLATE, iFrameCnt+1);
#endif
		frame1 = cvLoadImage(arrFileName);
		memset(arrFileName, 0, sizeof(char)*MAX_PATH);
#if(IMG_SEQUENCE_PARAMETER)
		sprintf_s(arrFileName, MAX_PATH*sizeof(char), argv[1], iFrameCnt+2);
#else
		sprintf_s(arrFileName, MAX_PATH*sizeof(char), IMG_SQN_TEMPLATE, iFrameCnt+2);
#endif
		frame2 = cvLoadImage(arrFileName);
	#else
		frame1 = cvQueryFrame(capture);
		frame2 = cvQueryFrame(capture);
	#endif
	pGMM = cvCreateFastBgGMM(pGMMParams, videoFrame, frame1, frame2);
#else
	pGMM = cvCreateFastBgGMM(pGMMParams, videoFrame);
#endif

#if(!TIMER_CUDA)
	LARGE_INTEGER lFrequency, lStart, lEnd;
	QueryPerformanceFrequency(&lFrequency);
#endif

	while(key != 'q')
	{

#if(GROUNDTRUTH_TEST)
#if(GROUNDTRUTH_IMG_SEQUENCE)
		if(frameGT)
			cvReleaseImage(&frameGT);
		memset(sFrameGTPath, 0, sizeof(char)*MAX_PATH);
		sprintf_s(sFrameGTPath, MAX_PATH*sizeof(char), GROUNDTRUTH_SQN_TEMPLATE, iFrameCnt/*+1*/);
		frameGT = cvLoadImage(sFrameGTPath);
#else
		frameGT = cvQueryFrame(captureGT);
#endif
#endif

#if(IMG_SEQUENCE)
		cvReleaseImage(&videoFrame);
		memset(arrFileName, 0, sizeof(char)*MAX_PATH);
#if(CUDAGMM_VERSION >= 4)
#if(IMG_SEQUENCE_PARAMETER)
		sprintf_s(arrFileName, MAX_PATH*sizeof(char), argv[1], iFrameCnt+3);
#else
		sprintf_s(arrFileName, MAX_PATH*sizeof(char), IMG_SQN_TEMPLATE, iFrameCnt+3);
#endif
#else
#if(IMG_SEQUENCE_PARAMETER)
		sprintf_s(arrFileName, MAX_PATH*sizeof(char), argv[1], iFrameCnt+1);
#else
		sprintf_s(arrFileName, MAX_PATH*sizeof(char), IMG_SQN_TEMPLATE, iFrameCnt+1);
#endif
#endif
		videoFrame = cvLoadImage(arrFileName);
#else
		videoFrame = cvQueryFrame(capture);
#endif
#if(GROUNDTRUTH_TEST)
		if( !videoFrame || !frameGT )
			break;
#else
		if( !videoFrame)
			break;
#endif
		iFrameCnt++;
		double fEllapsed;

#if(!TIMER_CUDA)
		QueryPerformanceCounter(&lStart);
#endif

		// Update model
#if(TIMER_CUDA)
		fEllapsed = cvUpdateFastBgGMMTimer(pGMM, videoFrame);
#else
		cvUpdateFastBgGMM(pGMM, videoFrame);
		QueryPerformanceCounter(&lEnd);
		fEllapsed = 1000.0*(lEnd.QuadPart - lStart.QuadPart)/(double)lFrequency.QuadPart;
#endif

#if(FG_FRACTION)
		fprintf(fFraction, "%d;%.4f;%.5lf\r\n", iFrameCnt, CountFG(pGMM->h_outputImg), fEllapsed);
#endif
		fTotalTime += (float)fEllapsed;

		//char sOutput[MAX_PATH];
		//sprintf_s(sOutput, "G:\\Ret\\gpu\\ret%05d.png", iFrameCnt);
		//cvSaveImage(sOutput, pGMM->h_outputImg);

		cvShowImage("BG", videoFrame);
		cvShowImage("FG", pGMM->h_outputImg);	

#if(GROUNDTRUTH_TEST)
		cvShowImage("GT", frameGT);

		if(iFrameCnt - iFrameStart >= 3)
		{
			CompareGT(pGMM->h_outputImg, frameGT, precision, recall, tp, fp);
			precisionAll += precision;
			recallAll += recall;
			tpAll += tp;
			fpAll += fp;
		}
#endif

		key = cvWaitKey(10);
	}

	iFrameCnt -= iFrameStart;
	printf("Average %.1f ms/frame, %.1f FPS\r\n", fTotalTime / (float)iFrameCnt, 1000.0f*(float)iFrameCnt/fTotalTime);
#if(GROUNDTRUTH_TEST)
	printf("Alpha = %f\r\n", pGMMParams->fAlphaT);
	printf(" False Positive: %f; True Positive: %f\r\n", fpAll / (float)iFrameCnt, tpAll/(float)iFrameCnt);
	printf("Precision: %f; Recall: %f\r\n", precisionAll / (float)iFrameCnt, recallAll/(float)iFrameCnt);
#endif

#if(FG_FRACTION)
	fclose(fFraction);
#endif

	cvDestroyWindow("BG");
	cvDestroyWindow("FG");

	cvReleaseFastBgGMM(&pGMM);

#if(!IMG_SEQUENCE)
	cvReleaseCapture(&capture);
#endif

#if(GROUNDTRUTH_TEST)
	cvDestroyWindow("GT");
#if(!GROUNDTRUTH_IMG_SEQUENCE)
	cvReleaseCapture(&captureGT);
#endif
#endif

	//CUT_EXIT(argc, argv);
	return 0;
}

/*==================================================================================*/

/*==================================================================================*/

#if(GROUNDTRUTH_TEST)
void CompareGT(IplImage* imRet, IplImage* imGT, float& precision, float& recall, float& tp, float& fp)
{
	char sRet[MAX_PATH] = "H:\\Ret.png";
	char sGT[MAX_PATH] = "H:\\GT.png";

	cvSaveImage(sRet, imRet);
	cvSaveImage(sGT, imGT);

	IplImage* imgRet = cvLoadImage(sRet);
	IplImage* imgGT = cvLoadImage(sGT);

	int tn, fn;
	unsigned char *pGT, *pRet;
	bool bRet, bGT;
	float fMaxTP;
	tn = fn = 0;
	tp = fp = 0.0f;
	fMaxTP = 0;

	_ASSERT(imgRet->width == imgGT->width && imgRet->height == imgGT->height);
	_ASSERT(imgRet->nChannels == 3 && imgGT->nChannels == 3 && imgGT->depth == IPL_DEPTH_8U);

	for(int i = imgRet->width - 1; i >= 0; --i)
	{
		for(int j = imgRet->height - 1; j >= 0; --j)
		{
			pRet = &CV_IMAGE_ELEM(imgRet, unsigned char, j, 3*i);
			bRet = ((pRet[0] + pRet[1] + pRet[2])/3) >= 100;
			pGT = &CV_IMAGE_ELEM(imgGT, unsigned char, j, 3*i);
			bGT = ((pGT[0] + pGT[1] + pGT[2])/3) >= 100;

			if(bGT)
			{
				fMaxTP ++;
			}

			if(bRet && !bGT)
			{
				fp++;
			}
			else if(!bRet && bGT)
			{
				fn++;
			}
			else if(!bRet)
			{
				tn++;
			}
			else
			{
				tp++;
			}
		}
	}

	if(tp+fp != 0)
		precision = tp*1.0f/(tp+fp);
	else
		precision = 1;
	if(tp+fn != 0)
		recall = tp*1.0f/(tp+fn);
	else
		recall = 1;

	if(fMaxTP == 0)
	{
		tp = 1;
		fp = 0;
	}
	else
	{
		float pxCnt = (imgRet->width * imgRet->height);
		tp /= fMaxTP;
		fp /= (pxCnt - fMaxTP);
	}

	//printf("Pr: %f, R: %f\r\n", precision, recall);
	cvReleaseImage(&imgGT);
	cvReleaseImage(&imgRet);
}

#endif	// GROUNDTRUTH_TEST

#if(FG_FRACTION)
float CountFG(IplImage* imgRet)
{
	int iCnt = 0;
	for(int i = imgRet->width - 1; i >=0; --i)
	{
		for(int j = imgRet->height - 1; j >= 0; --j)
		{
			if((CV_IMAGE_ELEM(imgRet, unsigned char, j, i)) >= 200)
			{
				iCnt++;
			}
		}
	}
	return iCnt*1.0f/(imgRet->width * imgRet->height);
}

#endif	// FG_FRACTION

void TestCUDAMem(int iSize)
{
	iSize *= (1024*1024);
	unsigned char* arrHost = new unsigned char[iSize];
	unsigned char* d_arrData;

	cudaEvent_t start, stop;
	float tCpu1 = 0, tCpu2 = 0, tGpu1 = 0, tGpu2 = 0;
	cudaMalloc((void**)&d_arrData, iSize);

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	cudaEventRecord( start, 0 );
	cudaMemcpy(d_arrData, arrHost, iSize, cudaMemcpyHostToDevice);
	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );
	cudaEventElapsedTime( &tCpu1, start, stop );

	cudaEventRecord( start, 0 );
	cudaMemcpy(arrHost, d_arrData, iSize, cudaMemcpyDeviceToHost);
	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );
	cudaEventElapsedTime( &tGpu1, start, stop );

	cudaFree(d_arrData);

	unsigned char* h_pinnedData, *d_arrData2;
	cudaStream_t copyStream;

	cudaStreamCreate(&copyStream);
	cudaHostAlloc((void**)&h_pinnedData, iSize, cudaHostAllocDefault);
	cudaMalloc((void**)&d_arrData2, iSize);

	cudaEventRecord( start, 0 );
	memcpy(h_pinnedData, arrHost, iSize);
	cudaMemcpyAsync(d_arrData2, h_pinnedData, iSize, cudaMemcpyHostToDevice, copyStream);
	cudaStreamSynchronize(copyStream);
	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );
	cudaEventElapsedTime( &tCpu2, start, stop );

	cudaEventRecord( start, 0 );
	cudaMemcpyAsync(h_pinnedData, d_arrData2, iSize, cudaMemcpyDeviceToHost, copyStream);
	cudaStreamSynchronize(copyStream);
	memcpy(arrHost, h_pinnedData, iSize);
	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );
	cudaEventElapsedTime( &tGpu2, start, stop );

	cudaEventDestroy( start );
	cudaEventDestroy( stop );

	cudaStreamDestroy(copyStream);
	cudaFree(d_arrData2);
	cudaFreeHost(h_pinnedData);

	delete[] arrHost;
	float fFactor = 1000.0f*iSize/1024.0f/1024.0f/1024.0f;

	printf("Transfer size: %d bytes\r\n", iSize);
	printf("Method 1 (pageable memory):\r\n\tCPU->GPU: %f ms (%f Gbps)\r\n\tGPU->CPU: %f ms (%f Gbps)\r\n",
		tCpu1, fFactor/tCpu1,
		tGpu1, fFactor/tGpu1);

	printf("Method 2 (pinned memory):\r\n\tCPU->GPU: %f ms (%f Gbps)\r\n\tGPU->CPU: %f ms (%f Gbps)\r\n", 
		tCpu2, fFactor/tCpu2,
		tGpu2, fFactor/tGpu2);
	
}
