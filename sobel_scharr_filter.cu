#include <opencv2/opencv.hpp>
#include <iostream>
#include <cuda_runtime.h>
#include <assert.h>
#include <stdio.h>

#define IMG_WIDTH 800
#define IMG_HEIGHT 600
#define SOBEL_THRESHOLD 128
#define SCHARR_THRESHOLD 128

#define FILTER_WIDTH 3
#define FILTER_HEIGHT 3
#define THREADS_PER_BLOCK_X 32
#define THREADS_PER_BLOCK_Y 32

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
    }                                                                          \
}

enum filter_type{Sobel,Scharr};


using namespace std;

__constant__ int sobel_horizontal[3][3];
__constant__ int sobel_vertical[3][3];
__constant__ int scharr_horizontal[3][3];
__constant__ int scharr_vertical[3][3];


void setup_filter (void)
{
    const int sob_h[3][3] = {1, 2, 1, 0, 0, 0, -1, -2, -1};
    const int sob_v[3][3] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
    const int sch_h[3][3] = {3, 10, 3, 0, 0, 0, -3, -10, -3};
    const int sch_v[3][3] = {3, 0, -3, 10, 0, -10, 3, 0, -3};
    CHECK(cudaMemcpyToSymbol( sobel_horizontal, sob_h, FILTER_HEIGHT*FILTER_WIDTH * sizeof(int)));
    CHECK(cudaMemcpyToSymbol( sobel_vertical, sob_v, FILTER_HEIGHT*FILTER_WIDTH * sizeof(int)));
    CHECK(cudaMemcpyToSymbol( scharr_horizontal, sch_h, FILTER_HEIGHT*FILTER_WIDTH * sizeof(int)));
    CHECK(cudaMemcpyToSymbol( scharr_vertical, sch_v, FILTER_HEIGHT*FILTER_WIDTH * sizeof(int)));

}

__global__ void sobel_filter(const int *gray_img, int *horizontal_output,int *vertical_output,unsigned int width, unsigned int height, unsigned int threshold){
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    unsigned int index = y*width + x;

    if (x>0 && y>0 && x<width-1 && y<height-1 ){
        for(int dy=-1;dy<=1;++dy){
            for(int dx=-1;dx<=1;++dx){
                horizontal_output[index]+=gray_img[(y+dy)*width+(x+dx)]*sobel_horizontal[1+dy][1+dx];
                vertical_output[index]+=gray_img[(y+dy)*width+(x+dx)]*sobel_vertical[1+dy][1+dx];
              }
        }

        if(horizontal_output[index]<0) horizontal_output[index]*=-1;
        if(vertical_output[index]<0) vertical_output[index]*=-1;
        horizontal_output[index]=min(255,horizontal_output[index]);
        vertical_output[index]=min(255,vertical_output[index]);

        if(horizontal_output[index]<threshold) horizontal_output[index]=0;
        if(vertical_output[index]<threshold) vertical_output[index]=0;

    }
}

__global__ void scharr_filter(const int *gray_img, int *horizontal_output,int *vertical_output,unsigned int width, unsigned int height, unsigned int threshold){
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    unsigned int index = y*width + x;

    if (x>0 && y>0 && x<width-1 && y<height-1 ){
        for(int dy=-1;dy<=1;++dy){
            for(int dx=-1;dx<=1;++dx){
                horizontal_output[index]+=gray_img[(y+dy)*width+(x+dx)]*scharr_horizontal[1+dy][1+dx];
                vertical_output[index]+=gray_img[(y+dy)*width+(x+dx)]*scharr_vertical[1+dy][1+dx];
              }
        }
        //horizontal_output[index]=gray_img[index];
        if(horizontal_output[index]<0) horizontal_output[index]*=-1;
        if(vertical_output[index]<0) vertical_output[index]*=-1;
        horizontal_output[index]=min(255,horizontal_output[index]);
        vertical_output[index]=min(255,vertical_output[index]);

        if(horizontal_output[index]<threshold) horizontal_output[index]=0;
        if(vertical_output[index]<threshold) vertical_output[index]=0;

    }
}


pair<cv::Mat,cv::Mat> call_cuda(cv::Mat gray_img,filter_type fflag){

  int dev = 0;
  cudaDeviceProp deviceProp;
  CHECK(cudaGetDeviceProperties(&deviceProp, dev));
  CHECK(cudaSetDevice(dev));
  cout<<deviceProp.major<<endl;

  assert (gray_img.channels()==1);  //Please input gray scale image
  assert (gray_img.cols==IMG_WIDTH && gray_img.rows==IMG_HEIGHT); //check the size of image

  setup_filter();

  gray_img.convertTo(gray_img,CV_32S);

  int img_size=gray_img.rows*gray_img.cols;
  int *device_input,*horizontal_output,*vertical_output;


  cudaMalloc(&device_input,sizeof(int)*img_size);
  cudaMalloc(&horizontal_output,sizeof(int)*img_size);
  cudaMalloc(&vertical_output,sizeof(int)*img_size);

  cudaMemcpy(device_input,gray_img.data,sizeof(int)*img_size,cudaMemcpyHostToDevice);

  dim3 ThreadsPerBlocks(THREADS_PER_BLOCK_X,THREADS_PER_BLOCK_Y);
  dim3 BlocksNum((IMG_WIDTH+ThreadsPerBlocks.x)/ThreadsPerBlocks.x,(IMG_HEIGHT+ThreadsPerBlocks.y)/ThreadsPerBlocks.y);

  switch(fflag){
    case Sobel:
      sobel_filter<<<BlocksNum,ThreadsPerBlocks>>>(device_input,horizontal_output,vertical_output,IMG_WIDTH,IMG_HEIGHT,SOBEL_THRESHOLD);
      break;
    case Scharr:
      scharr_filter<<<BlocksNum,ThreadsPerBlocks>>>(device_input,horizontal_output,vertical_output,IMG_WIDTH,IMG_HEIGHT,SCHARR_THRESHOLD);
      break;
  }


  CHECK(cudaDeviceSynchronize());

  int *host_filter_h=new int[img_size];
  int *host_filter_v=new int[img_size];
  cudaMemcpy(host_filter_h,horizontal_output,sizeof(int)*img_size,cudaMemcpyDeviceToHost);
  cudaMemcpy(host_filter_v,vertical_output,sizeof(int)*img_size,cudaMemcpyDeviceToHost);


  cv::Mat output_filter_h;
  cv::Mat output_filter_v;
  cv::Mat(gray_img.size(),CV_32S,host_filter_h).assignTo(output_filter_h,CV_8U);
  cv::Mat(gray_img.size(),CV_32S,host_filter_v).assignTo(output_filter_v,CV_8U);

  delete [] host_filter_h;
  delete [] host_filter_v;
  cudaFree(device_input);
  cudaFree(horizontal_output);
  cudaFree(vertical_output);
  cudaDeviceReset();

  return make_pair(move(output_filter_h),move(output_filter_v));
}
