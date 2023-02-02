#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>

namespace py = pybind11;

/*
*********************************************************************
function name: matmul_cuda
description: dot product of two arbitrarily sized matrices.
parameters:
  image: Input image of size m X n.
  weight: weight kernel of size n X k.
  bias: bias per output channel.
  output: output image of size m x k.
  m,n,k: sizes of matrices.
  batch_size: Number of images in each batch.
return: none
Acknowledgement: Original code from 'lzhengchun/matrix-cuda' on github.
link: https://github.com/lzhengchun/matrix-cuda/blob/master/matrix_cuda.cu
*********************************************************************
*/
__global__ void matmul_cuda(
  const float *image,
  const float *weight,
	const float *bias,
  float *output,
  const int m,
  const int n,
  const int k,
  const int batch_size)
{

    // This code doesn't really get much faster using shared memory, since
    // accesses to the image matrix are all sequential anyway. The first access
    // already caches everything, making shared memory useless.

    int img = blockIdx.z * blockDim.z + threadIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f, product_appx = 0.0f, product = 0.0f;
    if( col < k && row < m && img < batch_size){
        for(int i = 0; i < n; i++){
						sum += image[(img*m*n)+(row*n) + i] * weight[i * k + col];
				}
        output[(img*m*k)+(row*k) + col] = sum + bias[col];
    }
}

__global__ void my_matmul_cuda(
  const float **image,
  const float *weight,
	const float *bias,
  float *output,
  const int l,
  const int b,
  const int w,
  const int n,
  int *pixel_counts,
  int *feature_counts,
  const int k,
  const int batch_size)
{

  //CONVIENE CHE OGNI BLOCCO DI FEATURE SIA VISTO IN DUE DIMENSIONI OSSIA #PIXEL(BATCHxWIDTHxHEIGHT) E #FEATURES. ENTRAMBI QUESTI DUE VALORI CAMBIANO PER OGNI BLOCCO E VA
  //TENUTA TRACCIA DELLE DIMENSIONI DI OGNI BLOCCO IN DUE VETTORI pixel_counts E feature_counts

    // This code doesn't really get much faster using shared memory, since
    // accesses to the image matrix are all sequential anyway. The first access
    // already caches everything, making shared memory useless.
    //int img = blockIdx.z * blockDim.z + threadIdx.z;
    int pixel = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int m = b * w * w;

    float sum = 0.0f, product_appx = 0.0f, product = 0.0f;
    int feature_counter = 0;
    if(col < k && pixel < m){
      for(int i=0; i<l; i++){
        for(int j = 0; j < feature_counts[i]; j++){
          if(w == pixel_counts[i]){
            sum += image[i][pixel * feature_counts[i] + j] * weight[feature_counter * k + col];
          }else{
            //int actual_pixel = pixel * pixel_counts[i];
            //actual_pixel = actual_pixel / m;

            int actual_pixel = pixel / w;
            actual_pixel *= pixel_counts[i];
            actual_pixel /= w;
            actual_pixel *= pixel_counts[i];
            actual_pixel += (pixel % w * pixel_counts[i]) / w;

            sum += image[i][actual_pixel * feature_counts[i] + j] * weight[feature_counter * k + col];
          }
          feature_counter++;
        }
      }

      output[pixel * k + col] = bias[col] + sum;
    }

    return;


    /*int img = 0;//blockIdx.z * blockDim.z + threadIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f, product_appx = 0.0f, product = 0.0f;
    if( col < k && row < m && img < batch_size){
        for(int i = 0; i < n; i++){
						sum += image[(img*m*n)+(row*n) + i] * weight[i * k + col];
				}
        output[(img*m*k)+(row*k) + col] = sum + bias[col];
    }*/
}

/*
*********************************************************************
function name: conv_forward
description: convolutional layer that calls the matmul cuda kernel.
parameters:
  image: Input image of size m X n.
  weight: weight kernel of size n X k.
  bias: bias per output channel.
  m,n,k: sizes of matrices.
  b: Number of images in each batch.
return:
  output: output image of size m x k.
*********************************************************************
*/
torch::Tensor conv_forward(
  torch::Tensor input,
  torch::Tensor weight,
  torch::Tensor bias,
	int m,
	int n,
	int k,
	int b
) {

  // Create an output of size b X m X k, directly on the GPU.
	auto options = torch::TensorOptions().device(torch::kCUDA, 0);
	auto output = torch::zeros({b, m, k}, options);

  // Use this block size to not exceed 1024 threads across all 3 dimensions.
  // You can also do dimblock(16 x 16 x 4) to use all 1024 threads if your
  // batches are small.
	unsigned int block_size = 8;
	unsigned int grid_rows = (m + block_size - 1) / block_size;
	unsigned int grid_cols = (k + block_size - 1) / block_size;
	unsigned int grid_images = (b + block_size - 1) / block_size;

	dim3 dimGrid(grid_cols, grid_rows, grid_images);
	dim3 dimBlock(block_size, block_size, block_size);

  // This is not the 'pytorch recommended way' of launching this kernel.
  // But it works just fine so I've left it this way since it is easier to debug
  // if there is an issue launching the kernel for example.

	matmul_cuda<<<dimGrid, dimBlock>>>(
		input.data_ptr<float>(),
		weight.data_ptr<float>(),
		bias.data_ptr<float>(),
		output.data_ptr<float>(),
		m, n, k, b
	);

  cudaDeviceSynchronize();
  return output;
}

/*
*********************************************************************
function name: linear_forward
description: linear layer that calls the matmul cuda kernel.
parameters:
  image: Input image of size m X n.
  weight: weight kernel of size n X k.
  bias: bias per output channel.
  m,n,k: sizes of matrices.
return:
  output: output image of size m x k.
*********************************************************************
*/
torch::Tensor old_linear_forward(
  torch::Tensor input,
  torch::Tensor weight,
  torch::Tensor bias,
	int m,
	int n,
	int k
) {

	auto options = torch::TensorOptions().device(torch::kCUDA, 0);
	auto output = torch::zeros({m,k}, options);

	unsigned int block_size = 32;
	unsigned int grid_rows = (m + block_size - 1) / block_size;
	unsigned int grid_cols = (k + block_size - 1) / block_size;

	dim3 dimGrid(grid_cols, grid_rows);
	dim3 dimBlock(block_size, block_size);

  // Linear layers have a vector input. But to re-use the matmul kernel,
  // just pass in a 'batch' of inputs as an m X n matrix, to be multiplied
  // by the n x k weights, to get 'm' output images.

	matmul_cuda<<<dimGrid, dimBlock>>>(
		input.data_ptr<float>(),
		weight.data_ptr<float>(),
		bias.data_ptr<float>(),
		output.data_ptr<float>(),
		m, n, k, 1 // Pass in b=1 since there is no z-dimension for linear layers
	);

  cudaDeviceSynchronize();
  return output;
}




void myPrint(const float *tensor){
  std::cout << std::to_string(tensor[0]) << "\n";
}



//My new linear forward!

torch::Tensor linear_forward(
  std::vector<torch::Tensor> inputs,
  torch::Tensor weight,
  torch::Tensor bias,
	//int m,//batch size
	//int n,//input features
	int k//output features
) {
  //myPrint(bias.data_ptr<float>());
  //return inputs[1];

  int l = inputs.size();//number of input tensors
  int b;//batch size
  int w = 0;//width
  int n = 0;//input features
  
  std::vector<int> pixel_counts;//number of pixels per layer
  pixel_counts.reserve(l);
  std::vector<int> feature_counts;//number of features per layer
  feature_counts.reserve(l);
  std::vector<const float*> input_ptrs;
  input_ptrs.reserve(l);
  for (const auto& input : inputs) {
    b = input.size(0);
    w = std::max(w, int(input.size(1)));
    n += input.size(3);

    pixel_counts.push_back(input.size(1));
    feature_counts.push_back(input.size(3));
    input_ptrs.push_back(input.reshape({-1, input.size(3)}).data_ptr<float>());
  }
  int m = b * w * w;

  //std::cout << std::to_string(pixel_counts[0]) << "\n";
  //std::cout << std::to_string(pixel_counts[1]) << "\n";
  //std::cout << std::to_string(feature_counts[0]) << "\n";
  //std::cout << std::to_string(feature_counts[1]) << "\n";
  //std::cout << std::to_string(m) << "\n";

	auto options = torch::TensorOptions().device(torch::kCUDA, 0);
	auto output = torch::zeros({m, k}, options);

	unsigned int block_size = 32;
	unsigned int grid_pixels = (m + block_size - 1) / block_size;
	unsigned int grid_cols = (k + block_size - 1) / block_size;

	dim3 dimGrid(grid_cols, grid_pixels);
	dim3 dimBlock(block_size, block_size);

  // Linear layers have a vector input. But to re-use the matmul kernel,
  // just pass in a 'batch' of inputs as an m X n matrix, to be multiplied
  // by the n x k weights, to get 'm' output images.

  int* pixel_counts_gpu;
  int* feature_counts_gpu;
  
  cudaMalloc((void**)&pixel_counts_gpu, l * sizeof(int));
  cudaMalloc((void**)&feature_counts_gpu, l * sizeof(int));

  cudaMemcpy(pixel_counts_gpu, pixel_counts.data(), l * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(feature_counts_gpu, feature_counts.data(), l * sizeof(int), cudaMemcpyHostToDevice);

  const float** input_ptrs_gpu;
  cudaMalloc((void**)&input_ptrs_gpu, l * sizeof(float*));
  cudaMemcpy(input_ptrs_gpu, input_ptrs.data(), l * sizeof(float*), cudaMemcpyHostToDevice);


	my_matmul_cuda<<<dimGrid, dimBlock>>>(
		input_ptrs_gpu,//inputs[1].data_ptr<float>(),//input_ptrs.data(),//input.data_ptr<float>(),
		weight.data_ptr<float>(),
		bias.data_ptr<float>(),
		output.data_ptr<float>(),
		l, b, w, n, pixel_counts_gpu, feature_counts_gpu, k, 1 // Pass in b=1 since there is no z-dimension for linear layers
	);

  cudaFree(pixel_counts_gpu);
  cudaFree(feature_counts_gpu);
  cudaFree(input_ptrs_gpu);

  cudaDeviceSynchronize();
  return output;
}

// Binding to generate the .so file, to call from python.
PYBIND11_MODULE(cuda_layers, m) {
  m.doc() = "Implementation of forward pass of conv and linear layers in CUDA";
  m.def("conv_forward", &conv_forward, "conv_forward (CUDA)");
	m.def("linear_forward", &linear_forward, "linear_forward (CUDA)");
}
