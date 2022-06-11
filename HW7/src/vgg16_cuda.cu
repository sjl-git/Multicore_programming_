#include "vgg16_cuda.h"
#define BLOCKSIZE32 32
#define BLOCKSIZE16 16
#define BLOCKSIZE8 8
#define BLOCKSIZE4 4
#define BLOCKSIZE2 2
#define BLOCKSIZE1 1

__global__ void normalize(uint8_t* d_image, float* d_input, int B, int C, int H, int W, int w_grid) {
    int b = blockIdx.x;
    int c = blockIdx.y;
    int h = (blockIdx.z / w_grid) * BLOCKSIZE32 + threadIdx.y;
    int w = (blockIdx.z % w_grid) * BLOCKSIZE32 + threadIdx.x;
    int i = b * (C * H * W) + c * (H * W) + h * (W) + w;

    float max_int = 255.0L;
    float mean = 0.5L;
    float var = 0.5L;


    d_input[i] = d_image[i] / max_int;
    d_input[i] = (d_input[i] - mean) / var;
}

__global__ void pad(float* d_input, float* d_input_padded, int B, int C, int H, int W, int P, int w_grid) {
    int b = blockIdx.x;
    int c = blockIdx.y;
    int h = (blockIdx.z / w_grid) * blockDim.x + threadIdx.y;
    int w = (blockIdx.z % w_grid) * blockDim.x + threadIdx.x;

    int H_OUT = H+2*P;
    int W_OUT = W+2*P;
    int input_base = b * (C * H * W) + c * (H * W) + h * (W) + w;
    int output_index = b * (C * H_OUT * W_OUT) + c * (H_OUT * W_OUT) + (h+P) * W_OUT + (w + P);

    d_input_padded[output_index] = d_input[input_base];
}

__global__ void conv(float* d_input, float* d_output, float* d_conv_weight, float* d_conv_bias, int B, int H, int W, int IC, int OC, int K, int w_grid) {
    int b = blockIdx.x;
    int oc = blockIdx.y;
    int h = (blockIdx.z / w_grid) * blockDim.x + threadIdx.y;
    int w = (blockIdx.z % w_grid) * blockDim.x + threadIdx.x;
    int H_OUT = H - (K - 1);
    int W_OUT = W - (K - 1);

    int output_index = b * (OC * H_OUT * W_OUT) + oc * (H_OUT * W_OUT) + h * (W_OUT) + w;
    d_output[output_index] = d_conv_bias[oc];

    float val = 0.0;
    for (int ic = 0; ic < IC; ic++) {
        int input_base = b * (IC * H * W) + ic * (H * W) + h * (W) + w;
        int kernel_base = oc * (IC * K * K) + ic * (K * K);
        for (int kh = 0; kh < K; kh++) {
            for (int kw = 0; kw < K; kw++) {
                val += d_input[input_base + kh * (W) + kw] * d_conv_weight[kernel_base + kh * (K) + kw];
            }
        }
    }
    d_output[output_index] += val;
}

__global__ void relu(float* d_feature_map, int size, int B, int C, int H, int W, int w_grid) {
    int b = blockIdx.x;
    int c = blockIdx.y;
    int h = (blockIdx.z / w_grid) * blockDim.x + threadIdx.y;
    int w = (blockIdx.z % w_grid) * blockDim.x + threadIdx.x;
    int i = b * (C * H * W) + c * (H * W) + h * (W) + w;

    if (i < size) {
        if (d_feature_map[i] < (float)0.0) {
            d_feature_map[i] = (float)0.0;
        }
    }
}

__global__ void pool(float* d_feature_map, float* d_S_feature_map, int B, int C, int H, int W, int w_grid) {
    int b = blockIdx.x;
    int c = blockIdx.y;
    int h = (blockIdx.z / w_grid) * blockDim.x + threadIdx.y;
    int w = (blockIdx.z % w_grid) * blockDim.x + threadIdx.x;

    int scale = 2;
    int H_OUT = H / scale;
    int W_OUT = W / scale;

    int output_index = b * (C * H_OUT * W_OUT) + c * (H_OUT * W_OUT) + h * W_OUT + w;
    int i = b * (C * H * W) + c * (H * W) + (h*2) * (W) + w * 2;
    float max_val = 0;
    for (int sh = 0; sh < scale; sh++) {
        for (int sw = 0; sw < scale; sw++) {
            float val = d_feature_map[i + sh * (W) + sw];
            if (val > max_val) {
                max_val = val;
            }
        }
    }
    d_S_feature_map[output_index] = max_val;
}

__global__ void fc(float* d_S5_feature_map, float* d_output, float* d_fc1_weight, float* d_fc1_bias, int B, int IC, int OC) {
  // Fully Connected
    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int b = bx * 32 + ty;
    int oc = tx;
    d_output[b * OC + oc] = d_fc1_bias[oc];
    for (int ic=0; ic < IC; ic++) {
        d_output[b * OC + oc] += d_fc1_weight[oc * IC + ic] * d_S5_feature_map[b * IC + ic];
    }
}

void vgg16_cuda::predict(int batch) {

    //////////BLOCK 1/////////////////////////////////
    dim3 DimGrid0(batch, input_channel, (input_size*input_size)/(BLOCKSIZE32*BLOCKSIZE32));
    dim3 DimBlock0(BLOCKSIZE32, BLOCKSIZE32, 1);
    int w_grid = input_size / BLOCKSIZE32;
    normalize<<<DimGrid0, DimBlock0>>>(d_image, d_input, batch, input_channel, input_size, input_size, w_grid);
    // TODO: Implement pad
    pad<<<DimGrid0, DimBlock0>>>(d_input, d_input_padded, batch, input_channel, input_size, input_size, conv1_1_padding_size, w_grid);
    // TODO: Implement conv1_1
    dim3 DimGrid1_1(batch, C1_1_channel, (C1_1_size*C1_1_size)/(BLOCKSIZE32*BLOCKSIZE32));
    dim3 DimBlock1_1(BLOCKSIZE32, BLOCKSIZE32, 1);
    w_grid = C1_1_size / BLOCKSIZE32;
    conv<<<DimGrid1_1, DimBlock1_1>>>(d_input_padded, d_C1_1_feature_map, d_conv1_1_weight, d_conv1_1_bias, batch, input_size+2*conv1_1_padding_size, input_size+2*conv1_1_padding_size, conv1_1_in_channel, conv1_1_out_channel, conv1_1_kernel_size, w_grid);
    // TODO: Implement relu
    relu<<<DimGrid1_1, DimBlock1_1>>>(d_C1_1_feature_map, batch * C1_1_channel * C1_1_size * C1_1_size, batch, C1_1_channel, C1_1_size, C1_1_size, w_grid);
    // TODO: Implement pad
    pad<<<DimGrid1_1, DimBlock1_1>>>(d_C1_1_feature_map, d_C1_1_feature_map_padded, batch, C1_1_channel, C1_1_size, C1_1_size, conv1_2_padding_size, w_grid);
    // TODO: Implement conv1_2
    dim3 DimGrid1_2(batch, C1_2_channel, (C1_2_size*C1_2_size)/(BLOCKSIZE32*BLOCKSIZE32));
    dim3 DimBlock1_2(BLOCKSIZE32, BLOCKSIZE32, 1);
    w_grid = C1_2_size / BLOCKSIZE32;
    conv<<<DimGrid1_2, DimBlock1_2>>>(d_C1_1_feature_map_padded, d_C1_2_feature_map, d_conv1_2_weight, d_conv1_2_bias, batch, C1_1_size+2*conv1_2_padding_size, C1_1_size+2*conv1_2_padding_size, conv1_2_in_channel, conv1_2_out_channel, conv1_2_kernel_size, w_grid);
    // TODO: Implement relu
    relu<<<DimGrid1_2, DimBlock1_2>>>(d_C1_2_feature_map, batch * C1_2_channel * C1_2_size * C1_2_size, batch, C1_2_channel, C1_2_size, C1_2_size, w_grid);
    // TODO: Implement pool
    dim3 DimGrid1_3(batch, S1_channel, (S1_size*S1_size)/(BLOCKSIZE16*BLOCKSIZE16));
    dim3 DimBlock1_3(BLOCKSIZE16, BLOCKSIZE16, 1);
    w_grid = S1_size / BLOCKSIZE16;
    pool<<<DimGrid1_3, DimBlock1_3>>>(d_C1_2_feature_map, d_S1_feature_map, batch, C1_2_channel, C1_2_size, C1_2_size, w_grid);
    //////////BLOCK 2/////////////////////////////////
    // TODO: Implement pad
    pad<<<DimGrid1_3, DimBlock1_3>>>(d_S1_feature_map, d_S1_feature_map_padded, batch, S1_channel, S1_size, S1_size, conv2_1_padding_size, w_grid);
    // TODO: Implement conv2_1
    dim3 DimGrid2_1(batch, C2_1_channel, (C2_1_size*C2_1_size)/(BLOCKSIZE16*BLOCKSIZE16));
    dim3 DimBlock2_1(BLOCKSIZE16, BLOCKSIZE16, 1);
    w_grid = C2_1_size / BLOCKSIZE16;
    conv<<<DimGrid2_1, DimBlock2_1>>>(d_S1_feature_map_padded, d_C2_1_feature_map, d_conv2_1_weight, d_conv2_1_bias, batch, S1_size+2*conv2_1_padding_size, S1_size+2*conv2_1_padding_size, conv2_1_in_channel, conv2_1_out_channel, conv2_1_kernel_size, w_grid);
    // TODO: Implement relu
    relu<<<DimGrid2_1, DimBlock2_1>>>(d_C2_1_feature_map, batch * C2_1_channel * C2_1_size * C2_1_size, batch, C2_1_channel, C2_1_size, C2_1_size, w_grid);
    // TODO: Implement pad
    pad<<<DimGrid2_1, DimBlock2_1>>>(d_C2_1_feature_map, d_C2_1_feature_map_padded, batch, C2_1_channel, C2_1_size, C2_1_size, conv2_2_padding_size, w_grid);
    // TODO: Implement conv2_2
    dim3 DimGrid2_2(batch, C2_2_channel, (C2_2_size*C2_2_size)/(BLOCKSIZE16*BLOCKSIZE16));
    dim3 DimBlock2_2(BLOCKSIZE16, BLOCKSIZE16, 1);
    w_grid = C2_2_size / BLOCKSIZE16;
    conv<<<DimGrid2_2, DimBlock2_2>>>(d_C2_1_feature_map_padded, d_C2_2_feature_map, d_conv2_2_weight, d_conv2_2_bias, batch, C2_1_size+2*conv2_2_padding_size, C2_1_size+2*conv2_2_padding_size, conv2_2_in_channel, conv2_2_out_channel, conv2_2_kernel_size, w_grid);
    // TODO: Implement relu
    relu<<<DimGrid2_2, DimBlock2_2>>>(d_C2_2_feature_map, batch * C2_2_channel * C2_2_size * C2_2_size, batch, C2_2_channel, C2_2_size, C2_2_size, w_grid);
    // TODO: Implement pool
    dim3 DimGrid2_3(batch, S2_channel, (S2_size*S2_size)/(BLOCKSIZE8*BLOCKSIZE8));
    dim3 DimBlock2_3(BLOCKSIZE8, BLOCKSIZE8, 1);
    w_grid = S2_size / BLOCKSIZE8;
    pool<<<DimGrid2_3, DimBlock2_3>>>(d_C2_2_feature_map, d_S2_feature_map, batch, C2_2_channel, C2_2_size, C2_2_size, w_grid);
    //////////BLOCK 3/////////////////////////////////
    // TODO: Implement pad
    pad<<<DimGrid2_3, DimBlock2_3>>>(d_S2_feature_map, d_S2_feature_map_padded, batch, S2_channel, S2_size, S2_size, conv3_1_padding_size, w_grid);
    // TODO: Implement conv3_1
    dim3 DimGrid3_1(batch, C3_1_channel, (C3_1_size*C3_1_size)/(BLOCKSIZE8*BLOCKSIZE8));
    dim3 DimBlock3_1(BLOCKSIZE8, BLOCKSIZE8, 1);
    w_grid = C3_1_size / BLOCKSIZE8;
    conv<<<DimGrid3_1, DimBlock3_1>>>(d_S2_feature_map_padded, d_C3_1_feature_map, d_conv3_1_weight, d_conv3_1_bias, batch, S2_size+2*conv3_1_padding_size, S2_size+2*conv3_1_padding_size, conv3_1_in_channel, conv3_1_out_channel, conv3_1_kernel_size, w_grid);
    // TODO: Implement relu
    relu<<<DimGrid3_1, DimBlock3_1>>>(d_C3_1_feature_map, batch * C3_1_channel * C3_1_size * C3_1_size, batch, C3_1_channel, C3_1_size, C3_1_size, w_grid);
    // TODO: Implement pad
    pad<<<DimGrid3_1, DimBlock3_1>>>(d_C3_1_feature_map, d_C3_1_feature_map_padded, batch, C3_1_channel, C3_1_size, C3_1_size, conv3_2_padding_size, w_grid);
    // TODO: Implement conv3_2
    dim3 DimGrid3_2(batch, C3_2_channel, (C3_2_size*C3_2_size)/(BLOCKSIZE8*BLOCKSIZE8));
    dim3 DimBlock3_2(BLOCKSIZE8, BLOCKSIZE8, 1);
    w_grid = C3_2_size / BLOCKSIZE8;
    conv<<<DimGrid3_2, DimBlock3_2>>>(d_C3_1_feature_map_padded, d_C3_2_feature_map, d_conv3_2_weight, d_conv3_2_bias, batch, C3_1_size+2*conv3_2_padding_size, C3_1_size+2*conv3_2_padding_size, conv3_2_in_channel, conv3_2_out_channel, conv3_2_kernel_size, w_grid);
    // TODO: Implement relu
    relu<<<DimGrid3_2, DimBlock3_2>>>(d_C3_2_feature_map, batch * C3_2_channel * C3_2_size * C3_2_size, batch, C3_2_channel, C3_2_size, C3_2_size, w_grid);
    // TODO: Implement pad
    pad<<<DimGrid3_2, DimBlock3_2>>>(d_C3_2_feature_map, d_C3_2_feature_map_padded, batch, C3_2_channel, C3_2_size, C3_2_size, conv3_3_padding_size, w_grid);
    // TODO: Implement conv3_3
    dim3 DimGrid3_3(batch, C3_3_channel, (C3_3_size*C3_3_size)/(BLOCKSIZE8*BLOCKSIZE8));
    dim3 DimBlock3_3(BLOCKSIZE8, BLOCKSIZE8, 1);
    w_grid = C3_3_size / BLOCKSIZE8;
    conv<<<DimGrid3_3, DimBlock3_3>>>(d_C3_2_feature_map_padded, d_C3_3_feature_map, d_conv3_3_weight, d_conv3_3_bias, batch, C3_2_size+2*conv3_3_padding_size, C3_2_size+2*conv3_3_padding_size, conv3_3_in_channel, conv3_3_out_channel, conv3_3_kernel_size, w_grid);
    // TODO: Implement relu
    relu<<<DimGrid3_3, DimBlock3_3>>>(d_C3_3_feature_map, batch * C3_3_channel * C3_3_size * C3_3_size, batch, C3_3_channel, C3_3_size, C3_3_size, w_grid);
    // TODO: Implement pool
    dim3 DimGrid3_4(batch, S3_channel, (S3_size*S3_size)/(BLOCKSIZE4*BLOCKSIZE4));
    dim3 DimBlock3_4(BLOCKSIZE4, BLOCKSIZE4, 1);
    w_grid = S3_size / BLOCKSIZE4;
    pool<<<DimGrid3_4, DimBlock3_4>>>(d_C3_3_feature_map, d_S3_feature_map, batch, C3_3_channel, C3_3_size, C3_3_size, w_grid);
    //////////BLOCK 4/////////////////////////////////
    // TODO: Implement pad
    pad<<<DimGrid3_4, DimBlock3_4>>>(d_S3_feature_map, d_S3_feature_map_padded, batch, S3_channel, S3_size, S3_size, conv4_1_padding_size, w_grid);
    // TODO: Implement conv4_1
    dim3 DimGrid4_1(batch, C4_1_channel, (C4_1_size*C4_1_size)/(BLOCKSIZE4*BLOCKSIZE4));
    dim3 DimBlock4_1(BLOCKSIZE4, BLOCKSIZE4, 1);
    w_grid = C4_1_size / BLOCKSIZE4;
    conv<<<DimGrid4_1, DimBlock4_1>>>(d_S3_feature_map_padded, d_C4_1_feature_map, d_conv4_1_weight, d_conv4_1_bias, batch, S3_size+2*conv4_1_padding_size, S3_size+2*conv4_1_padding_size, conv4_1_in_channel, conv4_1_out_channel, conv4_1_kernel_size, w_grid);
    // TODO: Implement relu
    relu<<<DimGrid4_1, DimBlock4_1>>>(d_C4_1_feature_map, batch * C4_1_channel * C4_1_size * C4_1_size, batch, C4_1_channel, C4_1_size, C4_1_size, w_grid);
    // TODO: Implement pad
    pad<<<DimGrid4_1, DimBlock4_1>>>(d_C4_1_feature_map, d_C4_1_feature_map_padded, batch, C4_1_channel, C4_1_size, C4_1_size, conv4_2_padding_size, w_grid);
    // TODO: Implement conv4_2
    dim3 DimGrid4_2(batch, C4_2_channel, (C4_2_size*C4_2_size)/(BLOCKSIZE4*BLOCKSIZE4));
    dim3 DimBlock4_2(BLOCKSIZE4, BLOCKSIZE4, 1);
    w_grid = C4_2_size / BLOCKSIZE4;
    conv<<<DimGrid4_2, DimBlock4_2>>>(d_C4_1_feature_map_padded, d_C4_2_feature_map, d_conv4_2_weight, d_conv4_2_bias, batch, C4_1_size+2*conv4_2_padding_size, C4_1_size+2*conv4_2_padding_size, conv4_2_in_channel, conv4_2_out_channel, conv4_2_kernel_size, w_grid);
    // TODO: Implement relu
    relu<<<DimGrid4_2, DimBlock4_2>>>(d_C4_2_feature_map, batch * C4_2_channel * C4_2_size * C4_2_size, batch, C4_2_channel, C4_2_size, C4_2_size, w_grid);
    // TODO: Implement pad
    pad<<<DimGrid4_2, DimBlock4_2>>>(d_C4_2_feature_map, d_C4_2_feature_map_padded, batch, C4_2_channel, C4_2_size, C4_2_size, conv4_3_padding_size, w_grid);
    // TODO: Implement conv4_3
    dim3 DimGrid4_3(batch, C4_3_channel, (C4_3_size*C4_3_size)/(BLOCKSIZE4*BLOCKSIZE4));
    dim3 DimBlock4_3(BLOCKSIZE4, BLOCKSIZE4, 1);
    w_grid = C4_3_size / BLOCKSIZE4;
    conv<<<DimGrid4_3, DimBlock4_3>>>(d_C4_2_feature_map_padded, d_C4_3_feature_map, d_conv4_3_weight, d_conv4_3_bias, batch, C4_2_size+2*conv4_3_padding_size, C4_2_size+2*conv4_3_padding_size, conv4_3_in_channel, conv4_3_out_channel, conv4_3_kernel_size, w_grid);
    // TODO: Implement relu
    relu<<<DimGrid4_3, DimBlock4_3>>>(d_C4_3_feature_map, batch * C4_3_channel * C4_3_size * C4_3_size, batch, C4_3_channel, C4_3_size, C4_3_size, w_grid);
    // TODO: Implement pool
    dim3 DimGrid4_4(batch, S4_channel, (S4_size*S4_size)/(BLOCKSIZE2*BLOCKSIZE2));
    dim3 DimBlock4_4(BLOCKSIZE2, BLOCKSIZE2, 1);
    w_grid = S4_size / BLOCKSIZE2;
    pool<<<DimGrid4_4, DimBlock4_4>>>(d_C4_3_feature_map, d_S4_feature_map, batch, C4_3_channel, C4_3_size, C4_3_size, w_grid);
    //////////BLOCK 5/////////////////////////////////
    // TODO: Implement pad
    pad<<<DimGrid4_4, DimBlock4_4>>>(d_S4_feature_map, d_S4_feature_map_padded, batch, S4_channel, S4_size, S4_size, conv5_1_padding_size, w_grid);
    // TODO: Implement conv5_1
    dim3 DimGrid5_1(batch, C5_1_channel, (C5_1_size*C5_1_size)/(BLOCKSIZE2*BLOCKSIZE2));
    dim3 DimBlock5_1(BLOCKSIZE2, BLOCKSIZE2, 1);
    w_grid = C5_1_size / BLOCKSIZE2;
    conv<<<DimGrid5_1, DimBlock5_1>>>(d_S4_feature_map_padded, d_C5_1_feature_map, d_conv5_1_weight, d_conv5_1_bias, batch, S4_size+2*conv5_1_padding_size, S4_size+2*conv5_1_padding_size, conv5_1_in_channel, conv5_1_out_channel, conv5_1_kernel_size, w_grid);
    // TODO: Implement relu
    relu<<<DimGrid5_1, DimBlock5_1>>>(d_C5_1_feature_map, batch * C5_1_channel * C5_1_size * C5_1_size, batch, C5_1_channel, C5_1_size, C5_1_size, w_grid);
    // TODO: Implement pad
    pad<<<DimGrid5_1, DimBlock5_1>>>(d_C5_1_feature_map, d_C5_1_feature_map_padded, batch, C5_1_channel, C5_1_size, C5_1_size, conv5_2_padding_size, w_grid);
    // TODO: Implement conv5_2
    dim3 DimGrid5_2(batch, C5_2_channel, (C5_2_size*C5_2_size)/(BLOCKSIZE2*BLOCKSIZE2));
    dim3 DimBlock5_2(BLOCKSIZE2, BLOCKSIZE2, 1);
    w_grid = C5_2_size / BLOCKSIZE2;
    conv<<<DimGrid5_2, DimBlock5_2>>>(d_C5_1_feature_map_padded, d_C5_2_feature_map, d_conv5_2_weight, d_conv5_2_bias, batch, C5_1_size+2*conv5_2_padding_size, C5_1_size+2*conv5_2_padding_size, conv5_2_in_channel, conv5_2_out_channel, conv5_2_kernel_size, w_grid);
    // TODO: Implement relu
    relu<<<DimGrid5_2, DimBlock5_2>>>(d_C5_2_feature_map, batch * C5_2_channel * C5_2_size * C5_2_size, batch, C5_2_channel, C5_2_size, C5_2_size, w_grid);
    // TODO: Implement pad
    pad<<<DimGrid5_2, DimBlock5_2>>>(d_C5_2_feature_map, d_C5_2_feature_map_padded, batch, C5_2_channel, C5_2_size, C5_2_size, conv5_3_padding_size, w_grid);
    // TODO: Implement conv5_3
    dim3 DimGrid5_3(batch, C5_3_channel, (C5_3_size*C5_3_size)/(BLOCKSIZE2*BLOCKSIZE2));
    dim3 DimBlock5_3(BLOCKSIZE2, BLOCKSIZE2, 1);
    w_grid = C5_3_size / BLOCKSIZE2;
    conv<<<DimGrid5_3, DimBlock5_3>>>(d_C5_2_feature_map_padded, d_C5_3_feature_map, d_conv5_3_weight, d_conv5_3_bias, batch, C5_2_size+2*conv5_3_padding_size, C5_2_size+2*conv5_3_padding_size, conv5_3_in_channel, conv5_3_out_channel, conv5_3_kernel_size, w_grid);
    // TODO: Implement relu
    relu<<<DimGrid5_3, DimBlock5_3>>>(d_C5_3_feature_map, batch * C5_3_channel * C5_3_size * C5_3_size, batch, C5_3_channel, C5_3_size, C5_3_size, w_grid);
    // TODO: Implement pool
    dim3 DimGrid5_4(batch, S5_channel, (S5_size*S5_size)/(BLOCKSIZE1*BLOCKSIZE1));
    dim3 DimBlock5_4(BLOCKSIZE1, BLOCKSIZE1, 1);
    w_grid = S5_size / BLOCKSIZE1;
    pool<<<DimGrid5_4, DimBlock5_4>>>(d_C5_3_feature_map, d_S5_feature_map, batch, C5_3_channel, C5_3_size, C5_3_size, w_grid);
    // TODO: Implement fc1
    dim3 DimGridFC(4, 1, 1);
    dim3 DimBlockFC(10, 32, 1);
    fc<<<DimGridFC, DimBlockFC>>>(d_S5_feature_map, d_output, d_fc1_weight, d_fc1_bias, batch, fc1_in_channel, fc1_out_channel);
    // TODO: Implement relu
    /* NOTE: unless you want to make a major change to this class structure, 
    *  you need to write your output to the device memory d_output 
    *  so that classify() can handle the rest.
    */
}

void vgg16_cuda::prepare_device_memory(uint8_t* image) {
  // Alloc Model Parameters

  //////////BLOCK 1/////////////////////////////////
  cudaMalloc((void**)&d_conv1_1_weight,
             sizeof(float) * conv1_1_in_channel * conv1_1_out_channel *
                 conv1_1_kernel_size * conv1_1_kernel_size);
  cudaMalloc((void**)&d_conv1_1_bias, sizeof(float) * conv1_1_out_channel);
  cudaMalloc((void**)&d_conv1_2_weight,
             sizeof(float) * conv1_2_in_channel * conv1_2_out_channel *
                 conv1_2_kernel_size * conv1_2_kernel_size);
  cudaMalloc((void**)&d_conv1_2_bias, sizeof(float) * conv1_2_out_channel);

  //////////BLOCK 2/////////////////////////////////
  cudaMalloc((void**)&d_conv2_1_weight,
             sizeof(float) * conv2_1_in_channel * conv2_1_out_channel *
                 conv2_1_kernel_size * conv2_1_kernel_size);
  cudaMalloc((void**)&d_conv2_1_bias, sizeof(float) * conv2_1_out_channel);
  cudaMalloc((void**)&d_conv2_2_weight,
             sizeof(float) * conv2_2_in_channel * conv2_2_out_channel *
                 conv2_2_kernel_size * conv2_2_kernel_size);
  cudaMalloc((void**)&d_conv2_2_bias, sizeof(float) * conv2_2_out_channel);

  //////////BLOCK 3/////////////////////////////////
  cudaMalloc((void**)&d_conv3_1_weight,
             sizeof(float) * conv3_1_in_channel * conv3_1_out_channel *
                 conv3_1_kernel_size * conv3_1_kernel_size);
  cudaMalloc((void**)&d_conv3_1_bias, sizeof(float) * conv3_1_out_channel);
  cudaMalloc((void**)&d_conv3_2_weight,
             sizeof(float) * conv3_2_in_channel * conv3_2_out_channel *
                 conv3_2_kernel_size * conv3_2_kernel_size);
  cudaMalloc((void**)&d_conv3_2_bias, sizeof(float) * conv3_2_out_channel);
  cudaMalloc((void**)&d_conv3_3_weight,
             sizeof(float) * conv3_3_in_channel * conv3_3_out_channel *
                 conv3_3_kernel_size * conv3_3_kernel_size);
  cudaMalloc((void**)&d_conv3_3_bias, sizeof(float) * conv3_3_out_channel);

  //////////BLOCK 4/////////////////////////////////
  cudaMalloc((void**)&d_conv4_1_weight,
             sizeof(float) * conv4_1_in_channel * conv4_1_out_channel *
                 conv4_1_kernel_size * conv4_1_kernel_size);
  cudaMalloc((void**)&d_conv4_1_bias, sizeof(float) * conv4_1_out_channel);
  cudaMalloc((void**)&d_conv4_2_weight,
             sizeof(float) * conv4_2_in_channel * conv4_2_out_channel *
                 conv4_2_kernel_size * conv4_2_kernel_size);
  cudaMalloc((void**)&d_conv4_2_bias, sizeof(float) * conv4_2_out_channel);
  cudaMalloc((void**)&d_conv4_3_weight,
             sizeof(float) * conv4_3_in_channel * conv4_3_out_channel *
                 conv4_3_kernel_size * conv4_3_kernel_size);
  cudaMalloc((void**)&d_conv4_3_bias, sizeof(float) * conv4_3_out_channel);

  //////////BLOCK 5/////////////////////////////////
  cudaMalloc((void**)&d_conv5_1_weight,
             sizeof(float) * conv5_1_in_channel * conv5_1_out_channel *
                 conv5_1_kernel_size * conv5_1_kernel_size);
  cudaMalloc((void**)&d_conv5_1_bias, sizeof(float) * conv5_1_out_channel);
  cudaMalloc((void**)&d_conv5_2_weight,
             sizeof(float) * conv5_2_in_channel * conv5_2_out_channel *
                 conv5_2_kernel_size * conv5_2_kernel_size);
  cudaMalloc((void**)&d_conv5_2_bias, sizeof(float) * conv5_2_out_channel);
  cudaMalloc((void**)&d_conv5_3_weight,
             sizeof(float) * conv5_3_in_channel * conv5_3_out_channel *
                 conv5_3_kernel_size * conv5_3_kernel_size);
  cudaMalloc((void**)&d_conv5_3_bias, sizeof(float) * conv5_3_out_channel);

  //////////FC 1////////////////////////////////////
  cudaMalloc((void**)&d_fc1_weight,
             sizeof(float) * fc1_in_channel * fc1_out_channel);
  cudaMalloc((void**)&d_fc1_bias, sizeof(float) * fc1_out_channel);

  // Alloc Activations
  cudaMalloc((void**)&d_image,
             sizeof(uint8_t) * batch * input_size * input_size * input_channel);
  cudaMalloc((void**)&d_input,
             sizeof(float) * batch * input_channel * input_size * input_size);

  //////////BLOCK 1/////////////////////////////////
  cudaMalloc((void**)&d_input_padded,
             sizeof(float) * batch * input_channel * (input_size+2*conv1_1_padding_size) * (input_size+2*conv1_1_padding_size));
  cudaMalloc((void**)&d_C1_1_feature_map,
             sizeof(float) * batch * C1_1_channel * C1_1_size * C1_1_size);
  cudaMalloc((void**)&d_C1_1_feature_map_padded,
             sizeof(float) * batch * C1_1_channel * (C1_1_size+2*conv1_2_padding_size) * (C1_1_size+2*conv1_2_padding_size));
  cudaMalloc((void**)&d_C1_2_feature_map,
             sizeof(float) * batch * C1_2_channel * C1_2_size * C1_2_size);
  cudaMalloc((void**)&d_S1_feature_map,
             sizeof(float) * batch * S1_channel * S1_size * S1_size);

  //////////BLOCK 2/////////////////////////////////
  cudaMalloc((void**)&d_S1_feature_map_padded,
             sizeof(float) * batch * S1_channel * (S1_size+2*conv2_1_padding_size) * (S1_size+2*conv2_1_padding_size));
  cudaMalloc((void**)&d_C2_1_feature_map,
             sizeof(float) * batch * C2_1_channel * C2_1_size * C2_1_size);
  cudaMalloc((void**)&d_C2_1_feature_map_padded,
             sizeof(float) * batch * C2_1_channel * (C2_1_size+2*conv2_2_padding_size) * (C2_1_size+2*conv2_2_padding_size));
  cudaMalloc((void**)&d_C2_2_feature_map,
             sizeof(float) * batch * C2_2_channel * C2_2_size * C2_2_size);
  cudaMalloc((void**)&d_S2_feature_map,
             sizeof(float) * batch * S2_channel * S2_size * S2_size);

  //////////BLOCK 3/////////////////////////////////
  cudaMalloc((void**)&d_S2_feature_map_padded,
             sizeof(float) * batch * S2_channel * (S2_size+2*conv3_1_padding_size) * (S2_size+2*conv3_1_padding_size));
  cudaMalloc((void**)&d_C3_1_feature_map,
             sizeof(float) * batch * C3_1_channel * C3_1_size * C3_1_size);
  cudaMalloc((void**)&d_C3_1_feature_map_padded,
             sizeof(float) * batch * C3_1_channel * (C3_1_size+2*conv3_2_padding_size) * (C3_1_size+2*conv3_2_padding_size));
  cudaMalloc((void**)&d_C3_2_feature_map,
             sizeof(float) * batch * C3_2_channel * C3_2_size * C3_2_size);
  cudaMalloc((void**)&d_C3_2_feature_map_padded,
             sizeof(float) * batch * C3_2_channel * (C3_2_size+2*conv3_3_padding_size) * (C3_2_size+2*conv3_3_padding_size));
  cudaMalloc((void**)&d_C3_3_feature_map,
             sizeof(float) * batch * C3_3_channel * C3_3_size * C3_3_size);
  cudaMalloc((void**)&d_S3_feature_map,
             sizeof(float) * batch * S3_channel * S3_size * S3_size);

  //////////BLOCK 4/////////////////////////////////
  cudaMalloc((void**)&d_S3_feature_map_padded,
             sizeof(float) * batch * S3_channel * (S3_size+2*conv4_1_padding_size) * (S3_size+2*conv4_1_padding_size));
  cudaMalloc((void**)&d_C4_1_feature_map,
             sizeof(float) * batch * C4_1_channel * C4_1_size * C4_1_size);
  cudaMalloc((void**)&d_C4_1_feature_map_padded,
             sizeof(float) * batch * C4_1_channel * (C4_1_size+2*conv4_2_padding_size) * (C4_1_size+2*conv4_2_padding_size));
  cudaMalloc((void**)&d_C4_2_feature_map,
             sizeof(float) * batch * C4_2_channel * C4_2_size * C4_2_size);
  cudaMalloc((void**)&d_C4_2_feature_map_padded,
             sizeof(float) * batch * C4_2_channel * (C4_2_size+2*conv4_3_padding_size) * (C4_2_size+2*conv4_3_padding_size));
  cudaMalloc((void**)&d_C4_3_feature_map,
             sizeof(float) * batch * C4_3_channel * C4_3_size * C4_3_size);
  cudaMalloc((void**)&d_S4_feature_map,
             sizeof(float) * batch * S4_channel * S4_size * S4_size);

  //////////BLOCK 5/////////////////////////////////
  cudaMalloc((void**)&d_S4_feature_map_padded,
             sizeof(float) * batch * S4_channel * (S4_size+2*conv5_1_padding_size) * (S4_size+2*conv5_1_padding_size));
  cudaMalloc((void**)&d_C5_1_feature_map,
             sizeof(float) * batch * C5_1_channel * C5_1_size * C5_1_size);
  cudaMalloc((void**)&d_C5_1_feature_map_padded,
             sizeof(float) * batch * C5_1_channel * (C5_1_size+2*conv5_2_padding_size) * (C5_1_size+2*conv5_2_padding_size));
  cudaMalloc((void**)&d_C5_2_feature_map,
             sizeof(float) * batch * C5_2_channel * C5_2_size * C5_2_size);
  cudaMalloc((void**)&d_C5_2_feature_map_padded,
             sizeof(float) * batch * C5_2_channel * (C5_2_size+2*conv5_3_padding_size) * (C5_2_size+2*conv5_3_padding_size));
  cudaMalloc((void**)&d_C5_3_feature_map,
             sizeof(float) * batch * C5_3_channel * C5_3_size * C5_3_size);
  cudaMalloc((void**)&d_S5_feature_map,
             sizeof(float) * batch * S5_channel * S5_size * S5_size);


  cudaMalloc((void**)&d_output, sizeof(float) * batch * output_size);

  // Copy Parameters
  //////////BLOCK 1/////////////////////////////////
  cudaMemcpy(d_conv1_1_weight, conv1_1_weight,
             sizeof(float) * conv1_1_in_channel * conv1_1_out_channel *
                 conv1_1_kernel_size * conv1_1_kernel_size,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv1_1_bias, conv1_1_bias, sizeof(float) * conv1_1_out_channel,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv1_2_weight, conv1_2_weight,
              sizeof(float) * conv1_2_in_channel * conv1_2_out_channel *
                  conv1_2_kernel_size * conv1_2_kernel_size,
              cudaMemcpyHostToDevice);
   cudaMemcpy(d_conv1_2_bias, conv1_2_bias, sizeof(float) * conv1_2_out_channel,
              cudaMemcpyHostToDevice);

  //////////BLOCK 2/////////////////////////////////
  cudaMemcpy(d_conv2_1_weight, conv2_1_weight,
             sizeof(float) * conv2_1_in_channel * conv2_1_out_channel *
                 conv2_1_kernel_size * conv2_1_kernel_size,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv2_1_bias, conv2_1_bias, sizeof(float) * conv2_1_out_channel,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv2_2_weight, conv2_2_weight,
              sizeof(float) * conv2_2_in_channel * conv2_2_out_channel *
                  conv2_2_kernel_size * conv2_2_kernel_size,
              cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv2_2_bias, conv2_2_bias, sizeof(float) * conv2_2_out_channel,
              cudaMemcpyHostToDevice);

  //////////BLOCK 3/////////////////////////////////
  cudaMemcpy(d_conv3_1_weight, conv3_1_weight,
             sizeof(float) * conv3_1_in_channel * conv3_1_out_channel *
                 conv3_1_kernel_size * conv3_1_kernel_size,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv3_1_bias, conv3_1_bias, sizeof(float) * conv3_1_out_channel,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv3_2_weight, conv3_2_weight,
              sizeof(float) * conv3_2_in_channel * conv3_2_out_channel *
                  conv3_2_kernel_size * conv3_2_kernel_size,
              cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv3_2_bias, conv3_2_bias, sizeof(float) * conv3_2_out_channel,
              cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv3_3_weight, conv3_3_weight,
              sizeof(float) * conv3_3_in_channel * conv3_3_out_channel *
                  conv3_3_kernel_size * conv3_3_kernel_size,
              cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv3_3_bias, conv3_3_bias, sizeof(float) * conv3_3_out_channel,
              cudaMemcpyHostToDevice);

  //////////BLOCK 4/////////////////////////////////
  cudaMemcpy(d_conv4_1_weight, conv4_1_weight,
             sizeof(float) * conv4_1_in_channel * conv4_1_out_channel *
                 conv4_1_kernel_size * conv4_1_kernel_size,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv4_1_bias, conv4_1_bias, sizeof(float) * conv4_1_out_channel,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv4_2_weight, conv4_2_weight,
              sizeof(float) * conv4_2_in_channel * conv4_2_out_channel *
                  conv4_2_kernel_size * conv4_2_kernel_size,
              cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv4_2_bias, conv4_2_bias, sizeof(float) * conv4_2_out_channel,
              cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv4_3_weight, conv4_3_weight,
              sizeof(float) * conv4_3_in_channel * conv4_3_out_channel *
                  conv4_3_kernel_size * conv4_3_kernel_size,
              cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv4_3_bias, conv4_3_bias, sizeof(float) * conv4_3_out_channel,
              cudaMemcpyHostToDevice);

  //////////BLOCK 5/////////////////////////////////
  cudaMemcpy(d_conv5_1_weight, conv5_1_weight,
             sizeof(float) * conv5_1_in_channel * conv5_1_out_channel *
                 conv5_1_kernel_size * conv5_1_kernel_size,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv5_1_bias, conv5_1_bias, sizeof(float) * conv5_1_out_channel,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv5_2_weight, conv5_2_weight,
              sizeof(float) * conv5_2_in_channel * conv5_2_out_channel *
                  conv5_2_kernel_size * conv5_2_kernel_size,
              cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv5_2_bias, conv5_2_bias, sizeof(float) * conv5_2_out_channel,
              cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv5_3_weight, conv5_3_weight,
              sizeof(float) * conv5_3_in_channel * conv5_3_out_channel *
                  conv5_3_kernel_size * conv5_3_kernel_size,
              cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv5_3_bias, conv5_3_bias, sizeof(float) * conv5_3_out_channel,
              cudaMemcpyHostToDevice);


  cudaMemcpy(d_fc1_weight, fc1_weight,
             sizeof(float) * fc1_in_channel * fc1_out_channel,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_fc1_bias, fc1_bias, sizeof(float) * fc1_out_channel,
             cudaMemcpyHostToDevice);

  // copy input image
  size_t image_size = batch * input_size * input_size * input_channel;
  cudaMemcpy(d_image, image, image_size * sizeof(uint8_t),
             cudaMemcpyHostToDevice);
}

void vgg16_cuda::classify(int* predict, int batch) {
  // read logits back to cpu
  cudaMemcpy(output, d_output, sizeof(float) * output_size * batch,
             cudaMemcpyDeviceToHost);
  // Softmax
  softmax(output, predict, batch, output_size);
}

vgg16_cuda::~vgg16_cuda() {
  cudaFree(d_conv1_1_weight);   
  cudaFree(d_conv1_2_weight);   
  cudaFree(d_conv2_1_weight);   
  cudaFree(d_conv2_2_weight);  
  cudaFree(d_conv3_1_weight);   
  cudaFree(d_conv3_2_weight);   
  cudaFree(d_conv3_3_weight);   
  cudaFree(d_conv4_1_weight);   
  cudaFree(d_conv4_2_weight);   
  cudaFree(d_conv4_3_weight); 
  cudaFree(d_conv5_1_weight);   
  cudaFree(d_conv5_2_weight);   
  cudaFree(d_conv5_3_weight);   
 
  cudaFree(d_conv1_1_bias);   
  cudaFree(d_conv1_2_bias);   
  cudaFree(d_conv2_1_bias);   
  cudaFree(d_conv2_2_bias);  
  cudaFree(d_conv3_1_bias);   
  cudaFree(d_conv3_2_bias);   
  cudaFree(d_conv3_3_bias);   
  cudaFree(d_conv4_1_bias);   
  cudaFree(d_conv4_2_bias);   
  cudaFree(d_conv4_3_bias); 
  cudaFree(d_conv5_1_bias);   
  cudaFree(d_conv5_2_bias);   
  cudaFree(d_conv5_3_bias);   
   
  cudaFree(d_fc1_weight);     
  cudaFree(d_fc1_bias);        

  cudaFree(d_image);          
  cudaFree(d_input); 

  cudaFree(d_input_padded);          
  cudaFree(d_C1_1_feature_map); 
  cudaFree(d_C1_1_feature_map_padded); 
  cudaFree(d_C1_2_feature_map); 
  cudaFree(d_S1_feature_map); 

  cudaFree(d_S1_feature_map_padded); 
  cudaFree(d_C2_1_feature_map); 
  cudaFree(d_C2_1_feature_map_padded); 
  cudaFree(d_C2_2_feature_map); 
  cudaFree(d_S2_feature_map); 

  cudaFree(d_S2_feature_map_padded); 
  cudaFree(d_C3_1_feature_map); 
  cudaFree(d_C3_1_feature_map_padded); 
  cudaFree(d_C3_2_feature_map); 
  cudaFree(d_C3_2_feature_map_padded); 
  cudaFree(d_C3_3_feature_map); 
  cudaFree(d_S3_feature_map); 

  cudaFree(d_S3_feature_map_padded); 
  cudaFree(d_C4_1_feature_map); 
  cudaFree(d_C4_1_feature_map_padded); 
  cudaFree(d_C4_2_feature_map); 
  cudaFree(d_C4_2_feature_map_padded); 
  cudaFree(d_C4_3_feature_map); 
  cudaFree(d_S4_feature_map); 

  cudaFree(d_S4_feature_map_padded); 
  cudaFree(d_C5_1_feature_map); 
  cudaFree(d_C5_1_feature_map_padded); 
  cudaFree(d_C5_2_feature_map); 
  cudaFree(d_C5_2_feature_map_padded); 
  cudaFree(d_C5_3_feature_map); 
  cudaFree(d_S5_feature_map); 
 
  cudaFree(d_output);       
  cudaFree(d_predict_cuda);   
}
