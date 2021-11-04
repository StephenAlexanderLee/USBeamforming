
// cuda script for f-k migration based on code by rehmanali1994
// Author: Stephen Alexander Lee
// Email: stephenlee32904@gmail.com
// Last Update: 03-16-2021

#include <stdio.h>
#include <cuda.h>
#include <typeinfo>
#include <iostream>
#include <mex.h>
#include <cufft.h>
#include <math.h>
#include <matrix.h>
#include "gpu/mxGPUArray.h"
#include "cuda_fp16.h"

// Define macros for excessively long names
#define pi 3.141592653589793238462643383279502884197169399375105820974f

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      mexPrintf("GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      //if (abort) exit(code);
   }
}


//batched FFT on GPU device
void batchedFFT(cufftComplex* data, cufftHandle plan){

  if (cufftExecC2C(plan, data, data, CUFFT_FORWARD) != CUFFT_SUCCESS){
    fprintf(stderr, "CUFFT error: ExecC2C Forward failed");
  }
  if (cudaDeviceSynchronize() != cudaSuccess){
    fprintf(stderr, "Cuda error: Failed to synchronize\n");
  }
}

//batched IFFT on GPU device
void batchedIFFT(cufftComplex* data, cufftHandle plan){
  //cufftHandle plan;
  //if (cufftPlan1d(&plan, N, CUFFT_C2C, BATCH) != CUFFT_SUCCESS){
  //  fprintf(stderr, "CUFFT error: Plan creation failed");
  //}
  if (cufftExecC2C(plan, data, data, CUFFT_INVERSE) != CUFFT_SUCCESS){
    fprintf(stderr, "CUFFT error: ExecC2C Inverse failed");
  }
  if (cudaDeviceSynchronize() != cudaSuccess){
    fprintf(stderr, "Cuda error: Failed to synchronize\n");
  }
}

__global__ void rfTrim(cufftComplex *RF, int nf, int nx, int ntFFT, float sinA, float pitch, float c, float fs, float t0, float TXangle){
  int index = blockIdx.x*blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int ide = blockIdx.y*blockDim.y + threadIdx.y;

  for (int idf = index; idf < nf; idf += stride){
    if (ide < nx){
      float realRF = RF[idf + ide * ntFFT].x;
      float imagRF = RF[idf + ide * ntFFT].y;
      float dt = (float)((TXangle < 0) ? (nx-1-ide) : -ide)*sinA*pitch / c;
      float f0 = (float)idf*fs/ntFFT;
      float phase = -2 * (dt + t0) * f0;

      RF[idf + ide*ntFFT].x = realRF * cospif(phase) - imagRF * sinpif(phase);
      RF[idf + ide*ntFFT].y = realRF * sinpif(phase) + imagRF * cospif(phase);
    }
  }
}

__global__ void removeEvanescent(cufftComplex *RF, cufftComplex *RFhalf, int nf, int ntFFT, int nxFFT, float c, float fs, float pitch){
  int index = blockIdx.x*blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int idk = blockIdx.y*blockDim.y + threadIdx.y;

  for (int idf = index; idf < nf; idf += stride){
    if (idk < nxFFT){
      float f0 = (float)idf*fs/ntFFT;
      float kx = (float)(idk>nxFFT/2 ? idk - nxFFT : idk)/pitch/nxFFT;
      if (abs(f0)/abs(kx) < c){
        RF[idf + idk*ntFFT].x = 0;
        RF[idf + idk*ntFFT].y = 0;
      }
      RFhalf[idf + idk*nf] = RF[idf + idk*ntFFT];
    }
  }
}
__global__ void stoltmap_fma(cufftComplex *RF, cufftComplex *RFhalf, int band1, int band2, int nf, int ntFFT, int nxFFT, float fs, float pitch, float c, float beta, float v){
  int index = blockIdx.x*blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int idk = blockIdx.y*blockDim.y + threadIdx.y;

  for (int idf = index; idf < nf; idf += stride){
    if (idf > band1 && idf < band2){
      float f = (float)idf*fs/ntFFT;
      float Kx = (float)(idk>nxFFT/2 ? idk - nxFFT : idk)/pitch/nxFFT;
      float fkz = v*sqrt(Kx*Kx + 4 * ((f*f)/(c*c)) / (beta*beta));
      float fkz_idx = (float)(fkz / (fs/ntFFT));
      int fdx = (int)floor(fkz_idx);
      float t = fkz_idx - floor(fkz_idx);
      float t2 = (1-cospif(t))/2;
      float RFreal = fma(t2,RFhalf[(fdx+1) + idk*nf].x,
        fma(-t2, RFhalf[fdx + idk*nf].x, RFhalf[fdx + idk*nf].x));
      float RFimag = fma(t2,RFhalf[(fdx+1) + idk*nf].y,
        fma(-t2, RFhalf[fdx + idk*nf].y, RFhalf[fdx + idk*nf].y));
      RF[idf + idk*ntFFT].x = RFreal * f/fkz;
      RF[idf + idk*ntFFT].y = RFimag * f/fkz;
    }
    else{
      RF[idf + idk*ntFFT].x = 0;
      RF[idf + idk*ntFFT].y = 0;
    }
  }
}

// concatinate negative axial frequencies to fourier domain of migrated solution
__global__ void concatNegAxialFreq(cufftComplex *RF, int ntFFT, int nxFFT, int nf){
  int index = blockIdx.x*blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int idk = blockIdx.y*blockDim.y + threadIdx.y;

  for (int idf = index; idf < ntFFT; idf += stride){
    if (idf < nf){
      // Original Part
      RF[idf + idk*ntFFT].x = RF[idf + idk*ntFFT].x;
      RF[idf + idk*ntFFT].y = RF[idf + idk*ntFFT].y;
    }
    else {
      // bitwise (i % n) = (i & n-1)
      RF[idf + idk*ntFFT].x = RF[((nxFFT - idk) & nxFFT-1) * ntFFT + (ntFFT - idf)].x;
      RF[idf + idk*ntFFT].y = RF[((nxFFT - idk) & nxFFT-1) * ntFFT + (ntFFT - idf)].y;
    }
  }
}

//steering compensation for angle plane waves
__global__ void steerComp(cufftComplex *RFcmp, cufftComplex *RF, int nxFFT, int ntFFT, float pitch, float fs, float c, float gamma,int adx){
  int index = blockIdx.x*blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int idk = blockIdx.y*blockDim.y + threadIdx.y;

  float real, imag;

  for (int idt = index; idt < ntFFT; idt += stride){
    if (idk < nxFFT){
      float realRF = RF[idt + idk * ntFFT].x;
      float imagRF = RF[idt + idk * ntFFT].y;
      float dx = -gamma*idt / fs * c / 2;
      float kx = (float)(idk>nxFFT/2 ? idk - nxFFT : idk)/pitch/nxFFT;
      float phase = -2 * kx * dx;

      real = realRF * cospif(phase) - imagRF * sinpif(phase);
      imag = realRF * sinpif(phase) + imagRF * cospif(phase);

      //compounding
      float k = (float)adx;
      RFcmp[idt + idk * ntFFT].x = (k*RFcmp[idt + idk * ntFFT].x + real)/(k+1);
      RFcmp[idt + idk * ntFFT].y = (k*RFcmp[idt + idk * ntFFT].y + imag)/(k+1);
    }
  }
}

__global__ void allocateRF(cufftComplex *RF, float *rf, int nt, int nx, int ntFFT, int nxFFT, int RF_idx, int Nsamples){
  int index = blockIdx.x*blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int ide = blockIdx.y*blockDim.y + threadIdx.y;

  for (int idt = index; idt < ntFFT; idt += stride){
    if(idt < nt && ide < nx){
      RF[idt + ide*ntFFT].x = rf[RF_idx + idt + ide*Nsamples];
      RF[idt + ide*ntFFT].y = 0;
    }
    else{
      RF[idt + ide*ntFFT].x = 0;
      RF[idt + ide*ntFFT].y = 0;
    }
  }
}
__global__ void copyRF(float *output, cufftComplex *input, int ntFFT, int nt, int nx, int ntshift, int fdx){

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;

  for (int idx = index; idx < nt; idx += stride){
    if (idy < nx){
      output[idx + idy*nt + fdx*(nt*nx)] = input[ntshift + idx + idy*ntFFT].x;
    }
  }
}
// MATLAB gateway FUNCTION
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
  if (nlhs != 1){mexErrMsgTxt("Wrong number of outputs.\n");}
  if (nrhs != 9){mexErrMsgTxt("Wrong number of inputs.\n");}
  mxInitGPU();
  char const * const errId = "parallel:gpu:mexGPUExample:InvalidInput";
  // assign inputs
  mxGPUArray *rf = mxGPUCopyFromMxArray(prhs[0]);
  if (mxGPUGetClassID(rf) != mxSINGLE_CLASS){
    mexErrMsgIdAndTxt(errId, "Input RF must be of type single.");
  }

  const mwSize *dimRF = mxGPUGetDimensions(rf);
  float c = mxGetScalar(prhs[1]);
  float pitch = mxGetScalar(prhs[2]);
  float *angle = (float *)mxGetData(prhs[3]);
  float fs = mxGetScalar(prhs[4]);
  float t0 = mxGetScalar(prhs[5]);
  float *wn = (float *)mxGetData(prhs[6]);
  int na = mxGetScalar(prhs[7]);
  int Nframes = mxGetScalar(prhs[8]);
  mwSize Nsamples = dimRF[0];
  mwSize nt = Nsamples/na/Nframes;
  mwSize nx = dimRF[1];
  //mexPrintf("RF dimensions: [%d %d %d %d]\n",nt,nx,na,Nframes);

  float *d_rf = (float *)mxGPUGetData(rf);

  //zero-padding before FFTs
  // temporal padding: extensive zero-padding for linear interpolation
  int ntshift = (int)(2*ceil(t0*fs/2));
  int ntFFT = 4*nt+ntshift;
  ntFFT = (int) powf(2, (int)log2f(ntFFT)-1);
  // spatial padding: avoid lateral edge effects
  float factor = 1.5f;
  int nxFFT = (int)(2*ceil(factor*nx/2));
  nxFFT = (int) powf(2, (int)log2f(nxFFT));
  int nf = (ntFFT / 2) + 1;
  int band1 = (int)(wn[0]*(nf-1)+1);
  int band2 = (int)(wn[1]*(nf-1)+1);
  //mexPrintf("padded FFT time samples: %d\npadded FFT spatial samples: %d\nntshift: %d\n",ntFFT,nxFFT,ntshift);
  //mexPrintf("bandpass: [%d %d]\n",band1,band2);
  //cufftComplex *RF = (cufftComplex *)malloc(ntFFT*nxFFT*sizeof(cufftComplex));
  //cufftComplex *RFout = (cufftComplex *)malloc(nt*nx*sizeof(float));
  cufftComplex *d_RF, *d_RFcmp, *d_RFhalf;
  gpuErrchk(cudaMalloc(&d_RF, ntFFT*nxFFT*sizeof(cufftComplex)));
  gpuErrchk(cudaMalloc(&d_RFcmp, ntFFT*nxFFT*sizeof(cufftComplex)));
  gpuErrchk(cudaMalloc(&d_RFhalf, nf*nxFFT*sizeof(cufftComplex)));
  //gpuErrchk(cudaMemset(d_RF, 1, ntFFT*nxFFT*sizeof(cufftComplex)));

  dim3 dimBlock(256,2,1);
  dim3 dimGrid((ntFFT + dimBlock.x - 1) / dimBlock.x,
    (nxFFT  + dimBlock.y - 1) / dimBlock.y, 1);
  dim3 dimGridH((nf + dimBlock.x - 1) / dimBlock.x,
    (nxFFT + dimBlock.y - 1) / dimBlock.y, 1);
  dim3 dimGridO((nt + dimBlock.x - 1) / dimBlock.x,
    (nx + dimBlock.y - 1) / dimBlock.y, 1);

  //mexPrintf("Grid: {%d, %d, %d}\n",dimGrid.x, dimGrid.y, dimGrid.z);

  //const int nStreams = 8;
  //cudaStream_t stream[nStreams];
  //for (int i = 0; i < na; i++){
  //  cudaStreamCreate(&stream[i]);
  //}

  cufftHandle FFTplan;
  int rank = 1;
  int FFTn[] = { ntFFT };
  int istride = 1, ostride = 1;
  int idist = ntFFT, odist = ntFFT;
  int inembed[] = { 0 };
  int onembed[] = { 0 };
  int batch = nxFFT;
  cufftPlanMany(&FFTplan, rank, FFTn, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, batch);

  cufftHandle FFTplanT;
  int FFTm[] = { nxFFT };
  istride = ntFFT, ostride = ntFFT;
  idist = 1, odist = 1;
  batch = nf;
  cufftPlanMany(&FFTplanT, rank, FFTm, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, batch);
  //cufftSetStream(FFTplan, stream[1]);

  const mwSize dims[]={nt,nx,Nframes};
  //mxGPUArray *rfout = mxGPUCreateGPUArray(3,dims,mxSINGLE_CLASS,mxREAL,MX_GPU_INITIALIZE_VALUES);
  //float *d_rfout = (float *)(mxGPUGetData(rfout));
  //plhs[0] = mxCreateNumericArray(3,dims,mxSINGLE_CLASS,mxREAL);
  //float *RFout = (float *)mxGetData(plhs[0]);
  mxGPUArray * const RFout = mxGPUCreateGPUArray(3,dims,mxSINGLE_CLASS,mxREAL,MX_GPU_DO_NOT_INITIALIZE);
  float *d_RFout = (float *)mxGPUGetData(RFout);


  for (int fdx = 0; fdx < Nframes; fdx++){
    for (int adx = 0; adx < na; adx++){
      // exploding receptor model (ERM) velocity
      float TXangle = angle[adx];
      float sinA = sinf(TXangle);
      float cosA = cosf(TXangle);
      float v = c / sqrt(1 + cosA + sinA*sinA);
      float beta = (1 + cosA) *sqrt(1 + cosA) / (1 + cosA + sinA*sinA);
      float gamma = sinA / (2-cosA);
      //mexPrintf("angle: %0.4f\nERM velocity: %f\nsinA: %0.6f\ncosA: %0.6f\n",TXangle,v,sinA,cosA);

      int RF_idx = (adx + fdx*na)*nt;
      allocateRF<<<dimGrid,dimBlock>>>(d_RF,d_rf,nt,nx,ntFFT,nxFFT,RF_idx,Nsamples);
      batchedFFT(d_RF,FFTplan);
      rfTrim<<<dimGridH,dimBlock>>>(d_RF,nf,nx,ntFFT,sinA,pitch,c,fs,t0,TXangle);
      batchedFFT(d_RF,FFTplanT);
      removeEvanescent<<<dimGridH,dimBlock>>>(d_RF,d_RFhalf,nf,ntFFT,nxFFT,c,fs,pitch);
      stoltmap_fma<<<dimGridH,dimBlock>>>(d_RF,d_RFhalf,band1,band2,nf,ntFFT,nxFFT,fs,pitch,c,beta,v);
      concatNegAxialFreq<<<dimGrid,dimBlock>>>(d_RF,ntFFT,nxFFT,nf);
      batchedIFFT(d_RF,FFTplan);
      steerComp<<<dimGrid,dimBlock>>>(d_RFcmp,d_RF,nxFFT,ntFFT,pitch,fs,c,gamma,adx);

    }
    batchedIFFT(d_RFcmp,FFTplanT);
    copyRF<<<dimGrid,dimBlock>>>(d_RFout,d_RFcmp, ntFFT, nt, nx, ntshift, fdx);
    //gpuErrchk(cudaPeekAtLastError());
    //gpuErrchk(cudaDeviceSynchronize());
    //cudaMemcpyAsync(&RFout,&d_RF_t,nxFFT*ntFFT*sizeof(float),cudaMemcpyDeviceToHost);
  }

  plhs[0] = mxGPUCreateMxArrayOnGPU(RFout);


  cufftDestroy(FFTplan);
  cufftDestroy(FFTplanT);


  cudaFree(d_RF);
  cudaFree(d_RFcmp);
  cudaFree(d_RFhalf);
  mxGPUDestroyGPUArray(rf);
  mxGPUDestroyGPUArray(RFout);
  //mxGPUDestroyGPUArray(rfout);

}
