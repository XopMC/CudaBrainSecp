
#include "GPUSecp.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "GPUMath.h"
#include "GPUHash.h"

using namespace std;

inline void __cudaSafeCall(cudaError err, const char *file, const int line)
{
  if (cudaSuccess != err)
  {
    printf("cudaSafeCall() failed at %s:%i : %s\n", file, line, cudaGetErrorString(err));
    fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n", file, line, cudaGetErrorString(err));
    exit(-1);
  }
}

GPUSecp::GPUSecp(
  	int countPrime, 
		int countAffix,
    const uint8_t *gTableXCPU,
    const uint8_t *gTableYCPU,
		const uint8_t * inputBookPrimeCPU, 
		const uint8_t * inputBookAffixCPU, 
    const uint64_t *inputHashBufferCPU
    )
{
  printf("GPUSecp Starting\n");

  int gpuId = 0; // FOR MULTIPLE GPUS EDIT THIS
  CudaSafeCall(cudaSetDevice(gpuId));

  cudaDeviceProp deviceProp;
  CudaSafeCall(cudaGetDeviceProperties(&deviceProp, gpuId));

  printf("GPU.gpuId: #%d \n", gpuId);
  printf("GPU.deviceProp.name: %s \n", deviceProp.name);
  printf("GPU.multiProcessorCount: %d \n", deviceProp.multiProcessorCount);
  printf("GPU.BLOCKS_PER_GRID: %d \n", BLOCKS_PER_GRID);
  printf("GPU.THREADS_PER_BLOCK: %d \n", THREADS_PER_BLOCK);
  printf("GPU.CUDA_THREAD_COUNT: %d \n", COUNT_CUDA_THREADS);
  printf("GPU.countHash160: %d \n", COUNT_INPUT_HASH);
  printf("GPU.countPrime: %d \n", countPrime);
  printf("GPU.countAffix: %d \n", countAffix);

  if (countPrime > 0 && countPrime != COUNT_INPUT_PRIME) {
    printf("ERROR: countPrime must be equal to COUNT_INPUT_PRIME \n");
    printf("Please edit GPUSecp.h configuration and set COUNT_INPUT_PRIME to %d \n", countPrime);
    exit(-1);
  }

  CudaSafeCall(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
  CudaSafeCall(cudaDeviceSetLimit(cudaLimitStackSize, SIZE_CUDA_STACK));

  size_t limit = 0;
  cudaDeviceGetLimit(&limit, cudaLimitStackSize);
  printf("cudaLimitStackSize: %u\n", (unsigned)limit);
  cudaDeviceGetLimit(&limit, cudaLimitPrintfFifoSize);
  printf("cudaLimitPrintfFifoSize: %u\n", (unsigned)limit);
  cudaDeviceGetLimit(&limit, cudaLimitMallocHeapSize);
  printf("cudaLimitMallocHeapSize: %u\n", (unsigned)limit);

  if (countPrime > 0) {
    printf("Allocating inputBookPrime \n");
    CudaSafeCall(cudaMalloc((void **)&inputBookPrimeGPU, countPrime * MAX_LEN_WORD_PRIME));
    CudaSafeCall(cudaMemcpy(inputBookPrimeGPU, inputBookPrimeCPU, countPrime * MAX_LEN_WORD_PRIME, cudaMemcpyHostToDevice));

    printf("Allocating inputBookAffix \n");
    CudaSafeCall(cudaMalloc((void **)&inputBookAffixGPU, countAffix * MAX_LEN_WORD_AFFIX));
    CudaSafeCall(cudaMemcpy(inputBookAffixGPU, inputBookAffixCPU, countAffix * MAX_LEN_WORD_AFFIX, cudaMemcpyHostToDevice));
  } else {
    printf("Allocating inputCombo buffer \n");
    CudaSafeCall(cudaMalloc((void **)&inputComboGPU, SIZE_COMBO_MULTI));
  }
  
  printf("Allocating inputHashBuffer \n");
  CudaSafeCall(cudaMalloc((void **)&inputHashBufferGPU, COUNT_INPUT_HASH * SIZE_LONG));
  CudaSafeCall(cudaMemcpy(inputHashBufferGPU, inputHashBufferCPU, COUNT_INPUT_HASH * SIZE_LONG, cudaMemcpyHostToDevice));

  printf("Allocating gTableX \n");
  CudaSafeCall(cudaMalloc((void **)&gTableXGPU, COUNT_GTABLE_POINTS * SIZE_GTABLE_POINT));
  CudaSafeCall(cudaMemset(gTableXGPU, 0, COUNT_GTABLE_POINTS * SIZE_GTABLE_POINT));
  CudaSafeCall(cudaMemcpy(gTableXGPU, gTableXCPU, COUNT_GTABLE_POINTS * SIZE_GTABLE_POINT, cudaMemcpyHostToDevice));

  printf("Allocating gTableY \n");
  CudaSafeCall(cudaMalloc((void **)&gTableYGPU, COUNT_GTABLE_POINTS * SIZE_GTABLE_POINT));
  CudaSafeCall(cudaMemset(gTableYGPU, 0, COUNT_GTABLE_POINTS * SIZE_GTABLE_POINT));
  CudaSafeCall(cudaMemcpy(gTableYGPU, gTableYCPU, COUNT_GTABLE_POINTS * SIZE_GTABLE_POINT, cudaMemcpyHostToDevice));

  printf("Allocating outputBuffer \n");
  CudaSafeCall(cudaMalloc((void **)&outputBufferGPU, COUNT_CUDA_THREADS));
  CudaSafeCall(cudaHostAlloc(&outputBufferCPU, COUNT_CUDA_THREADS, cudaHostAllocWriteCombined | cudaHostAllocMapped));

  printf("Allocating outputHashes \n");
  CudaSafeCall(cudaMalloc((void **)&outputHashesGPU, COUNT_CUDA_THREADS * SIZE_HASH160));
  CudaSafeCall(cudaHostAlloc(&outputHashesCPU, COUNT_CUDA_THREADS * SIZE_HASH160, cudaHostAllocWriteCombined | cudaHostAllocMapped));

  printf("Allocating outputPrivKeys \n");
  CudaSafeCall(cudaMalloc((void **)&outputPrivKeysGPU, COUNT_CUDA_THREADS * SIZE_PRIV_KEY));
  CudaSafeCall(cudaHostAlloc(&outputPrivKeysCPU, COUNT_CUDA_THREADS * SIZE_PRIV_KEY, cudaHostAllocWriteCombined | cudaHostAllocMapped));

  printf("Allocation Complete \n");
  CudaSafeCall(cudaGetLastError());
}

//Cuda Secp256k1 Point Multiplication
//Takes 32-byte privKey + gTable and outputs 64-byte public key [qx,qy]
__device__ void _PointMultiSecp256k1(uint64_t *qx, uint64_t *qy, uint16_t *privKey, uint8_t *gTableX, uint8_t *gTableY) {

    int chunk = 0;
    uint64_t qz[5] = {1, 0, 0, 0, 0};

    //Find the first non-zero point [qx,qy]
    for (; chunk < NUM_GTABLE_CHUNK; chunk++) {
      if (privKey[chunk] > 0) {
        int index = (CHUNK_FIRST_ELEMENT[chunk] + (privKey[chunk] - 1)) * SIZE_GTABLE_POINT;
        memcpy(qx, gTableX + index, SIZE_GTABLE_POINT);
        memcpy(qy, gTableY + index, SIZE_GTABLE_POINT);
        chunk++;
        break;
      }
    }

    //Add the remaining chunks together
    for (; chunk < NUM_GTABLE_CHUNK; chunk++) {
      if (privKey[chunk] > 0) {
        uint64_t gx[4];
        uint64_t gy[4];

        int index = (CHUNK_FIRST_ELEMENT[chunk] + (privKey[chunk] - 1)) * SIZE_GTABLE_POINT;
        
        memcpy(gx, gTableX + index, SIZE_GTABLE_POINT);
        memcpy(gy, gTableY + index, SIZE_GTABLE_POINT);

        _PointAddSecp256k1(qx, qy, qz, gx, gy);
      }
    }

    //Performing modular inverse on qz to obtain the public key [qx,qy]
    _ModInv(qz);
    _ModMult(qx, qz);
    _ModMult(qy, qz);
}


//GPU kernel function for computing Secp256k1 public key from input books
__global__ void
CudaRunSecp256k1Books(
    int iteration, uint8_t * gTableXGPU, uint8_t * gTableYGPU,
    uint8_t *inputBookPrimeGPU, uint8_t *inputBookAffixGPU, uint64_t *inputHashBufferGPU,
    uint8_t *outputBufferGPU, uint8_t *outputHashesGPU, uint8_t *outputPrivKeysGPU) {

  //Load affix word from global memory based on thread index
  uint32_t offsetAffix = (COUNT_CUDA_THREADS * iteration * MAX_LEN_WORD_AFFIX) + (IDX_CUDA_THREAD * MAX_LEN_WORD_AFFIX);
  uint8_t wordAffix[MAX_LEN_WORD_AFFIX];
  uint8_t privKey[SIZE_PRIV_KEY];
  uint8_t sizeAffix = inputBookAffixGPU[offsetAffix];
  for (uint8_t i = 0; i < sizeAffix; i++) {
    wordAffix[i] = inputBookAffixGPU[offsetAffix + i + 1];
  }
  
  for (int idxPrime = 0; idxPrime < COUNT_INPUT_PRIME; idxPrime++) {
  
  _SHA256Books((uint32_t *)privKey, inputBookPrimeGPU, wordAffix, sizeAffix, idxPrime);

    uint64_t qx[4];
    uint64_t qy[4];

    _PointMultiSecp256k1(qx, qy, (uint16_t *)privKey, gTableXGPU, gTableYGPU);

    uint8_t hash160[SIZE_HASH160];
    uint64_t hash160Last8Bytes;

    _GetHash160Comp(qx, (uint8_t)(qy[0] & 1), hash160);
    GET_HASH_LAST_8_BYTES(hash160Last8Bytes, hash160);

    if (_BinarySearch(inputHashBufferGPU, COUNT_INPUT_HASH, hash160Last8Bytes) >= 0) {
      int idxCudaThread = IDX_CUDA_THREAD;
      outputBufferGPU[idxCudaThread] += 1;
      for (int i = 0; i < SIZE_HASH160; i++) {
        outputHashesGPU[(idxCudaThread * SIZE_HASH160) + i] = hash160[i];
      }
      for (int i = 0; i < SIZE_PRIV_KEY; i++) {
        outputPrivKeysGPU[(idxCudaThread * SIZE_PRIV_KEY) + i] = privKey[i];
      }
    }
    
    _GetHash160(qx, qy, hash160);
    GET_HASH_LAST_8_BYTES(hash160Last8Bytes, hash160);

    if (_BinarySearch(inputHashBufferGPU, COUNT_INPUT_HASH, hash160Last8Bytes) >= 0) {
      int idxCudaThread = IDX_CUDA_THREAD;
      outputBufferGPU[idxCudaThread] += 1;
      for (int i = 0; i < SIZE_HASH160; i++) {
        outputHashesGPU[(idxCudaThread * SIZE_HASH160) + i] = hash160[i];
      }
      for (int i = 0; i < SIZE_PRIV_KEY; i++) {
        outputPrivKeysGPU[(idxCudaThread * SIZE_PRIV_KEY) + i] = privKey[i];
      }
    }
  }
}

__global__ void CudaRunSecp256k1Combo(
    int8_t * inputComboGPU, uint8_t * gTableXGPU, uint8_t * gTableYGPU, uint64_t *inputHashBufferGPU,
    uint8_t *outputBufferGPU, uint8_t *outputHashesGPU, uint8_t *outputPrivKeysGPU) {

  int8_t combo[SIZE_COMBO_MULTI] = {};
  _FindComboStart(inputComboGPU, combo);

  for (combo[0] = 0; combo[0] < COUNT_COMBO_SYMBOLS; combo[0]++) {
    for (combo[1] = 0; combo[1] < COUNT_COMBO_SYMBOLS; combo[1]++) {

      uint8_t privKey[SIZE_PRIV_KEY];
      _SHA256Combo((uint32_t *)privKey, combo);

      uint64_t qx[4];
      uint64_t qy[4];

      _PointMultiSecp256k1(qx, qy, (uint16_t *)privKey, gTableXGPU, gTableYGPU);

      uint8_t hash160[SIZE_HASH160];
      uint64_t hash160Last8Bytes;

      _GetHash160Comp(qx, (uint8_t)(qy[0] & 1), hash160);
      GET_HASH_LAST_8_BYTES(hash160Last8Bytes, hash160);

      if (_BinarySearch(inputHashBufferGPU, COUNT_INPUT_HASH, hash160Last8Bytes) >= 0) {
        int idxCudaThread = IDX_CUDA_THREAD;
        outputBufferGPU[idxCudaThread] += 1;
        for (int i = 0; i < SIZE_HASH160; i++) {
          outputHashesGPU[(idxCudaThread * SIZE_HASH160) + i] = hash160[i];
        }
        for (int i = 0; i < SIZE_PRIV_KEY; i++) {
          outputPrivKeysGPU[(idxCudaThread * SIZE_PRIV_KEY) + i] = privKey[i];
        }
      }
      
      _GetHash160(qx, qy, hash160);
      GET_HASH_LAST_8_BYTES(hash160Last8Bytes, hash160);

      if (_BinarySearch(inputHashBufferGPU, COUNT_INPUT_HASH, hash160Last8Bytes) >= 0) {
        int idxCudaThread = IDX_CUDA_THREAD;
        outputBufferGPU[idxCudaThread] += 1;
        for (int i = 0; i < SIZE_HASH160; i++) {
          outputHashesGPU[(idxCudaThread * SIZE_HASH160) + i] = hash160[i];
        }
        for (int i = 0; i < SIZE_PRIV_KEY; i++) {
          outputPrivKeysGPU[(idxCudaThread * SIZE_PRIV_KEY) + i] = privKey[i];
        }
      }
    }
  }
}


void GPUSecp::doIterationSecp256k1Books(int iteration) {
  CudaSafeCall(cudaMemset(outputBufferGPU, 0, COUNT_CUDA_THREADS));
  CudaSafeCall(cudaMemset(outputHashesGPU, 0, COUNT_CUDA_THREADS * SIZE_HASH160));
  CudaSafeCall(cudaMemset(outputPrivKeysGPU, 0, COUNT_CUDA_THREADS * SIZE_PRIV_KEY));

  CudaRunSecp256k1Books<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(
    iteration, gTableXGPU, gTableYGPU,
    inputBookPrimeGPU, inputBookAffixGPU, inputHashBufferGPU,
    outputBufferGPU, outputHashesGPU, outputPrivKeysGPU);

  CudaSafeCall(cudaMemcpy(outputBufferCPU, outputBufferGPU, COUNT_CUDA_THREADS, cudaMemcpyDeviceToHost));
  CudaSafeCall(cudaMemcpy(outputHashesCPU, outputHashesGPU, COUNT_CUDA_THREADS * SIZE_HASH160, cudaMemcpyDeviceToHost));
  CudaSafeCall(cudaMemcpy(outputPrivKeysCPU, outputPrivKeysGPU, COUNT_CUDA_THREADS * SIZE_PRIV_KEY, cudaMemcpyDeviceToHost));
  CudaSafeCall(cudaGetLastError());
}

void GPUSecp::doIterationSecp256k1Combo(int8_t * inputComboCPU) {
  CudaSafeCall(cudaMemset(outputBufferGPU, 0, COUNT_CUDA_THREADS));
  CudaSafeCall(cudaMemset(outputHashesGPU, 0, COUNT_CUDA_THREADS * SIZE_HASH160));
  CudaSafeCall(cudaMemset(outputPrivKeysGPU, 0, COUNT_CUDA_THREADS * SIZE_PRIV_KEY));

  CudaSafeCall(cudaMemcpy(inputComboGPU, inputComboCPU, SIZE_COMBO_MULTI, cudaMemcpyHostToDevice));
  CudaSafeCall(cudaGetLastError());

  CudaRunSecp256k1Combo<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(
    inputComboGPU, gTableXGPU, gTableYGPU, inputHashBufferGPU,
    outputBufferGPU, outputHashesGPU, outputPrivKeysGPU);

  CudaSafeCall(cudaMemcpy(outputBufferCPU, outputBufferGPU, COUNT_CUDA_THREADS, cudaMemcpyDeviceToHost));
  CudaSafeCall(cudaMemcpy(outputHashesCPU, outputHashesGPU, COUNT_CUDA_THREADS * SIZE_HASH160, cudaMemcpyDeviceToHost));
  CudaSafeCall(cudaMemcpy(outputPrivKeysCPU, outputPrivKeysGPU, COUNT_CUDA_THREADS * SIZE_PRIV_KEY, cudaMemcpyDeviceToHost));
  CudaSafeCall(cudaGetLastError());
}

void GPUSecp::doPrintOutput() {
  for (int idxThread = 0; idxThread < COUNT_CUDA_THREADS; idxThread++) {
    if (outputBufferCPU[idxThread] > 0) {
      printf("HASH: ");
      for (int h = 0; h < SIZE_HASH160; h++) {
				printf("%02X", outputHashesCPU[(idxThread * SIZE_HASH160) + h]);
			}
      printf(" PRIV: ");
      for (int k = 0; k < SIZE_PRIV_KEY; k++) {
				printf("%02X", outputPrivKeysCPU[(idxThread * SIZE_PRIV_KEY) + k]);
			}
      printf("\n");

      FILE *file = stdout;
      file = fopen(NAME_FILE_OUTPUT, "a");
      if (file != NULL) {
        fprintf(file, "HASH: ");
        for (int h = 0; h < SIZE_HASH160; h++) {
          fprintf(file, "%02X", outputHashesCPU[(idxThread * SIZE_HASH160) + h]);
        }
        fprintf(file, " PRIV: ");
        for (int k = 0; k < SIZE_PRIV_KEY; k++) {
          fprintf(file, "%02X", outputPrivKeysCPU[(idxThread * SIZE_PRIV_KEY) + k]);
        }
        fprintf(file, "\n");
        fclose(file);
      }
    }
  }
}

void GPUSecp::doFreeMemory() {
  printf("\nGPUSecp Freeing memory... ");

  CudaSafeCall(cudaFree(inputComboGPU));
  CudaSafeCall(cudaFree(inputBookPrimeGPU));
  CudaSafeCall(cudaFree(inputBookAffixGPU));
  CudaSafeCall(cudaFree(inputHashBufferGPU));

  CudaSafeCall(cudaFree(gTableXGPU));
  CudaSafeCall(cudaFree(gTableYGPU));

  CudaSafeCall(cudaFreeHost(outputBufferCPU));
  CudaSafeCall(cudaFree(outputBufferGPU));

  CudaSafeCall(cudaFreeHost(outputHashesCPU));
  CudaSafeCall(cudaFree(outputHashesGPU));

  CudaSafeCall(cudaFreeHost(outputPrivKeysCPU));
  CudaSafeCall(cudaFree(outputPrivKeysGPU));

  printf("Done \n");
}