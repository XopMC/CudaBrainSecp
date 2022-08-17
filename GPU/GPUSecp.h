

#ifndef GPUSECP
#define GPUSECP

#include <vector>
#include <stdint.h>
#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>

#define NAME_HASH_FOLDER "TestHash"
#define NAME_SEED_FOLDER "TestBook"
#define NAME_HASH_BUFFER "merged-sorted-unique-8-byte-hashes"
#define NAME_INPUT_PRIME NAME_SEED_FOLDER "/list_prime"
#define NAME_INPUT_AFFIX NAME_SEED_FOLDER "/list_affix"
#define NAME_FILE_OUTPUT "TEST_OUTPUT"

//CUDA-specific parameters that determine occupancy and thread-count
//Please read more about them in CUDA docs and adjust according to your GPU specs
#define BLOCKS_PER_GRID 30
#define THREADS_PER_BLOCK 256

//Maximum length of each Prime word (+1 because first byte contains word length) 
#define MAX_LEN_WORD_PRIME 20

//Maximum length of each Affix word (+1 because first byte contains word length) 
#define MAX_LEN_WORD_AFFIX 4

//Determines if book Affix words will be added as prefix or as suffix to Prime words.
#define AFFIX_IS_SUFFIX true

//This is how many hashes are in NAME_HASH_FOLDER, Defined as constant to save one register in device kernel
#define COUNT_INPUT_HASH 204

//This is how many prime words are in NAME_INPUT_PRIME file, Defined as constant to save one register in device kernel
#define COUNT_INPUT_PRIME 100

//Combo symbol count - how many unique symbols exist in the COMBO_SYMBOLS array
#define COUNT_COMBO_SYMBOLS 100

//Combo multiplication / buffer size - how many times symbols will be multiplied with each-other (maximum supported size is 8)
#define SIZE_COMBO_MULTI 4

//CPU stack size in bytes that will be allocated to this program - needs to fit GTable / InputBooks / InputHashes 
#define SIZE_CPU_STACK 1024 * 1024 * 1024

//GPU stack size in bytes that will be allocated to each thread - has complex functionality - please read cuda docs about this
#define SIZE_CUDA_STACK 32768

//---------------------------------------------------------------------------------------------------------------------------
// Don't edit configuration below this line
//---------------------------------------------------------------------------------------------------------------------------

#define SIZE_LONG 8            // Each Long is 8 bytes
#define SIZE_HASH160 20        // Each Hash160 is 20 bytes
#define SIZE_PRIV_KEY 32 	   // Length of the private key that is generated from input seed (in bytes)
#define NUM_GTABLE_CHUNK 16    // Number of GTable chunks that are pre-computed and stored in global memory
#define NUM_GTABLE_VALUE 65536 // Number of GTable values per chunk (all possible states) (2 ^ NUM_GTABLE_CHUNK)
#define SIZE_GTABLE_POINT 32   // Each Point in GTable consists of two 32-byte coordinates (X and Y)
#define IDX_CUDA_THREAD ((blockIdx.x * blockDim.x) + threadIdx.x)
#define COUNT_GTABLE_POINTS (NUM_GTABLE_CHUNK * NUM_GTABLE_VALUE)
#define COUNT_CUDA_THREADS (BLOCKS_PER_GRID * THREADS_PER_BLOCK)

//Contains the first element index for each chunk
//Pre-computed to save one multiplication
__constant__ int CHUNK_FIRST_ELEMENT[NUM_GTABLE_CHUNK] = {
  65536*0,  65536*1,  65536*2,  65536*3,
  65536*4,  65536*5,  65536*6,  65536*7,
  65536*8,  65536*9,  65536*10, 65536*11,
  65536*12, 65536*13, 65536*14, 65536*15,
};

//Contains index multiplied by 8
//Pre-computed to save one multiplication
__device__ __constant__ int MULTI_EIGHT[65] = { 0,
    0 + 8,   0 + 16,   0 + 24,   0 + 32,   0 + 40,   0 + 48,   0 + 56,   0 + 64,
   64 + 8,  64 + 16,  64 + 24,  64 + 32,  64 + 40,  64 + 48,  64 + 56,  64 + 64,
  128 + 8, 128 + 16, 128 + 24, 128 + 32, 128 + 40, 128 + 48, 128 + 56, 128 + 64,
  192 + 8, 192 + 16, 192 + 24, 192 + 32, 192 + 40, 192 + 48, 192 + 56, 192 + 64,
  256 + 8, 256 + 16, 256 + 24, 256 + 32, 256 + 40, 256 + 48, 256 + 56, 256 + 64,
  320 + 8, 320 + 16, 320 + 24, 320 + 32, 320 + 40, 320 + 48, 320 + 56, 320 + 64,
  384 + 8, 384 + 16, 384 + 24, 384 + 32, 384 + 40, 384 + 48, 384 + 56, 384 + 64,
  448 + 8, 448 + 16, 448 + 24, 448 + 32, 448 + 40, 448 + 48, 448 + 56, 448 + 64,
};

//Contains combo symbols that are used in the Combo input mode
//Currently has all ASCII keyboard bytes + 5 non-keyboard characters (to have exactly 100 symbols)
__device__ __constant__ uint8_t COMBO_SYMBOLS[COUNT_COMBO_SYMBOLS] = {
	//  0     1     2     3     4     5     6     7     8     9
		0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 
	
	//  space !     "     #     $     %     &     '     (     )     *     +     ,     -     .     /
		0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2A, 0x2B, 0x2C, 0x2D, 0x2E, 0x2F,
	
	//  :     ;     <     =     >     ?     @     [     slash ]     ^     _     `     {     |     }     ~
		0x3A, 0x3B, 0x3C, 0x3D, 0x3E, 0x3F, 0x40, 0x5B, 0x5C, 0x5D, 0x5E, 0x5F, 0x60, 0x7B, 0x7C, 0x7D, 0x7E,
	
	//  A     B     C     D     E     F     G     H     I     J     K     L     M     N     O     P     Q     R     S     T     U     V     W     X     Y     Z
		0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4A, 0x4B, 0x4C, 0x4D, 0x4E, 0x4F, 0x50, 0x51, 0x52, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A, 
	
	//  a     b     c     d     e     f     g     h     i     j     k     l     m     n     o     p     q     r     s     t     u     v     w     x     y     z
		0x61, 0x62, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x6B, 0x6C, 0x6D, 0x6E, 0x6F, 0x70, 0x71, 0x72, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A,
	
	//  NUL   DEL   NEW   TAB   ENTER
		0x00, 0x7F, 0xFF, 0x09, 0x0D
};


#define CudaSafeCall(err) __cudaSafeCall(err, __FILE__, __LINE__)

class GPUSecp
{

public:
	GPUSecp(
		int primeCount, 
		int affixCount,
		const uint8_t * gTableXCPU,
		const uint8_t * gTableYCPU,
		const uint8_t * inputBookPrimeCPU, 
		const uint8_t * inputBookAffixCPU, 
		const uint64_t * inputHashBufferCPU
		);

	void doIterationSecp256k1Books(int iteration);
	void doIterationSecp256k1Combo(int8_t * inputComboCPU);
	void doPrintOutput();
	void doFreeMemory();

private:
	//Input combo buffer, used only in Combo Mode, defines the starting position for each thread
	int8_t * inputComboGPU;

	//GTable buffer containing ~1 million pre-computed points for Secp256k1 point multiplication
	uint8_t * gTableXGPU;
	uint8_t * gTableYGPU;

	//Input buffer that holds Prime wordlist in global memory of the GPU device
	uint8_t * inputBookPrimeGPU;

	//Input buffer that holds Affix wordlist in global memory of the GPU device
	uint8_t * inputBookAffixGPU;

	//Input buffer that holds merged-sorted-unique-8-byte-hashes in global memory of the GPU device
	uint64_t * inputHashBufferGPU;

	//Output buffer containing result of single iteration
	//If seed created a known Hash160 then outputBufferGPU for that affix will be 1
	uint8_t * outputBufferGPU;
	uint8_t * outputBufferCPU;

	//Output buffer containing result of succesful hash160
	//Each hash160 is 20 bytes long, total size is N * 20 bytes
	uint8_t * outputHashesGPU;
	uint8_t * outputHashesCPU;

	//Output buffer containing private keys that were used in succesful hash160
	//Each private key is 32-byte number that was the output of SHA256
	uint8_t * outputPrivKeysGPU;
	uint8_t * outputPrivKeysCPU;
};



#endif // GPUSecpH
