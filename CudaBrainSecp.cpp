#include <cstring>
#include <cmath>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <cassert>
#include <pthread.h>
#include <fstream>
#include "GPU/GPUSecp.h"
#include "CPU/SECP256k1.h"
#include "CPU/HashMerge.cpp"
#include "CPU/Combo.cpp"
#include <sys/resource.h>
#include <chrono>

long getFileContent(std::string fileName, std::vector<std::string> &vecOfStrs) {
	long totalSizeBytes = 0;

	std::ifstream in(fileName.c_str());
	if (!in)
	{
		std::cerr << "Can not open the File : " << fileName << std::endl;
		return 0;
	}
	std::string str;
	while (std::getline(in, str))
	{
		vecOfStrs.push_back(str);
		totalSizeBytes += str.size();
	}
	
	in.close();
	return totalSizeBytes;
}

int getBookWordCount(std::string inputName) {
	std::vector<std::string> bookVector;
	getFileContent(inputName, bookVector);
	return bookVector.size();
}

uint8_t* loadInputBook(std::string inputName, int wordMaxLength) {
	std::cout << "loadInputBook " << inputName << " started" << std::endl;
	std::vector<std::string> bookVector;

	getFileContent(inputName, bookVector);
	int bookWordCount = bookVector.size();
	
	uint8_t* inputBookCPU = (uint8_t*)malloc(bookWordCount * wordMaxLength);
	memset(inputBookCPU, 0, bookWordCount * wordMaxLength);

	int idx = 0;
	for (std::string &line : bookVector)
	{
		inputBookCPU[idx] = (uint8_t)line.length();
		memcpy(inputBookCPU + idx + 1, line.c_str(), line.length());
		idx += wordMaxLength;
	}

	std::cout << "loadInputBook " << inputName << " finished! wordCount: " << bookWordCount << std::endl;
	return inputBookCPU;
}

void loadInputHash(uint64_t *inputHashBufferCPU) {
	std::cout << "Loading hash buffer from file: " << NAME_HASH_BUFFER << std::endl;

	FILE *fileSortedHash = fopen(NAME_HASH_BUFFER, "rb");
	if (fileSortedHash == NULL)
	{
		printf("Error: not able to open input file: %s\n", NAME_HASH_BUFFER);
		exit(1);
	}

	fseek(fileSortedHash, 0, SEEK_END);
	long hashBufferSizeBytes = ftell(fileSortedHash);
	long hashCount = hashBufferSizeBytes / SIZE_LONG;
	rewind(fileSortedHash);

	if (hashCount != COUNT_INPUT_HASH) {
		printf("ERROR - Constant COUNT_INPUT_HASH is %d, but the actual hashCount is %lu \n", COUNT_INPUT_HASH, hashCount);
		exit(-1);
	}

	size_t size = fread(inputHashBufferCPU, 1, hashBufferSizeBytes, fileSortedHash);
	fclose(fileSortedHash);

	std::cout << "loadInputHash " << NAME_HASH_BUFFER << " finished!" << std::endl;
	std::cout << "hashCount: " << hashCount << ", hashBufferSizeBytes: " << hashBufferSizeBytes << std::endl;
}

void loadGTable(uint8_t *gTableX, uint8_t *gTableY) {
	std::cout << "loadGTable started" << std::endl;

	Secp256K1 *secp = new Secp256K1();
	secp->Init();

	for (int i = 0; i < NUM_GTABLE_CHUNK; i++)
	{
		for (int j = 0; j < NUM_GTABLE_VALUE - 1; j++)
		{
			int element = (i * NUM_GTABLE_VALUE) + j;
			Point p = secp->GTable[element];
			for (int b = 0; b < 32; b++) {
				gTableX[(element * SIZE_GTABLE_POINT) + b] = p.x.GetByte64(b);
				gTableY[(element * SIZE_GTABLE_POINT) + b] = p.y.GetByte64(b);
			}
		}
	}

	std::cout << "loadGTable finished!" << std::endl;
}

void startSecp256k1ModeBooks(uint8_t * gTableXCPU, uint8_t * gTableYCPU, uint64_t * inputHashBufferCPU) {

	printf("CudaBrainSecp.ModeBooks Starting \n");

	int countPrime = getBookWordCount(NAME_INPUT_PRIME);
	int countAffix = getBookWordCount(NAME_INPUT_AFFIX);

	uint8_t* inputBookPrimeCPU = loadInputBook(NAME_INPUT_PRIME, MAX_LEN_WORD_PRIME);
	uint8_t* inputBookAffixCPU = loadInputBook(NAME_INPUT_AFFIX, MAX_LEN_WORD_AFFIX);

	GPUSecp *gpuSecp = new GPUSecp(
		countPrime,
		countAffix,
		gTableXCPU,
		gTableYCPU,
		inputBookPrimeCPU,
		inputBookAffixCPU,
		inputHashBufferCPU
	);

	long timeTotal = 0;
	long totalCount = (countAffix * countPrime);
	int maxIteration = countAffix / COUNT_CUDA_THREADS;

	for (int iter = 0; iter < maxIteration; iter++) {
		const auto clockIter1 = std::chrono::system_clock::now();
		gpuSecp->doIterationSecp256k1Books(iter);
		const auto clockIter2 = std::chrono::system_clock::now();
		gpuSecp->doPrintOutput();

		long timeIter1 = std::chrono::duration_cast<std::chrono::milliseconds>(clockIter1.time_since_epoch()).count();
		long timeIter2 = std::chrono::duration_cast<std::chrono::milliseconds>(clockIter2.time_since_epoch()).count();
		long iterationDuration = (timeIter2 - timeIter1);
		timeTotal += iterationDuration;

		printf("CudaBrainSecp.ModeBooks Iteration: %d, time: %ld \n", iter, iterationDuration);
	}

	printf("CudaBrainSecp.ModeBooks Complete \n");

	printf("Finished %d iterations in %ld milliseconds \n", maxIteration, timeTotal);

	printf("Total Seed Count: %lu \n", totalCount);

	printf("Seeds Per Second: %0.2lf Million\n", totalCount / (double)(timeTotal * 1000));
}

void startSecp256k1ModeCombo(uint8_t * gTableXCPU, uint8_t * gTableYCPU, uint64_t * inputHashBufferCPU) {

	printf("CudaBrainSecp.ModeCombo Starting \n");

	if (SIZE_COMBO_MULTI < 4 || SIZE_COMBO_MULTI > 8) {
		printf("Currently supported combination sizes are 4, 5, 6, 7 and 8. \n");
		printf("If you wish you can easily add logic for larger combination buffers. \n");
		printf("Simply edit Combo->adjustComboBuffer, GPUHash->_FindComboStart, GPUHash->_SHA256Combo functions. \n");
		exit(-1);
	}

	GPUSecp *gpuSecp = new GPUSecp(
		0,
		0,
		gTableXCPU,
		gTableYCPU,
		NULL,
		NULL,
		inputHashBufferCPU
	);

	long timeTotal = 0;
	long totalComboCount = 1;

	for (int i = 0; i < SIZE_COMBO_MULTI; i++) {
		totalComboCount = totalComboCount * COUNT_COMBO_SYMBOLS;
	}

	long comboPerIteration = (COUNT_CUDA_THREADS * COUNT_COMBO_SYMBOLS * COUNT_COMBO_SYMBOLS);
	long maxIteration = 1 + (totalComboCount / comboPerIteration);
	int8_t comboCPU[SIZE_COMBO_MULTI] = {};

	printf("CudaBrainSecp.ModeCombo maxIteration: %ld \n", maxIteration);
	printf("CudaBrainSecp.ModeCombo totalComboCount: %ld \n", totalComboCount);
	printf("CudaBrainSecp.ModeCombo comboPerIteration: %ld \n", comboPerIteration);

	for (int iter = 0; iter < maxIteration; iter++) {
		printf("CudaBrainSecp.ModeCombo Combination: [");
		for (int i = 0; i < SIZE_COMBO_MULTI; i++) {
			printf("%d ", comboCPU[i]);
		}
		printf("]\n");

		const auto clockIter1 = std::chrono::system_clock::now();
		gpuSecp->doIterationSecp256k1Combo(comboCPU);
		const auto clockIter2 = std::chrono::system_clock::now();
		gpuSecp->doPrintOutput();

		long timeIter1 = std::chrono::duration_cast<std::chrono::milliseconds>(clockIter1.time_since_epoch()).count();
		long timeIter2 = std::chrono::duration_cast<std::chrono::milliseconds>(clockIter2.time_since_epoch()).count();
		long iterationDuration = (timeIter2 - timeIter1);
		timeTotal += iterationDuration;

		printf("CudaBrainSecp.ModeCombo Iteration: %d, time: %ld \n", iter, iterationDuration);

		adjustComboBuffer(comboCPU, COUNT_CUDA_THREADS);
	}

	printf("CudaBrainSecp.ModeCombo Complete \n");

	printf("Finished %ld iterations in %ld milliseconds \n", maxIteration, timeTotal);

	printf("Total Seed Count: %lu \n", totalComboCount);

	printf("Seeds Per Second: %0.2lf Million\n", totalComboCount / (double)(timeTotal * 1000));
}

void increaseStackSizeCPU() {
	const rlim_t cpuStackSize = SIZE_CPU_STACK;
	struct rlimit rl;
	int result;

	printf("Increasing Stack Size to %lu \n", cpuStackSize);

	result = getrlimit(RLIMIT_STACK, &rl);
	if (result == 0)
	{
		if (rl.rlim_cur < cpuStackSize)
		{
			rl.rlim_cur = cpuStackSize;
			result = setrlimit(RLIMIT_STACK, &rl);
			if (result != 0)
			{
				fprintf(stderr, "setrlimit returned result = %d\n", result);
			}
		}
	}
}

int main(int argc, char **argv) {
	printf("CudaBrainSecp Starting \n");

	increaseStackSizeCPU();

	mergeHashes(NAME_HASH_FOLDER, NAME_HASH_BUFFER);

	uint8_t* gTableXCPU = new uint8_t[COUNT_GTABLE_POINTS * SIZE_GTABLE_POINT];
	uint8_t* gTableYCPU = new uint8_t[COUNT_GTABLE_POINTS * SIZE_GTABLE_POINT];

	loadGTable(gTableXCPU, gTableYCPU);

	printf("%u \n",gTableXCPU[6]);
	printf("%u \n",gTableYCPU[6]);

	uint64_t* inputHashBufferCPU = new uint64_t[COUNT_INPUT_HASH];

	loadInputHash(inputHashBufferCPU);

	startSecp256k1ModeBooks(gTableXCPU, gTableYCPU, inputHashBufferCPU);
	
	//startSecp256k1ModeCombo(gTableXCPU, gTableYCPU, inputHashBufferCPU);

	free(gTableXCPU);
	free(gTableYCPU);
	free(inputHashBufferCPU);

	printf("CudaBrainSecp Complete \n");
	return 0;
}
