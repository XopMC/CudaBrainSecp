#include <iostream>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <stdio.h>
#include <cstdlib>
#include <string>
#include <fstream>
#include <filesystem>
#include <bits/stdc++.h>
namespace fs = std::filesystem;

using namespace std;

#define LEN_HASH160 20

static void write_file(const uint8_t* DATA, int64_t N, int64_t L, const char* filename)
{
	FILE* fd = fopen(filename, "wb");
	if (fd == NULL) {
		printf("Error: not able to open output file: %s\n", filename);
		exit(1);
	}
	fwrite(DATA, 1, N * L, fd);
	fclose(fd);
}


struct ShorterString {
  bool operator()(const uint64_t& a, const uint64_t& b) const {
    return a < b;
  }
};

void mergeHashes(std::string name_hash_folder, std::string name_hash_buffer)
{
	printf("HashMerge starting \n");

	std::string name_hash_unsorted = "UNSORTED_HASH_FILE";

	printf("HashMerge removing outdated files \n");
	remove( name_hash_unsorted.c_str() );
	remove( name_hash_buffer.c_str() );

	std::ofstream outputStreamUnsorted(name_hash_unsorted, std::ios_base::binary);
	
	printf("HashMerge combining all hash files into %s \n", name_hash_unsorted.c_str());
    for (const auto & entry : fs::directory_iterator(name_hash_folder)) {
		std::cout << entry.path() << std::endl;
		std::ifstream inputStreamEntry(entry.path(), std::ios_base::binary);
		outputStreamUnsorted << inputStreamEntry.rdbuf();
	}

	outputStreamUnsorted.close();

	FILE* fileUnsorted = fopen(name_hash_unsorted.c_str(), "rb");
	if (fileUnsorted == NULL) {
		printf("Error: not able to open input file: %s\n", name_hash_unsorted.c_str());
		exit(1);
	}

	printf("HashMerge reading %s \n", name_hash_unsorted.c_str());
	fseek(fileUnsorted, 0, SEEK_END);
	long fileSizeBytes20 = ftell(fileUnsorted);
	long hashCount20 = fileSizeBytes20 / LEN_HASH160;
	rewind(fileUnsorted);

	printf("HashMerge %s fileSizeBytes: %lu \n", name_hash_unsorted.c_str(), fileSizeBytes20);
	printf("HashMerge %s hashCount: %lu \n", name_hash_unsorted.c_str(), hashCount20);
	
	uint8_t* bufferUnsorted20 = (uint8_t *)malloc(fileSizeBytes20);
	memset(bufferUnsorted20, 0, fileSizeBytes20);

	printf("HashMerge copying data from %s to bufferUnsorted20 \n", name_hash_unsorted.c_str());
	size_t size = fread(bufferUnsorted20, 1, fileSizeBytes20, fileUnsorted);
	fclose(fileUnsorted);
	
	printf("HashMerge initializing HashSet \n");
	set <uint64_t, ShorterString> uniqueHashSet08;

	printf("HashMerge inserting last 8 bytes of each unsorted hash into unique sorted HashSet \n");
	for (int h=0; h < hashCount20; h++) {
		int idx = (h * LEN_HASH160) + 12;
		uint64_t number = 
			static_cast<uint64_t>(bufferUnsorted20[idx + 7]) |
			static_cast<uint64_t>(bufferUnsorted20[idx + 6]) << 8 |
			static_cast<uint64_t>(bufferUnsorted20[idx + 5]) << 16 |
			static_cast<uint64_t>(bufferUnsorted20[idx + 4]) << 24 |
			static_cast<uint64_t>(bufferUnsorted20[idx + 3]) << 32 |
			static_cast<uint64_t>(bufferUnsorted20[idx + 2]) << 40 |
			static_cast<uint64_t>(bufferUnsorted20[idx + 1]) << 48 |
			static_cast<uint64_t>(bufferUnsorted20[idx + 0]) << 56;

		uniqueHashSet08.insert(number);
	}
	
	int uniqueSize = uniqueHashSet08.size();
	printf("HashMerge insertion complete. unique sorted HashSet Size: %d \n", uniqueSize);
	
	printf("HashMerge creating file %s \n", name_hash_buffer.c_str());
	FILE* fileOut = fopen(name_hash_buffer.c_str(), "wb");
	if (fileOut == NULL) {
		printf("Error: not able to open output file: %s\n", name_hash_buffer.c_str());
		exit(1);
	}

	printf("HashMerge copying unique sorted hashes from HashSet to %s \n", name_hash_buffer.c_str());
	set<uint64_t>::iterator iter;
    for (iter = uniqueHashSet08.begin(); iter != uniqueHashSet08.end(); ++iter) {
		uint64_t number = *iter;
		fwrite(&number, sizeof(uint64_t), 1, fileOut);
		//printf("08-byte-Hash: %016llx [%lu]\n", number, number);
	}
	
	printf("HashMerge closing files \n");
	fclose(fileOut);
	remove(name_hash_unsorted.c_str());
	
	printf("HashMerge completed \n");

}
