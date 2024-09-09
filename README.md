## This is not my project!! FORK FROM https://github.com/kpurens  

# :brain: CudaBrainSecp
Cuda Secp256k1 Brain Wallet Recovery Tool. <br/>
Performs Secp256k1 Point Multiplication directly on GPU. <br/>
Can be used for efficient brain-wallet or mnemonic-phrase recovery. <br/>

## :notebook_with_decorative_cover: Design
System design can be illustrated with data-flow diagram:
![DiagramV12](https://user-images.githubusercontent.com/8969128/185214693-8632ee9b-b748-4cd5-bf43-77434ea62284.png)

## :heavy_check_mark: When to use CudaBrainSecp
CudaBrainSecp is most useful **when private keys can not be derived from each-other.**<br>
In the example diagram an extra calculation **Sha256 Transform** is done before Secp256k1.<br>
This calculation makes it (nearly) impossible to guess the previous or the next private key.<br>
In such cases CudaBrainSecp is very useful as it performs full Point Multiplication on each thread.<br>
This includes:
- Brain Wallets
- Seed Phrases
- Mnemonic Phrases / [BIP39](https://github.com/bitcoin/bips/blob/master/bip-0039.mediawiki)

## :x: When NOT to use CudaBrainSecp
CudaBrainSecp should be avoided **when it's possible to derive private keys from each-other.** <br>
In such cases CudaBrainSecp is sub-optimal as it would be much quicker to re-use already calculated public keys.<br>
This includes:
- Bitcoin Puzzle (Where you have to simply increment the private key very quickly)<br>
- WIF Recovery (Wallet-Import-Format **is not hashed** and the private key can be re-used / incremented)<br>
- Non-hashed seeds (Each public key can be calculated with just one point-addition from previous public key)<br>

## :spiral_notepad: Requirements
- Linux Operating System
- Nvidia GPU
- Nvidia Display Driver
- Nvidia Cuda Toolkit

## :wrench: Quick Setup
1. Clone or download the repository
2. Find Compute Capability of your Nvidia GPU in [developer.nvidia.com/cuda-gpus](https://developer.nvidia.com/cuda-gpus)
3. Edit Makefile **CCAP** value to your Compute Capability (as an Integer, without decimal places!)
4. Edit Makefile **CUDA** value to your Nvidia Cuda Toolkit directory
5. Open terminal and run `make all` to generate the binaries (don't need administrator privileges)
6. Run `./CudaBrainSecp` to launch the test cases (execution should take about ~3 seconds)
7. You should see output with 112 iterations and ~50 succesfully found hashes + private keys<br /><br />
If you see an error message `system has unsupported display driver / cuda driver combination` or `forward compatibility was attempted on non supporte HW` that means your cuda toolkit is incompatible with your display driver. (can try installing another display driver or another cuda toolkit version + restarting your PC).

## :gear: Configuration
You will need to set-up your IDE (development environment) to link the Cuda Toolkit / dependencies.<br />
I personally use VS Code, it has the `c_cpp_properties.json` configuration file.<br />
In this file it's important to set correct `includePath` and `compilerPath`, otherwise you'll see compiler errors.<br />
<br />
All other configuration is done through `GPU/GPUSecp.h` file.<br />
It contains comments / info about each configurable value, so it should be relatively simple.<br />
You will need to adjust `BLOCKS_PER_GRID` and `THREADS_PER_BLOCK` based on your GPU architecture.<br />
After editing this file you will need to rebuild the project with `make clean` and `make all`<br />

## :books: Modes
CudaBrainSecp includes two operation modes:<br />
- ModeBooks
- ModeCombo

**ModeBooks** <br />
This is the default mode - called from `startSecp256k1ModeBooks` function.<br />
It loads two input wordlists and combines them inside the GPU-kernel.<br />
Current implementation is designed around small Prime wordlist and large Affix wordlist.<br />
In single iteration each thread takes one Affix word and multiplies it with all Prime words.<br />
Prime wordlist is very small because it allows [GPU Global Memory Coalescing](https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/)<br />
Optionally you can change `AFFIX_IS_SUFFIX` to false - this will add Affix words as prefix to Prime words.<br />
File `hash160_book_btc` includes exactly 100 hashes that should be found by combining given wordlists.<br />
Around half of these hashes should be found when affix is prefix, other half when affix is suffix.<br />
You can verify found test hashes using `original_addr` and `original_seed` files.<br />

**ModeCombo** <br />
This is an optional mode - can be enabled by calling `startSecp256k1ModeCombo` method.<br />
It functions as Combination lock / cypher - by doing all possible permutations of given symbols.<br />
Every iteration each thread loads the starting state (`inputComboGPU`) by using the thread index.<br />
And then fully iterates `combo[0]` and `combo[1]` symbols - calculating N<sup>2</sup> public keys, where N = `COUNT_COMBO_SYMBOLS`.<br />
Three configuration parameters for this mode are:
- `COUNT_COMBO_SYMBOLS` - how many unique symbols exist in the `COMBO_SYMBOLS` buffer
- `SIZE_COMBO_MULTI` -  how many times symbols will be multiplied with each-other
- `COMBO_SYMBOLS` - the constant buffer containing combination ASCII symbols / bytes<br />

Currently maximum multiplication count is 8 - additional code needs to be added for larger combinations.<br />
Test Hash160 files include 20 hashes that should be found for each combo multiplication count (4,5,6,7,8).<br />

## :rocket: Real Launch
Once you are satisfied with the test cases / performance, you can setup the data for real-world launch:

1. Select Recovery Addresses:<br />
  1.1 You can select one or more specific addresses that you wish to recover<br />
  1.2 Optionally you can obtain non-zero addresses for currencies of your choice<br />
  1.3 [Blockchair](https://blockchair.com/dumps) is one of the websites that provide such data<br />
2. Use the `addr_to_hash.py` tool to convert chosen addresses into Hash160 files
3. Create folder for holding real Hash160 files (you shouldn't mix them with the test hashes)
4. Move your newly generated Hash160 files into the real Hash160 folder (they will be combined upon launch)
5. Setup input data for specific mode:<br />
  5.1. If using ModeBooks - Create folder with your desired Prime / Affix wordlists<br />
  5.2. If using ModeCombo - Set your desired ASCII combination symbols in the `COMBO_SYMBOLS` buffer<br />
7. Edit `GPU/GPUSecp.h` configuration values that have changed<br />
8. Execute `make clean`, `make all` and `./CudaBrainSecp` to launch the application.

## :memo: Implementation
### Private Key Generation
The only input for ECDSA Public Key calculation is 32-byte number. (also called scalar or private key)<br>
While it should always be randomly generated, it can also be chosen manually in many different ways.<br>
This project focuses on recovering private keys that were chosen from seed phrase and then partially lost / forgotten.<br>
Current implementation combines two words and then performs SHA256 Tranformation to obtain exactly 32 bytes.<br>
You could also use different hashing algorithm (SHA1 / SHA512 / Keccak / Ripemd / etc) or repeat it multiple times.<br>
One side-note about the current implementation of SHA256 function:<br>
It is quite complex, as it takes integers as input and outputs integers as output.<br>
Thus every four bytes will have inverse-order and it can be very confusing to work with.<br>
But it's extremely fast and much faster than than the [Chuda-Hashing-Algos implementation](https://github.com/mochimodev/cuda-hashing-algos/blob/master/sha256.cu)<br>


### Secp256k1 Point Multiplication
The main problem for doing Point Multiplication on GPUs is that it's simply too many calculations.<br>
To multiply Secp256k1 point G with scalar you have to perform ~255 Point-Doubling operations and ~127 Point-Addition operations.<br>
Each Point-Doubling and Point-Addition operation has several expensive `mul_mod` / `add_mod` / `sub_mod` operations.<br>
This is computationally very intensive for one kernel thread to handle. (especially with limiter register / memory size)<br>
However all of the values in Secp256k1 calculations (except for the scalar vector) are constant and never change.<br>
This allows pre-computing some values and passing them to kernel directly. (either through global memory or constants)<br>
In this implementation i chose to pre-compute 16 chunks, 2-bytes each, for a total of 1048576 Points. (taking ~67MB of space)<br>
To calculate the public key we simply need to add these 16 chunks together and perform modular inverse.<br>
Which is around ~20 times more efficient than doing Point Multiplication without pre-computation.<br>
However this comes with a performance penalty of having to frequently access global memory from GPU threads.<br>

### Hash160 / Address
Simple hashing operation can be performed to obtain the actual address from public key.<br>
Bitcoin Wiki has an excellent article & diagram about this: [Bitcoin_addresses](https://en.bitcoin.it/wiki/Technical_background_of_version_1_Bitcoin_addresses)<br>
Basically Hash160 (20-byte hash) is the actual address. (just without the Base58 encoding)<br>
Once the compressed / uncompressed Hash160 is calculated, it is searched in the `inputHashBuffer`.<br>
`inputHashBuffer` is generated by merging / sorting all hash files located in `NAME_HASH_FOLDER` folder.<br>

### Binary Search
In this implementation i use Binary Search to find `uint64_t` (the last 8 bytes) of each hash.<br>
Binary Search was used instead of Bloom Filter for multiple reasons:<br>
- Simplicity (code is smaller and easier to understand / debug)<br>
- Can be verified (If a hash is found - it's guaranteed to be in the inputHashBuffer)<br>
- Should be faster (Bloom Filter requires calculating Murmurhashes and does more calls to globabl memory)<br>

There can still be false positives, since it's only 8 bytes, but the amount of false positives is less than Bloom Filter.<br>

## :bar_chart: Performance
![performanceV3](https://user-images.githubusercontent.com/8969128/184475356-b6cf5359-ae80-4b0d-8207-22fa6d419f78.png)

### Comparison
Performance was tested on two of my personal devices.<br>
Results are quite interesting as Laptop RTX3060 performs slightly faster than Desktop RTX2070.<br>
For simple kernel calculations it should be the other way around - desktop GPUs should perform faster.<br>
However as explained in the next sections - kernel performance is heavily bottlenecked in several areas.<br>
This is helped by Compute Capability 86 which adds useful optimizations over older generation CCAP 75<br>

### Alternatives
At first glance the performance numbers may seem lower than other public Secp256k1 Cuda projects.<br>
However an important difference is that we perform **the full point multiplication on each thread**<br>
Unlike most other projects which derive / increment private key and perform just one addition + inverse.<br>
As mentioned in first chapter - this is intentional, allowing us to solve tasks that no other cuda library can solve.<br>
The only other public library that does full Secp256k1 point multiplication on GPUs is [Hashcat](https://github.com/hashcat/hashcat/blob/80229e34e4f09a1decd4ba1cb73e5f067bdc977c/OpenCL/inc_ecc_secp256k1.cl)<br>
After doing some quick testing it seems that Hashcat's implementation is about 20-30 times slower.<br>
This could be attributed to the lack of pre-computed GTable and not using assambler directives.<br>

## :arrow_double_down: Bottlenecks
### Global Memory Access
This is probably the main bottleneck in the entire application.<br>
Global memory (DRAM) access is required by design - there is no way around it.<br>
CudaSecp needs to access Wordlist / GTable / Hash160 buffers in DRAM from GPU.<br>
Wordlist and Hash160 buffers are very efficient, but the main problem is accessing GTable.<br>
GTable is very large and currently doesn't have coalescing. (since each thread accesses random parts of the table)<br>
And we need to access 16*64 bytes of GTable memory for each seed / public-key, which is a lot.<br>

### Register Pressure
Register Pressure is the other main issue with the current implementation.<br>
There are simply too many variables that take too much space in device registers.<br>
Each part of the `CudaRunSecp256k1` function has some important byte arrays / values.<br>
Not to mention that each Secp256k1 point takes 64 bytes of memory to store.<br>
This severely impacts the occupancy and how many threads are used simultaneously.<br>
Basically the threads have to wait until some device registers are finally available.<br>

## :heavy_plus_sign: Optimizations
```diff
+ Wordlists
Current implementation of CudaSecp relies hevily on wordlists.
I believe the current setup is efficient, as it only loads Affix word once, and Prime words should have coalescing.
However the test setup only has 100 Prime words and with large amount of words you could encounter slower performance.
If you plan on using large wordlists - consider splitting and passing them to GPU in smaller batches.

+ GTable Chunk Size
It's possible to pre-compute different size chunks for the GTable.
Ideally we would have 8 chunks that are 4-bytes each, which would allow doing only 7 point-additions instead of 15.
However that would require 2TB of DRAM space. (maybe if you're reading this in the year 2030 then it's actually possible.)
It's also possibe to go the other way - by doing 32 chunks that are 1-byte each.
That would take up only 512KB of space, but it's still not small enough to be put into device constants.
I also considered doing some hybrid approach (with 22 or 23-bit chunks), but sadly my hardware doesn't have enough DRAM.

+ Register Management
As mentioned in the Bottlenecks section - Register Pressure is one of the main problems.
One possible solution would be to restructure the CudaRunSecp256k1 function to allow better optimization of variables.
Currently the compiler is forced to remember privKey / ctxSha256 for the entire thread lifetime.
Maybe these values could be put into separate functions to allow the compiler to 'forget' them sooner.

+ Memcpy
Currently the _PointMultiSecp256k1 function uses memcpy to copy bytes from global memory.
Initially points were loaded from memory as 4 Long variables (4 * 8 = 32 bytes for x and y coordiates)
However that resulted in 10% slower performance compared to simply doing memcpy on all 32 bytes.
It's possible that there is (or will be) some CUDA-specific alternative to memcpy - that could be worth researching.

+ GPUSha256 Function
Initially the Sha256 function was taken from CUDA Hashing Algorithms Collection.
But for some reason their implementation stores 64 integers in registers to calculate the hash.
That caused significant reduction in performance, since registers are already a bottleneck.
The current Sha256 is very optimal and causes almost no performance loss.
If you plan on using other Hashing function then you may need to do some benchmarking or optimizations.

+ Cuda Generations
Each new generation of CUDA GPUs adds new important optimizations that improve performance.
As noted in performance section - newer generation RTX3060 performs faster than older generation RTX2070.
The main improvements come from larger GPU registers - as threads won't have to wait as much to store their variables.
So using the latest generation hardware & CUDA drivers has a big advantage.
```

## :gift: Support / Donations

If you enjoy this project please consider buying me a coffee.<br>
Any support would be greatly appreciated.<br>

[![coffeeSmol](https://user-images.githubusercontent.com/8969128/185217689-be09e29f-321f-4aaa-a20a-3a867a86d3f8.png)](https://buymeacoffee.com/kpurens)

## :copyright: Credits / References
- [Jean Luc PONS Secp256k1 Math library](https://github.com/JeanLucPons/VanitySearch) (Nice set of Secp256k1 Math operations / functions)
- [CUDA Hashing Algorithms Collection](https://github.com/mochimodev/cuda-hashing-algos) (Contains multiple useful CUDA hashing algorithms)
- [Secp256k1 Calculator](https://github.com/MrMaxweII/Secp256k1-Calculator) (Excellent tool for verifying / testing Secp256k1 calculations)
- [PrivKeys Database](https://privatekeys.pw/) (Tool used to obtain test seeds / addresses) (**DO NOT** enter real private keys here)
- addr_to_hash Python tool designed by Pieter Wuille (Used to convert non-zero addresses to hash files)

## :grey_question: Frequently Asked Questions
```What is the goal this project?```<br>
Main goal was to help people who have forgotten or lost their seed phrase and can't access their funds.<br>
More personal goal was to become familiar with CUDA and Secp256k1 algorithm.<br>

```But shouldn't Seeds / PrivKeys be randomly generated?```<br>
Yes, you should always use randomly generated seeds instead of a memorized phrase. <br>
But this project can still be useful for people who have created / forgotten brain-wallet or seed phrase.<br>

```Why create new project if Cuda Secp256k1 Math Library already exists?``` <br />
Because Secp256k1 Math Library doesn't actually do point multiplication on GPU. <br />
VanitySearch and similar repos all perform point multiplication on CPU and derive private keys from each-other. <br />

```Which crypto-currencies are currently supported in this repository?```<br />
All cryptos that use Secp256k1 public keys. This includes BTC / BCH / ETH / LTC / Doge / Dash and many others.<br />
For Ethereum the final hashing part is slightly different and will need some adjustments in code.<br />

```Can i only do GPU Point Multiplication? I don't need to generate seeds / find hashes.```<br />
Yes, you can just call **_PointMultiSecp256k1** function from your own custom kernel (and pass GTable from global memory)<br />
Thus this repository can also be used as general-purpose Secp256k1 point multiplication system on GPUs.<br />

```Why is CUDA used instead of OpenCL?```<br />
The GPU Math operations designed by Jean Luc PONS are written for CUDA GPUs.<br />
It would be very hard to re-write them in OpenCL and have similar performance level.<br />

```Why isn't Mnemonic Phrase recovery included in this repository?```<br />
Mostly because there are many different implementations of mnemonic phrases / wallets.<br />
Each crypto-exchange can use their own implementation with different salt / key-stretching / mnemonic-wordlists.<br />
However if you have that information then it should be trivial to modify this repository for mnemonic wallet recovery.<br />
