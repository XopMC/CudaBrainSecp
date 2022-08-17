
SRC = CudaBrainSecp.cpp \
      CPU/Point.cpp \
      CPU/Int.cpp \
      CPU/IntMod.cpp \
      CPU/SECP256K1.cpp

OBJDIR = obj

OBJET = $(addprefix $(OBJDIR)/, \
		GPU/GPUSecp.o \
		CPU/Point.o \
		CPU/Int.o \
		CPU/IntMod.o \
		CPU/SECP256K1.o \
        CudaBrainSecp.o \
)

CCAP      = 86
CUDA	  = /usr/local/cuda-11.7
CXX       = g++
CXXCUDA   = /usr/bin/g++
CXXFLAGS  = -DWITHGPU -m64 -mssse3 -Wno-write-strings -O2 -I. -I$(CUDA)/include
LFLAGS    = -lgmp -lpthread -L$(CUDA)/lib64 -lcudart
NVCC      = $(CUDA)/bin/nvcc

#--------------------------------------------------------------------

$(OBJDIR)/GPU/GPUSecp.o: GPU/GPUSecp.cu
	$(NVCC) --compile --compiler-options -fPIC -ccbin $(CXXCUDA) -m64 -O2 -I$(CUDA)/include -gencode=arch=compute_$(CCAP),code=sm_$(CCAP) -o $(OBJDIR)/GPU/GPUSecp.o -c GPU/GPUSecp.cu

$(OBJDIR)/%.o : %.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<
	
$(OBJDIR)/CPU/%.o : %.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<

all: CudaBrainSecp

CudaBrainSecp: $(OBJET)
	@echo Making CudaBrainSecp...
	$(CXX) $(OBJET) $(LFLAGS) -o CudaBrainSecp

$(OBJET): | $(OBJDIR) $(OBJDIR)/GPU $(OBJDIR)/CPU

$(OBJDIR):
	mkdir -p $(OBJDIR)

$(OBJDIR)/GPU: $(OBJDIR)
	cd $(OBJDIR) &&	mkdir -p GPU
	
$(OBJDIR)/CPU: $(OBJDIR)
	cd $(OBJDIR) &&	mkdir -p CPU

clean:
	@echo Cleaning...
	@rm -rf obj || true

