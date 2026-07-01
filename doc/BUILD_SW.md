# Build SYCL-MLIR

Set .gitconfig
```bash
[url "repo:"]
	insteadOf = https://github.com/
```

```bash
source /usr/sw/swllvm/setenv-18.sh
python /home/oyyc/sycl-mlir/buildbot/configure.py --build-compiler-c /home/Tools/gcc/11.2.0/bin/gcc --build-compiler-cpp /home/Tools/gcc/11.2.0/bin/g++ --cmake-gen "Unix Makefiles" --cmake-opt=-DSYCL_LIBDEVICE_GCC_TOOLCHAIN=/home/Tools/gcc/11.2.0 -o /home/oyyc/sycl-mlir/build
python /home/oyyc/sycl-mlir/buildbot/compile.py -o /home/oyyc/sycl-mlir/build -j16
```

# Compile SYCL Code

```bash
source /usr/sw/swllvm/setenv-18.sh
/home/oyyc/SWCLCC/install/bin/swsyclmlir --target=athread -fsycl x.cpp -o a.out
```

# Run SYCL Code

```bash
export LD_LIBRARY_PATH=/home/export/online1/mdt00/shisuan/swyjs/oyyc/lib:$LD_LIBRARY_PATH
bsub -J sycl -o output -q q_share -b -n 1 -cgsp 64 -share_size 13000 -priv_size 16 -host_stack 1024 -cache_size 128 a.out
```
