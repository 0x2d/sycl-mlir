# Build SYCL-MLIR

Set ```.gitconfig```

```bash
[url "repo@10.208.130.147:github.com/"]
	insteadOf = https://github.com/
```

由于OpenCL-Headers最新Commit与sycl-mlir不兼容，因此需要手动选择旧版本，修改 OpenCL-Headers.git/refs/heads/main 内容为 8275634cf9ec31b6484c2e6be756237cb583999d

```bash
python /home/oyyc/sycl-mlir/buildbot/configure.py -o /home/oyyc/sycl-mlir/build
python /home/oyyc/sycl-mlir/buildbot/compile.py -o /home/oyyc/sycl-mlir/build -j16
```

# Compile SYCL Code

```bash
export PATH=/home/oyyc/sycl-mlir/build/install/bin:$PATH
export LD_LIBRARY_PATH=/home/oyyc/sycl-mlir/build/install/lib:/opt/intel/oneapi/tbb/2022.0/lib:$LD_LIBRARY_PATH
clang++ -fsycl -fsycl-targets=spir64-unknown-unknown-syclmlir x.cpp -o a.out
```

# Run SYCL Code

```bash
export PATH=/home/oyyc/sycl-mlir/build/install/bin:$PATH
export LD_LIBRARY_PATH=/home/oyyc/sycl-mlir/build/install/lib:/opt/intel/oneapi/tbb/2022.0/lib:$LD_LIBRARY_PATH
./a.out
```
