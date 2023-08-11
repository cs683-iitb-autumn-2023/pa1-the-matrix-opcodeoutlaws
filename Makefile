BUILD=build
CC=g++

blocking: build
	g++ pa1-the-matrix.c -D OPTIMIZE_BLOCKING -o $(BUILD)/$@

prefetch: build
	g++ pa1-the-matrix.c -D OPTIMIZE_PREFETCH -o $(BUILD)/$@

simd: build
	g++ pa1-the-matrix.c -D OPTIMIZE_SIMD -o $(BUILD)/$@

blocking-prefetch: build
	g++ pa1-the-matrix.c -D OPTIMIZE_BLOCKING_PREFETCH -o $(BUILD)/$@

blocking-simd: build
	g++ pa1-the-matrix.c -D OPTIMIZE_BLOCKING_SIMD -o $(BUILD)/$@

simd-prefetch: build
	g++ pa1-the-matrix.c -D OPTIMIZE_SIMD_PREFETCH -o $(BUILD)/$@

blocking-simd-prefetch: build
	g++ pa1-the-matrix.c -D OPTIMIZE_BLOCKING_SIMD_PREFETCH -o $(BUILD)/$@

all: build
	g++ pa1-the-matrix.c -D OPTIMIZE_BLOCKING -D OPTIMIZE_SIMD -D OPTIMIZE_PREFETCH -D OPTIMIZE_BLOCKING_PREFETCH -D OPTIMIZE_BLOCKING_SIMD -D OPTIMIZE_SIMD_PREFETCH -D OPTIMIZE_BLOCKING_SIMD_PREFETCH -o $(BUILD)/$@

clean:
	@rm -rf $(BUILD)
	@rm -f out.txt

build:
	@mkdir -p $(BUILD)