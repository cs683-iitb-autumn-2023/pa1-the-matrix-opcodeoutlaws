
// CS 683 (Autumn 2023)
// PA 1: The Matrix

// includes
#include <stdio.h>
#include <time.h>			// for time-keeping
#include <xmmintrin.h> 		// for intrinsic functions
#include <math.h>

// defines
// NOTE: you can change this value as per your requirement
#define BLOCK_SIZE	20		// size of the block

/**
 * @brief 		Generates random numbers between values fMin and fMax.
 * @param 		fMin 	lower range
 * @param 		fMax 	upper range
 * @return 		random floating point number
 */
double fRand(double fMin, double fMax) {

	double f = (double)rand() / RAND_MAX;
	return fMin + f * (fMax - fMin);
}

void print_matrix(double* matrix, int dim, char* label) {
	printf("\n**************%s**************\n", label);
	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < dim; j++) {
			printf("%f ", matrix[i * dim + j]);
		}
		printf("\n");
	}
	printf("\n");
}

/**
 * @brief 		Initialize a matrix of given dimension with random values.
 * @param 		matrix 		pointer to the matrix
 * @param 		rows 		number of rows in the matrix
 * @param 		cols 		number of columns in the matrix
 */
void initialize_matrix(double *matrix, int rows, int cols) {

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			matrix[i * cols + j] = fRand(0.0001, 1.0000); // random values between 0 and 1
		}
	}
	// print_matrix(matrix, rows, "Initialization");
}

/**
 * @brief 		Initialize result matrix of given dimension with 0.
 * @param 		matrix 		pointer to the matrix
 * @param 		rows 		number of rows in the matrix
 * @param 		cols 		number of columns in the matrix
 */
void initialize_result_matrix(double *matrix, int rows, int cols) {

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			matrix[i * cols + j] = 0.0;
		}
	}
}


/**
 * @brief 		Performs matrix multiplication of two matrices.
 * @param 		A 			pointer to the first matrix
 * @param 		B 			pointer to the second matrix
 * @param 		C 			pointer to the resultant matrix
 * @param 		dim 		dimension of the matrices
 */
void normal_mat_mul(double *A, double *B, double *C, int dim) {

	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < dim; j++) {
			for (int k = 0; k < dim; k++) {
				C[i * dim + j] += A[i * dim + k] * B[k * dim + j];
			}
		}
	}
	// print_matrix(C, dim, "Expected");
}

/**
 * @brief 		Task 1: Performs matrix multiplication of two matrices using blocking.
 * @param 		A 			pointer to the first matrix
 * @param 		B 			pointer to the second matrix
 * @param 		C 			pointer to the resultant matrix
 * @param 		dim 		dimension of the matrices
 * @param 		block_size 	size of the block
 * @note 		The block size should be a multiple of the dimension of the matrices.
 * 				For example, if the dimension is 1024, then the block size can be 32, 64, 128, etc.
 * 				You can assume that the matrices are square matrices.
*/
void blocking_mat_mul(double *A, double *B, double *C, int dim, int block_size) {
	for (int ib = 0; ib < dim; ib+=block_size){
		for (int jb = 0; jb < dim; jb+=block_size){
			for (int kb = 0; kb < dim; kb+=block_size){

				for (int i = ib; i < (ib+block_size); i++) {
					for (int j = jb; j < (jb+block_size); j++) {
						double res = C[i * dim + j];
						for (int k = kb; k < (kb+block_size); k++) {
							res += A[i * dim + k] * B[k * dim + j];
						}
						C[i * dim + j] = res;
					}
				}


			}
		}
	}
	// print_matrix(C, dim, "Actual");	

}

void verify_correctness(double *C, double *D, int dim) {
	double epsilon = 1e-9;
	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < dim; j++) {
			if (fabs(C[i * dim + j] -  D[i * dim + j]) > epsilon) {
				printf("%f & %f at (%d %d)\n", C[i * dim + j], D[i * dim + j], i, j);
				printf("Matrix multiplication is incorrect!\n");
				return;
			}
		}
	}
	printf("Matrix multiplication is correct!\n");
	return;
}

void print_reg(__m128d reg, char* label) {
	double values[2];
	_mm_storeu_pd(values, reg);
	printf("reg %s: %f %f\n", label, values[0], values[1]);
	return;
}

/**
 * @brief 		Task 2: Performs matrix multiplication of two matrices using SIMD instructions.
 * @param 		A 			pointer to the first matrix
 * @param 		B 			pointer to the second matrix
 * @param 		C 			pointer to the resultant matrix
 * @param 		dim 		dimension of the matrices
 * @note 		You can assume that the matrices are square matrices.
*/
void simd_mat_mul(double *A, double *B, double *C, int dim) {	
	__m128d rA1, rA2, res1, res2, b1, b2, b3, b4, b5, b6, b7, b8, tb1, tb2, tb3, tb4, tb5, tb6, tb7, tb8;
	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < dim; j+=4) {
			res1 = _mm_setzero_pd();
			res2 = _mm_setzero_pd();

			for (int k = 0; k < dim; k+=4) {
				// Get rows form matrix A
				rA1 = _mm_loadu_pd(&A[i*dim + k]);
				rA2 = _mm_loadu_pd(&A[i*dim + k + 2]);
				
				//Get rows of matrix B
				b1 = _mm_loadu_pd(&B[k*dim + j]);
				b2 = _mm_loadu_pd(&B[(k+1)*dim + j]);
				b3 = _mm_loadu_pd(&B[k*dim + j + 2]);
				b4 = _mm_loadu_pd(&B[(k+1)*dim + j + 2]);
				
				b5 = _mm_loadu_pd(&B[(k+2)*dim + j]);
				b6 = _mm_loadu_pd(&B[(k+3)*dim + j]);
				b7 = _mm_loadu_pd(&B[(k+2)*dim + j + 2]);
				b8 = _mm_loadu_pd(&B[(k+3)*dim + j + 2]);

				// Shuffle them to align for multiplication			
				tb1 = _mm_mul_pd(rA1, _mm_shuffle_pd(b1, b2, 0x00));
				tb2 = _mm_mul_pd(rA1, _mm_shuffle_pd(b1, b2, 0xff));
				tb3 = _mm_mul_pd(rA1, _mm_shuffle_pd(b3, b4, 0x00));
				tb4 = _mm_mul_pd(rA1, _mm_shuffle_pd(b3, b4, 0xff));

				tb5 = _mm_mul_pd(rA2, _mm_shuffle_pd(b5, b6, 0x00));
				tb6 = _mm_mul_pd(rA2, _mm_shuffle_pd(b5, b6, 0xff));
				tb7 = _mm_mul_pd(rA2, _mm_shuffle_pd(b7, b8, 0x00));
				tb8 = _mm_mul_pd(rA2, _mm_shuffle_pd(b7, b8, 0xff));

				// Reshuffle to align for addition
				res1 = _mm_add_pd(res1, _mm_add_pd(_mm_shuffle_pd(tb1, tb2, 0x00), _mm_shuffle_pd(tb1, tb2, 0xff)));
				res2 = _mm_add_pd(res2, _mm_add_pd(_mm_shuffle_pd(tb3, tb4, 0x00), _mm_shuffle_pd(tb3, tb4, 0xff)));
				
				res1 = _mm_add_pd(res1, _mm_add_pd(_mm_shuffle_pd(tb5, tb6, 0x00), _mm_shuffle_pd(tb5, tb6, 0xff)));
				res2 = _mm_add_pd(res2, _mm_add_pd(_mm_shuffle_pd(tb7, tb8, 0x00), _mm_shuffle_pd(tb7, tb8, 0xff)));
			}
			// Save results to C
			_mm_storeu_pd((double*) &C[i*dim + j], res1);
			_mm_storeu_pd((double*) &C[i*dim + j+2], res2);
		}
	}
	// print_matrix(C, dim, "Actual");		
}

/**
 * @brief 		Task 3: Performs matrix multiplication of two matrices using software prefetching.
 * @param 		A 			pointer to the first matrix
 * @param 		B 			pointer to the second matrix
 * @param 		C 			pointer to the resultant matrix
 * @param 		dim 		dimension of the matrices
 * @note 		You can assume that the matrices are square matrices.
*/
void prefetch_mat_mul(double *A, double *B, double *C, int dim) {

	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < dim; j++) {
			double sum=0;
			    __builtin_prefetch(&C[i * dim + j],1,3);
				__builtin_prefetch(&A[i * dim],0,3);
				__builtin_prefetch(&A[i * dim + 8],0,3);
				__builtin_prefetch(&B[j], 0,3);
				__builtin_prefetch(&B[dim + j],0,3);
				__builtin_prefetch(&B[2 * dim + j],0,3);
				__builtin_prefetch(&B[3 * dim + j],0,3);
				__builtin_prefetch(&B[4 * dim + j],0,3);	
				
				int k;
			for (k= 0; k < dim-5; k=k+5) {				
				__builtin_prefetch(&A[i * dim + k],0,3);
				__builtin_prefetch(&B[(k + 5)* dim + j], 0,3);
				__builtin_prefetch(&B[(k + 6)* dim + j],0,3);
				__builtin_prefetch(&B[(k + 7) * dim + j],0,3);
				__builtin_prefetch(&B[(k + 8) * dim + j],0,3);
				__builtin_prefetch(&B[(k + 9) * dim + j],0,3);		
				__builtin_prefetch(&A[i * dim + k+8],0,3);
			
				sum += A[i * dim + k] * B[k * dim + j];
				sum += A[i * dim + k+1] * B[(k + 1)* dim + j];
				sum += A[i * dim + k+2] * B[(k + 2) * dim + j];
				sum += A[i * dim + k+3] * B[(k + 3) * dim + j];
				sum += A[i * dim + k+4] * B[(k + 4) * dim + j];				
				
			}
				
				sum += A[i * dim + k] * B[k * dim + j];
				sum += A[i * dim + k+1] * B[(k + 1)* dim + j];
				sum += A[i * dim + k+2] * B[(k + 2) * dim + j];
				sum += A[i * dim + k+3] * B[(k + 3) * dim + j];
				sum += A[i * dim + k+4] * B[(k + 4) * dim + j];				

				C[i * dim + j]=sum;
		}
	}
	// print_matrix(C, dim, "Actual");	
}

/**
 * @brief 		Bonus Task 1: Performs matrix multiplication of two matrices using blocking along with SIMD instructions.
 * @param 		A 			pointer to the first matrix
 * @param 		B 			pointer to the second matrix
 * @param 		C 			pointer to the resultant matrix
 * @param 		dim 		dimension of the matrices
 * @param 		block_size 	size of the block
 * @note 		The block size should be a multiple of the dimension of the matrices.
*/
void blocking_simd_mat_mul(double *A, double *B, double *C, int dim, int block_size) {
	__m128d rA1, rA2, res1, res2, b1, b2, b3, b4, b5, b6, b7, b8, tb1, tb2, tb3, tb4, tb5, tb6, tb7, tb8;
	for (int ib = 0; ib < dim; ib+=block_size){
		for (int jb = 0; jb < dim; jb+=block_size){
			for (int kb = 0; kb < dim; kb+=block_size){

				for (int i = ib; i < (ib+block_size); i++) {
					for (int j = jb; j < (jb+block_size); j+=4) {
						// double res = C[i * dim + j];

						res1 = _mm_loadu_pd(&C[i*dim + j]);
						res2 = _mm_loadu_pd(&C[i*dim + j+2]);

						for (int k = kb; k < (kb+block_size); k+=4) {
							// Get rows form matrix A
							rA1 = _mm_loadu_pd(&A[i*dim + k]);
							rA2 = _mm_loadu_pd(&A[i*dim + k + 2]);
							
							//Get rows of matrix B
							b1 = _mm_loadu_pd(&B[k*dim + j]);
							b2 = _mm_loadu_pd(&B[(k+1)*dim + j]);
							b3 = _mm_loadu_pd(&B[k*dim + j + 2]);
							b4 = _mm_loadu_pd(&B[(k+1)*dim + j + 2]);
							
							b5 = _mm_loadu_pd(&B[(k+2)*dim + j]);
							b6 = _mm_loadu_pd(&B[(k+3)*dim + j]);
							b7 = _mm_loadu_pd(&B[(k+2)*dim + j + 2]);
							b8 = _mm_loadu_pd(&B[(k+3)*dim + j + 2]);

							// Shuffle them to align for multiplication			
							tb1 = _mm_mul_pd(rA1, _mm_shuffle_pd(b1, b2, 0x00));
							tb2 = _mm_mul_pd(rA1, _mm_shuffle_pd(b1, b2, 0xff));
							tb3 = _mm_mul_pd(rA1, _mm_shuffle_pd(b3, b4, 0x00));
							tb4 = _mm_mul_pd(rA1, _mm_shuffle_pd(b3, b4, 0xff));

							tb5 = _mm_mul_pd(rA2, _mm_shuffle_pd(b5, b6, 0x00));
							tb6 = _mm_mul_pd(rA2, _mm_shuffle_pd(b5, b6, 0xff));
							tb7 = _mm_mul_pd(rA2, _mm_shuffle_pd(b7, b8, 0x00));
							tb8 = _mm_mul_pd(rA2, _mm_shuffle_pd(b7, b8, 0xff));

							// Reshuffle to align for addition
							res1 = _mm_add_pd(res1, _mm_add_pd(_mm_shuffle_pd(tb1, tb2, 0x00), _mm_shuffle_pd(tb1, tb2, 0xff)));
							res2 = _mm_add_pd(res2, _mm_add_pd(_mm_shuffle_pd(tb3, tb4, 0x00), _mm_shuffle_pd(tb3, tb4, 0xff)));
							
							res1 = _mm_add_pd(res1, _mm_add_pd(_mm_shuffle_pd(tb5, tb6, 0x00), _mm_shuffle_pd(tb5, tb6, 0xff)));
							res2 = _mm_add_pd(res2, _mm_add_pd(_mm_shuffle_pd(tb7, tb8, 0x00), _mm_shuffle_pd(tb7, tb8, 0xff)));
							
							// res += A[i * dim + k] * B[k * dim + j];


						}
						// C[i * dim + j] = res;
						_mm_storeu_pd((double*) &C[i*dim + j], res1);
						_mm_storeu_pd((double*) &C[i*dim + j+2], res2);
					}
				}
			}
		}
	}

}

/**
 * @brief 		Bonus Task 2: Performs matrix multiplication of two matrices using blocking along with software prefetching.
 * @param 		A 			pointer to the first matrix
 * @param 		B 			pointer to the second matrix
 * @param 		C 			pointer to the resultant matrix
 * @param 		dim 		dimension of the matrices
 * @param 		block_size 	size of the block
 * @note 		The block size should be a multiple of the dimension of the matrices.
*/
void blocking_prefetch_mat_mul(double *A, double *B, double *C, int dim, int block_size) {


	for (int ib = 0; ib < dim; ib+=block_size){
		for (int jb = 0; jb < dim; jb+=block_size){
			for (int kb = 0; kb < dim; kb+=block_size){

				for (int i = ib; i < (ib+block_size); i++) {
					for (int j = jb; j < (jb+block_size); j++) {
						double sum = C[i * dim + j];
						int k;
						for (k = kb; k < (kb+block_size-8); k=k+8) {
							__builtin_prefetch(&A[i * dim + (k + 8)], 0, 3);
							__builtin_prefetch(&B[i * dim + (k + 8)], 0, 3);
							sum += A[i * dim + k] * B[k * dim + j];
							sum += A[i * dim + (k + 1)] * B[(k + 1) * dim + j];
							sum += A[i * dim + (k + 2)] * B[(k + 2) * dim + j];
							sum += A[i * dim + (k + 3)] * B[(k + 3) * dim + j];
							sum += A[i * dim + (k + 4)] * B[(k + 4) * dim + j];
							sum += A[i * dim + (k + 5)] * B[(k + 5) * dim + j];
							sum += A[i * dim + (k + 6)] * B[(k + 6) * dim + j];
							sum += A[i * dim + (k + 7)] * B[(k + 7) * dim + j];
							
						}
						for (int s = k;s < (kb+block_size); s++){
							sum += A[i * dim + s] * B[s * dim + j];
						}
						C[i * dim + j] = sum;
					}
				}


			}
		}
	}

}

/**
 * @brief 		Bonus Task 3: Performs matrix multiplication of two matrices using SIMD instructions along with software prefetching.
 * @param 		A 			pointer to the first matrix
 * @param 		B 			pointer to the second matrix
 * @param 		C 			pointer to the resultant matrix
 * @param 		dim 		dimension of the matrices
 * @note 		You can assume that the matrices are square matrices.
*/
void simd_prefetch_mat_mul(double *A, double *B, double *C, int dim) {
	__m128d rA1, rA2, res1, res2, b1, b2, b3, b4, b5, b6, b7, b8, tb1, tb2, tb3, tb4, tb5, tb6, tb7, tb8;
	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < dim; j+=4) {
			res1 = _mm_setzero_pd();
			res2 = _mm_setzero_pd();

			for (int k = 0; k < dim; k+=4) {
				if (k + 4 < dim) {
					__builtin_prefetch(&A[(i+1)*dim + k], 0, 3);
					__builtin_prefetch(&B[(k+4)*dim + j], 0, 3);
					__builtin_prefetch(&B[(k+5)*dim + j], 0, 3);
					__builtin_prefetch(&B[(k+6)*dim + j], 0, 3);
					__builtin_prefetch(&B[(k+7)*dim + j], 0, 3);
				}
				
				// Get rows form matrix A
				rA1 = _mm_loadu_pd(&A[i*dim + k]);
				rA2 = _mm_loadu_pd(&A[i*dim + k + 2]);
				
				//Get rows of matrix B
				b1 = _mm_loadu_pd(&B[k*dim + j]);
				b2 = _mm_loadu_pd(&B[(k+1)*dim + j]);
				b3 = _mm_loadu_pd(&B[k*dim + j + 2]);
				b4 = _mm_loadu_pd(&B[(k+1)*dim + j + 2]);
				
				b5 = _mm_loadu_pd(&B[(k+2)*dim + j]);
				b6 = _mm_loadu_pd(&B[(k+3)*dim + j]);
				b7 = _mm_loadu_pd(&B[(k+2)*dim + j + 2]);
				b8 = _mm_loadu_pd(&B[(k+3)*dim + j + 2]);

				// Shuffle them to align for multiplication			
				tb1 = _mm_mul_pd(rA1, _mm_shuffle_pd(b1, b2, 0x00));
				tb2 = _mm_mul_pd(rA1, _mm_shuffle_pd(b1, b2, 0xff));
				tb3 = _mm_mul_pd(rA1, _mm_shuffle_pd(b3, b4, 0x00));
				tb4 = _mm_mul_pd(rA1, _mm_shuffle_pd(b3, b4, 0xff));

				tb5 = _mm_mul_pd(rA2, _mm_shuffle_pd(b5, b6, 0x00));
				tb6 = _mm_mul_pd(rA2, _mm_shuffle_pd(b5, b6, 0xff));
				tb7 = _mm_mul_pd(rA2, _mm_shuffle_pd(b7, b8, 0x00));
				tb8 = _mm_mul_pd(rA2, _mm_shuffle_pd(b7, b8, 0xff));

				// Reshuffle to align for addition
				res1 = _mm_add_pd(res1, _mm_add_pd(_mm_shuffle_pd(tb1, tb2, 0x00), _mm_shuffle_pd(tb1, tb2, 0xff)));
				res2 = _mm_add_pd(res2, _mm_add_pd(_mm_shuffle_pd(tb3, tb4, 0x00), _mm_shuffle_pd(tb3, tb4, 0xff)));
				
				res1 = _mm_add_pd(res1, _mm_add_pd(_mm_shuffle_pd(tb5, tb6, 0x00), _mm_shuffle_pd(tb5, tb6, 0xff)));
				res2 = _mm_add_pd(res2, _mm_add_pd(_mm_shuffle_pd(tb7, tb8, 0x00), _mm_shuffle_pd(tb7, tb8, 0xff)));
			}
			// Save results to C
			_mm_storeu_pd((double*) &C[i*dim + j], res1);
			_mm_storeu_pd((double*) &C[i*dim + j+2], res2);
		}
	}
	// print_matrix(C, dim, "Actual");	
}

/**
 * @brief 		Bonus Task 4: Performs matrix multiplication of two matrices using blocking along with SIMD instructions and software prefetching.
 * @param 		A 			pointer to the first matrix
 * @param 		B 			pointer to the second matrix
 * @param 		C 			pointer to the resultant matrix
 * @param 		dim 		dimension of the matrices
 * @param 		block_size 	size of the block
 * @note 		The block size should be a multiple of the dimension of the matrices.
 * @note 		You can assume that the matrices are square matrices.
*/
void blocking_simd_prefetch_mat_mul(double *A, double *B, double *C, int dim, int block_size) {
	__m128d rA1, rA2, res1, res2, b1, b2, b3, b4, b5, b6, b7, b8, tb1, tb2, tb3, tb4, tb5, tb6, tb7, tb8;
	for (int ib = 0; ib < dim; ib+=block_size){
		for (int jb = 0; jb < dim; jb+=block_size){
			for (int kb = 0; kb < dim; kb+=block_size){

				for (int i = ib; i < (ib+block_size); i++) {
					for (int j = jb; j < (jb+block_size); j+=4) {
						// double res = C[i * dim + j];

						res1 = _mm_loadu_pd(&C[i*dim + j]);
						res2 = _mm_loadu_pd(&C[i*dim + j+2]);

						for (int k = kb; k < (kb+block_size); k+=4) {
							if (k + 4 < (kb + block_size)) {
								__builtin_prefetch(&A[(i+1)*dim + k], 0, 3);
								__builtin_prefetch(&B[(k+4)*dim + j], 0, 3);
								__builtin_prefetch(&B[(k+5)*dim + j], 0, 3);
								__builtin_prefetch(&B[(k+6)*dim + j], 0, 3);
								__builtin_prefetch(&B[(k+7)*dim + j], 0, 3);
							}
							// Get rows form matrix A
							rA1 = _mm_loadu_pd(&A[i*dim + k]);
							rA2 = _mm_loadu_pd(&A[i*dim + k + 2]);
							
							//Get rows of matrix B
							b1 = _mm_loadu_pd(&B[k*dim + j]);
							b2 = _mm_loadu_pd(&B[(k+1)*dim + j]);
							b3 = _mm_loadu_pd(&B[k*dim + j + 2]);
							b4 = _mm_loadu_pd(&B[(k+1)*dim + j + 2]);
							
							b5 = _mm_loadu_pd(&B[(k+2)*dim + j]);
							b6 = _mm_loadu_pd(&B[(k+3)*dim + j]);
							b7 = _mm_loadu_pd(&B[(k+2)*dim + j + 2]);
							b8 = _mm_loadu_pd(&B[(k+3)*dim + j + 2]);

							// Shuffle them to align for multiplication			
							tb1 = _mm_mul_pd(rA1, _mm_shuffle_pd(b1, b2, 0x00));
							tb2 = _mm_mul_pd(rA1, _mm_shuffle_pd(b1, b2, 0xff));
							tb3 = _mm_mul_pd(rA1, _mm_shuffle_pd(b3, b4, 0x00));
							tb4 = _mm_mul_pd(rA1, _mm_shuffle_pd(b3, b4, 0xff));

							tb5 = _mm_mul_pd(rA2, _mm_shuffle_pd(b5, b6, 0x00));
							tb6 = _mm_mul_pd(rA2, _mm_shuffle_pd(b5, b6, 0xff));
							tb7 = _mm_mul_pd(rA2, _mm_shuffle_pd(b7, b8, 0x00));
							tb8 = _mm_mul_pd(rA2, _mm_shuffle_pd(b7, b8, 0xff));

							// Reshuffle to align for addition
							res1 = _mm_add_pd(res1, _mm_add_pd(_mm_shuffle_pd(tb1, tb2, 0x00), _mm_shuffle_pd(tb1, tb2, 0xff)));
							res2 = _mm_add_pd(res2, _mm_add_pd(_mm_shuffle_pd(tb3, tb4, 0x00), _mm_shuffle_pd(tb3, tb4, 0xff)));
							
							res1 = _mm_add_pd(res1, _mm_add_pd(_mm_shuffle_pd(tb5, tb6, 0x00), _mm_shuffle_pd(tb5, tb6, 0xff)));
							res2 = _mm_add_pd(res2, _mm_add_pd(_mm_shuffle_pd(tb7, tb8, 0x00), _mm_shuffle_pd(tb7, tb8, 0xff)));
							
							// res += A[i * dim + k] * B[k * dim + j];


						}
						// C[i * dim + j] = res;
						_mm_storeu_pd((double*) &C[i*dim + j], res1);
						_mm_storeu_pd((double*) &C[i*dim + j+2], res2);
					}
				}
			}
		}
	}

}

// NOTE: DO NOT CHANGE ANYTHING BELOW THIS LINE
/**
 * @brief 		Main function
 * @param 		argc 		number of command line arguments
 * @param 		argv 		array of command line arguments
 * @return 		0 on success
 * @note 		DO NOT CHANGE THIS FUNCTION
 * 				DO NOT ADD OR REMOVE ANY COMMAND LINE ARGUMENTS
*/
int main(int argc, char **argv) {

	if ( argc <= 1 ) {
		printf("Pass the matrix dimension as argument :)\n\n");
		return 0;
	}

	else {
		int matrix_dim = atoi(argv[1]);

		// variables definition and initialization
		clock_t t_normal_mult, t_blocking_mult, t_prefetch_mult, t_simd_mult, t_blocking_simd_mult, t_blocking_prefetch_mult, t_simd_prefetch_mult, t_blocking_simd_prefetch_mult;
		double time_normal_mult, time_blocking_mult, time_prefetch_mult, time_simd_mult, time_blocking_simd_mult, time_blocking_prefetch_mult, time_simd_prefetch_mult, time_blocking_simd_prefetch_mult;

		double *A = (double *)malloc(matrix_dim * matrix_dim * sizeof(double));
		double *B = (double *)malloc(matrix_dim * matrix_dim * sizeof(double));
		double *C = (double *)calloc(matrix_dim * matrix_dim, sizeof(double));
		double *D = (double *)calloc(matrix_dim * matrix_dim, sizeof(double));

		// initialize random seed
		srand(time(NULL));

		// initialize matrices A and B with random values
		initialize_matrix(A, matrix_dim, matrix_dim);
		initialize_matrix(B, matrix_dim, matrix_dim);

		// perform normal matrix multiplication
		t_normal_mult = clock();
		normal_mat_mul(A, B, C, matrix_dim);
		t_normal_mult = clock() - t_normal_mult;

		time_normal_mult = ((double)t_normal_mult) / CLOCKS_PER_SEC; // in seconds
		printf("Normal matrix multiplication took %f seconds to execute \n\n", time_normal_mult);

	#ifdef OPTIMIZE_BLOCKING
		// Task 1: perform blocking matrix multiplication

		// initialize result matrix to 0
		initialize_result_matrix(D, matrix_dim, matrix_dim);
		
		t_blocking_mult = clock();
		blocking_mat_mul(A, B, D, matrix_dim, BLOCK_SIZE);
		t_blocking_mult = clock() - t_blocking_mult;

		time_blocking_mult = ((double)t_blocking_mult) / CLOCKS_PER_SEC; // in seconds
		printf("Blocking matrix multiplication took %f seconds to execute \n", time_blocking_mult);
		printf("Normalized performance: %f \n\n", time_normal_mult / time_blocking_mult);
		// verify_correctness(C, D, matrix_dim);
	#endif

	#ifdef OPTIMIZE_SIMD
		// Task 2: perform matrix multiplication with SIMD instructions
		// initialize result matrix to 0

		initialize_result_matrix(D, matrix_dim, matrix_dim);
		
		t_simd_mult = clock();
		simd_mat_mul(A, B, D, matrix_dim);
		t_simd_mult = clock() - t_simd_mult;

		time_simd_mult = ((double)t_simd_mult) / CLOCKS_PER_SEC; // in seconds
		printf("SIMD matrix multiplication took %f seconds to execute \n", time_simd_mult);
		printf("Normalized performance: %f \n\n", time_normal_mult / time_simd_mult);
		// verify_correctness(C, D, matrix_dim);
	#endif

	#ifdef OPTIMIZE_PREFETCH
		// Task 3: perform matrix multiplication with prefetching
		
		// initialize result matrix to 0
		initialize_result_matrix(D, matrix_dim, matrix_dim);		

		t_prefetch_mult = clock();
		prefetch_mat_mul(A, B, D, matrix_dim);
		t_prefetch_mult = clock() - t_prefetch_mult;

		time_prefetch_mult = ((double)t_prefetch_mult) / CLOCKS_PER_SEC; // in seconds
		printf("Prefetching matrix multiplication took %f seconds to execute \n", time_prefetch_mult);
		printf("Normalized performance: %f \n\n", time_normal_mult / time_prefetch_mult);
		// verify_correctness(C, D, matrix_dim);
	#endif

	#ifdef OPTIMIZE_BLOCKING_SIMD
		// Bonus Task 1: perform matrix multiplication using blocking along with SIMD instructions
		
		// initialize result matrix to 0
		initialize_result_matrix(D, matrix_dim, matrix_dim);
		
		t_blocking_simd_mult = clock();
		blocking_simd_mat_mul(A, B, D, matrix_dim, BLOCK_SIZE);
		t_blocking_simd_mult = clock() - t_blocking_simd_mult;

		time_blocking_simd_mult = ((double)t_blocking_simd_mult) / CLOCKS_PER_SEC; // in seconds
		printf("Blocking with SIMD matrix multiplication took %f seconds to execute \n", time_blocking_simd_mult);
		printf("Normalized performance: %f \n\n", time_normal_mult / time_blocking_simd_mult);
		// verify_correctness(C, D, matrix_dim);
	#endif

	#ifdef OPTIMIZE_BLOCKING_PREFETCH
		// Bonus Task 2: perform matrix multiplication using blocking along with software prefetching
		
		// initialize result matrix to 0
		initialize_result_matrix(D, matrix_dim, matrix_dim);
		
		t_blocking_prefetch_mult = clock();
		blocking_prefetch_mat_mul(A, B, D, matrix_dim, BLOCK_SIZE);
		t_blocking_prefetch_mult = clock() - t_blocking_prefetch_mult;

		time_blocking_prefetch_mult = ((double)t_blocking_prefetch_mult) / CLOCKS_PER_SEC; // in seconds
		printf("Blocking with prefetching matrix multiplication took %f seconds to execute \n", time_blocking_prefetch_mult);
		printf("Normalized performance: %f \n\n", time_normal_mult / time_blocking_prefetch_mult);
		// verify_correctness(C, D, matrix_dim);
	#endif

	#ifdef OPTIMIZE_SIMD_PREFETCH
		// Bonus Task 3: perform matrix multiplication using SIMD instructions along with software prefetching
		
		// initialize result matrix to 0
		initialize_result_matrix(D, matrix_dim, matrix_dim);
		
		t_simd_prefetch_mult = clock();
		simd_prefetch_mat_mul(A, B, D, matrix_dim);
		t_simd_prefetch_mult = clock() - t_simd_prefetch_mult;

		time_simd_prefetch_mult = ((double)t_simd_prefetch_mult) / CLOCKS_PER_SEC; // in seconds
		printf("SIMD with prefetching matrix multiplication took %f seconds to execute \n", time_simd_prefetch_mult);
		printf("Normalized performance: %f \n\n", time_normal_mult / time_simd_prefetch_mult);
		// verify_correctness(C, D, matrix_dim);
	#endif

	#ifdef OPTIMIZE_BLOCKING_SIMD_PREFETCH
		// Bonus Task 4: perform matrix multiplication using blocking, SIMD instructions and software prefetching
		
		// initialize result matrix to 0
		initialize_result_matrix(D, matrix_dim, matrix_dim);
		
		t_blocking_simd_prefetch_mult = clock();
		blocking_simd_prefetch_mat_mul(A, B, D, matrix_dim, BLOCK_SIZE);
		t_blocking_simd_prefetch_mult = clock() - t_blocking_simd_prefetch_mult;

		time_blocking_simd_prefetch_mult = ((double)t_blocking_simd_prefetch_mult) / CLOCKS_PER_SEC; // in seconds
		printf("Blocking with SIMD and prefetching matrix multiplication took %f seconds to execute \n", time_blocking_simd_prefetch_mult);
		printf("Normalized performance: %f \n\n", time_normal_mult / time_blocking_simd_prefetch_mult);
		// verify_correctness(C, D, matrix_dim);
	#endif

		// free allocated memory
		free(A);
		free(B);
		free(C);
		free(D);

		return 0;
	}
}
