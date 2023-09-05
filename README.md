[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/mnOJa0WY)
# PA1-The-Matrix

Perform matrix multiplication using various optimization techniques:
- blocked matrix multiplication
- SIMD instructions
- software prefetching

and combination of these techniques.

<br>

---
## Task 1: Blocked Matrix Multiplication
### Implementation:
#### Basic Idea of blocking/tilling
- We have divided the matrices into small tiles and trying to reuse the blocks fetched in the cache
- To implement it we are using 6 for loops, where first three loops select which tiles to multiply and last three loops use normal matrix multiplication to multiply the selected tiles.
#### Comparison with Normal Matrix Multiplication
- In normal matrix multiplication, while performing C = A * B, we access B in column-major order. If the size of a column is greater than the cache size whatever block of matrix B that will be brought into the cache while fetching elements of matrix B will be replaced and we will be facing a cache miss for every element access of matrix B

  ![image](https://github.com/cs683-iitb-autumn-2023/pa1-the-matrix-opcodeoutlaws/assets/142027995/ede6508f-0696-43a7-9d86-9fdf92642ae7)

- The block that were brought into the cache while accessing B are not used expect for the 1st element. So the idea of Blocked matrix multiplication is to use the other elements of that block before that block is evicted from the cache.  
#### Miss rate reduction using Blocking
- Now in blocking the martix will be divided into small tiles (lets say of size B) and these tiles will be multiplied with each other instead of whole matrix.

  ![image](https://github.com/cs683-iitb-autumn-2023/pa1-the-matrix-opcodeoutlaws/assets/142027995/45c4b7ea-f108-49e5-9944-ad765929dd9b)

- Now to compute a tile in the result matrix C, consider a tile C<sub>i,j</sub> ,

  We multipy i<sup>th</sup> row tiles of matrix A with j<sup>th</sup> column tiles of matrix B

  As seen from the above image, to compute tile  C<sub>1,1</sub>,

     C<sub>1,1</sub> =  A<sub>1,1</sub> x  B<sub>1,1</sub> +  A<sub>1,2</sub> x B<sub>2,1</sub>

  ![image](https://github.com/cs683-iitb-autumn-2023/pa1-the-matrix-opcodeoutlaws/assets/142027995/87b3aa52-2b08-426a-91b0-68466676c49c)
 
  

- Now while accessing a tile of B in column major order, the blocks that are fetched in the cache won't be evicted till the whole tile is accessed. This is because the tile size will be small and fit completely into the cache.
- So while accessing the later elements (2<sup>nd</sup> column onwards) of that tile in column major order we won't get a miss as the blocks are already in the cache. This is how the blocks brought into the cache while accessing the tile of B are reused and will reduce the cache miss rate.  

### Observations:
- We have recorded the execution time, data cache miss rate and speed up for different matrix sizes and represented in the table below:
  
  ![image](https://github.com/cs683-iitb-autumn-2023/pa1-the-matrix-opcodeoutlaws/assets/142027995/a52a4d6d-84b8-4fde-a148-19b4e1cd849e)


### Analysis:
- It can seen from the table that we have a significant decrease in the miss rate specially for matrix of dimension 800. This is because the single column of martix will be very large. And each element access will bring a block (of size 64B) into the cache. So accessing just one column of the matrix B will be very large (800 * 64B) and won't fit completely in the cache. So the blocks fetched will be evicted and would result in unnecessary cache misses.
- This won't happen for matrix of dimension 100 and 200 as their coloumns will be of smaller size. Because of this we can see a high miss rate (4.5 %) for normal matrix multiplication in 800 dimension matrix. Blocking reduces the miss rate to 0.1 % in each case by reusing the fetched blocks and gives a good speedup. 

<br>

---
## Task 2: SIMD instructions
### Implementation:
#### SIMD Registers
- We are using 128 bits wide registers for implementing SIMD instructions. One double is of 64 bits in size. Hence we can store two doubles adjacent to each other in a 64 bit register. Here's how it looks:

  ![image](https://github.com/cs683-iitb-autumn-2023/pa1-the-matrix-opcodeoutlaws/assets/48720143/d5b47259-464c-4fa0-b2c5-a79b6944ab99)


#### SIMD Instructions
- Here we'll see how SIMD multiplication works when we have two `_m128d` registers containing a total of four doubles.

  ![image](https://github.com/cs683-iitb-autumn-2023/pa1-the-matrix-opcodeoutlaws/assets/48720143/7f2f49d2-2b00-4f35-9b16-86c7a78ae041)


- Here is the list of SIMD functions we are using for our implementation:
  1. `_mm_setzero_pd`
  2. `_mm_loadu_pd`
  3. `_mm_mul_pd`
  4. `_mm_add_pd`
  5. `_mm_shuffle_pd`
  6. `_mm_storeu_pd`

- Informaton about each of them can be found here: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html

#### Working

- Below is a snippet of how C1 and C2 is calculated in one step.
  
  ![image](https://github.com/cs683-iitb-autumn-2023/pa1-the-matrix-opcodeoutlaws/assets/48720143/2b852f03-8f50-485b-a864-45ed41abd267)

- We first load A11, A12 in one `_m128d` register. Alongside we also load A21, A22 in another `_m128d` register. This is done using the `_mm_loadu_pd`.
- We load 16 elements from matrix B into 8 `_m128d` registers.
- We need to shuffle the registers to align them for multiplication as shown in above figure. We use  `_mm_shuffle_pd` for the same.
- Next step is to multiply the aligned registers. This gives us multiplication of `A11` × `B11` and `A12` × `B12`, `A11` × `B21` and `A12` × `B22` and so on.
- We realign the registers using `_mm_shuffle_pd` for addition so that we can get `A11 × B11`+ `A12 × B12` and so on.
- We have applied loop unrolling to get some additional boost in performance.
- Finally we store the result to C11, C12, C13 & C14 as two registers using `_mm_storeu_pd`

### Observations:
- We have recorded the execution time, data cache miss rate and speed up for different matrix sizes and represented in the table below:
  
  ![image](https://github.com/cs683-iitb-autumn-2023/pa1-the-matrix-opcodeoutlaws/assets/142027995/453cd3e1-2aa7-4f40-9a56-c65c55b43b52)


### Analysis:
- First we tried to use more prefetch instruction in the 3rd for loop just before their access 

### Limitations:
- Because of some loop unrolling and the way we have accessed elements using `_mm_storeu_pd`, the matrix multiplication works for dimensions which are multiples of 4.

<br>

---
## Task 3: Software Prefetching
Software prefetching uses explicit prefetch instruction that needs to be inserted in the code. It can be used to minimize cache miss latency by prefetching the data blocks before their access. 

### Implementation:
 - `__builtin_prefetch` is a function provided by GCC that needs 3 arguments(memory address, [`rw`](## "pass 0 for read or 1 for write/Default value is 0"), [`locality`](## "value can range between 0 to 3  0: no need of block in future  1: block should reside in l3 cache after use  2: block should reside in l2 & l3 cache after use  3: block should reside in l1,l2 & l3 cache after use/Default value is 3")) among which rw & locality options are optional.
- First we did loop unrolling such that more prefetch instruction can be inserted in the code.
- We have used `__builtin_prefetch` instruction in such a way that needed blocks should be brought to l1 cache ahead of their access.
- To bring blocks ahead of their access we inserted prefetch instructions 1 iteration before the block access.

### Observation:
- We have recorded the execution time, data cache miss rate and speed up for different matrix sizes and represented in the table below:
  
  ![image](https://github.com/cs683-iitb-autumn-2023/pa1-the-matrix-opcodeoutlaws/assets/142027995/2694bc71-d35b-4646-acc6-dd1ea30ca50e)


### Analysis:
- There is a decent speedup, however the effect is majorly beacause of loop unrolling.
- Loop unrolling reduces the number of instructions being executed, thus reduing the total cycles spent while doing so. All this at an expense of slight reduction in code readability.
- We see the number of I refs has decreased in the prefetching implementation despite the fact that we aare adding the prefetch function calls.
- Upon refering multiple sources we have found that the `__builtin_prefetch` does not affect the performance that much (although not using it properly can reduce the performance) because of the default optimizations of the compiler. (https://stackoverflow.com/questions/8460563/builtin-prefetch-how-much-does-it-read/8460606#8460606)

<br>

---
## Bonus Task 1: Blocked Matrix Multiplication + SIMD instructions
This implementation takes advantage of both: blocking and simd instructions. We aim to reduce the number of multiplication operations and also reduce the miss rate with this implementation.

### Implementation:
- We have started with the approach that we used for blocking
- And we changed the actual matrix multiplication taking place in the innermost for loop of blocking implementation with SIMD based multiplication.
- Some amount of loop unrolling is also done to acheive greater performance.

### Observations:
- We have recorded the execution time, data cache miss rate and speed up for different matrix sizes and represented in the table below:
  
  ![image](https://github.com/cs683-iitb-autumn-2023/pa1-the-matrix-opcodeoutlaws/assets/142027995/36665387-0a41-425c-b558-934f60e58384)


### Limitations:
- Here we have limitations of SIMD implementation intersected with limitations of blocking implementation.
- The dimension of matrix should be divisible by 4 and also be divisible by `BLOCK_SIZE`. In order to satisfy the requirement of optimizing matrix multiplication of dimensions 100, 200 and 800, we have decided the `BLOCK_SIZE` to be 20.
  
<br>

---
## Bonus Task 2: Blocked Matrix Multiplication + Software Prefetching

### Implementation:
#### Basic Idea
- Here we are using software prefetching along with blocked matrix multiplication. In blocked multiplication we already reduced the miss rate by reusing the blocks of matrix B.
- Here we try to prefetch the blocks in a tile of both matrices A and B to have a decrease in the miss penalty.

#### Miss rate reduction using blocking & prefetching
- Now to access a tile (lets say of size B x B) of either matrix A or matrix B, we get B<sup>2</sup>/8 misses.
  
  ![image](https://github.com/cs683-iitb-autumn-2023/pa1-the-matrix-opcodeoutlaws/assets/142027995/dc7dc911-b05e-4da9-929a-2fc75e03a4dc)


- The cache line size of our machine is 64 Bytes and each element (type double) is 8 bytes, so every cache line will contain 8 elements.
- We prefetch the first element of each block so we get the whole block into the cache before we try to access the elements. We do this by unrolling the loop and use `__builtin_prefetch` for the 8<sup>th</sup> element in the current iteration and fetch 0<sup>th</sup> to 7<sup>th</sup> element normally. 
- So in the next iteration we already have the 8th element and so here we prefetch the 16<sup>th</sup> element and fetch elements 9<sup>th</sup> to 15<sup>th</sup> (as they are already loaded in the cache beacause of the prefetch in the previous iteration) and so on to the next iteration.
- So we should get less miss penalty for other blocks except for the first 1<sup>st</sup> block.

### Observation:
- We have recorded the execution time, data cache miss rate and speed up for different matrix sizes and represented in the table below:

  ![image](https://github.com/cs683-iitb-autumn-2023/pa1-the-matrix-opcodeoutlaws/assets/142027995/60c1f79a-8048-4e67-bba6-54e56730bd30)


### Analysis:
- Because of blocking we get a good improvement in the miss rate compared to normal matrix multiplication and prefetch improves miss penalty.
- As prefetch adds two extra prefetch instructions in the loop, we get an increase in total instruction references.
- Assuming that either becasue of compiler optimization or hardware prefetching we don't see that much effect of `__builtin_prefetch` and the performance we see is not so different from that of blocking.

<br>

---
## Bonus Task 3: SIMD instructions + Software Prefetching
### Implementation:
- This combines the idea of software prefetching and SIMD instructions.
- We aim to fetch the blocks that are accessed in current loop at the begin before the elements are actually accessed. We expect this would improve the cache hits when performing the multiplications.

### Observation:
- We have recorded the execution time, data cache miss rate and speed up for different matrix sizes and represented in the table below:
  
  ![image](https://github.com/cs683-iitb-autumn-2023/pa1-the-matrix-opcodeoutlaws/assets/48720143/b9a5cc72-cffe-4cea-9f37-f20afc28b465)

### Analysis:
- The performance is not all that different from SIMD multiplications.
- Given our assumptions in the previous sections, we have not seen a significant improvement after adding software prefetching.

### Limitations:
- Given this uses the logic from SIMD matrix multiplication code, this implementation only works for matrix dimensions which are multiples of 4.

<br>

---
## Bonus Task 4: Bloced Matrix Multiplication + SIMD instructions + Software Prefetching
### Implementation:
- Here we combine all the ideas for optimization
- We aim to reduce the miss rates significantly beacause of blocking while prefetching the blocks which are about to be accessed all the while reducing the total number of actual multiplication operations.


### Observation:
- We have recorded the execution time, data cache miss rate and speed up for different matrix sizes and represented in the table below:
  
  ![image](https://github.com/cs683-iitb-autumn-2023/pa1-the-matrix-opcodeoutlaws/assets/48720143/3f5b0764-68f7-4a9e-ab8b-20769c1b6bbc)

### Analysis:
- The performance is not very different from SIMD+Blocking as we have seen this recurring pattern of software prefetching not providing a significant boost.
- The speedup decreases as the matrix dimension increases.

### Limitations:
- Here we have limitations of SIMD implementation intersected with limitations of blocking implementation.
- The dimension of matrix should be divisible by 4 and also be divisible by `BLOCK_SIZE`. In order to satisfy the requirement of optimizing matrix multiplication of dimensions 100, 200 and 800, we have decided the `BLOCK_SIZE` to be 20.

<br>


## All optimizations comparison
- The below chart is plotted cosidering the highest speedup we acheived using all techniques and cache block size of 64 Bytes
  
  ![image](https://github.com/cs683-iitb-autumn-2023/pa1-the-matrix-opcodeoutlaws/assets/48720143/1e6d10ff-6a46-445c-9b8b-37d7c0b9a227)


- Here we can observe that we have acheived the best speedup at matrix dimension 100 for BLOCKING+SIMD.
- Even though we expected the BLOCKING+SIMD+PREFETCHING to provide the best speedup, we have observed that the PREFETCH has been a limiting factor because of pre-optimizations from compiler.
- The speedup decreases as the matrix dimension increases. We estimate this to be beacause of the fact that as the dimension increases we start receiving more and more page faults as the data that needs to be brought into the cache is so huge. Hence the speedup factor falls short against the large penalties of page faults.
