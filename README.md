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
  
  ![image](https://github.com/cs683-iitb-autumn-2023/pa1-the-matrix-opcodeoutlaws/assets/142027995/efc86f32-0df8-4255-8301-fa0a5cac72a7)

### Analysis:
- It can seen from the table that we have a significant decrease in the miss rate specially for matrix of dimension 800. This is because the single column of martix will be very large. And each element access will bring a block (of size 64B) into the cache. So accessing just one column of the matrix B will be very large (800 * 64B) and won't fit completely in the cache. So the blocks fetched will be evicted and would result in unnecessary cache misses.
- This won't happen for matrix of dimension 100 and 200 as their coloumns will be of smaller size. Because of this we can see a high miss rate (4.5 %) for normal matrix multiplication in 800 dimension matrix. Blocking reduces the miss rate to 0.1 % in each case by reusing the fetched blocks and gives a good speedup. 

<br>

---
## Task 2: SIMD instructions
### Implementation
#### SIMD Registers
- We are using 128 bits wide registers for implementing SIMD instructions. One double is of 64 bits in size. Hence we can store two doubles adjacent to each other in a 64 bit register. Here's how it looks:

![image](https://github.com/cs683-iitb-autumn-2023/pa1-the-matrix-opcodeoutlaws/assets/48720143/d5b47259-464c-4fa0-b2c5-a79b6944ab99)


#### SIMD Instructions
- Here we'll see how SIMD multiplication works when we have two `_m128d` registers containing a total of four doubles.

  ![image](https://github.com/cs683-iitb-autumn-2023/pa1-the-matrix-opcodeoutlaws/assets/48720143/7f2f49d2-2b00-4f35-9b16-86c7a78ae041)


Here is the list of SIMD functions we are using for our implementation:
1. `_mm_setzero_pd`
2. `_mm_loadu_pd`
3. `_mm_mul_pd`
4. `_mm_add_pd`
5. `_mm_shuffle_pd`
6. `_mm_storeu_pd`

Informaton about each of them can be found here: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html

#### Implementation
  ![image](https://github.com/cs683-iitb-autumn-2023/pa1-the-matrix-opcodeoutlaws/assets/48720143/2b852f03-8f50-485b-a864-45ed41abd267)
Above is a snippet of how C1 and C2 is calculated in one step.
- We first load A11, A12 in one `_m128d` register. Alongside we also load A21, A22 in another `_m128d` register. This is done using the `_mm_loadu_pd`.
- We load 16 elements from matrix B into 8 `_m128d` registers.
- We need to shuffle the registers to align them for multiplication as shown in above figure. We use  `_mm_shuffle_pd` for the same.
- Next step is to multiply the aligned registers. This gives us multiplication of `A11` × `B11` and `A12` × `B12`, `A11` × `B21` and `A12` × `B22` and so on.
- We realign the registers using `_mm_shuffle_pd` for addition so that we can get `A11 × B11`+ `A12 × B12` and so on.
- We have applied loop unrolling to get some additional boost in performance.
- Finally we store the result to C11, C12, C13 & C14 as two registers using `_mm_storeu_pd`

#### Observations

![image](https://github.com/cs683-iitb-autumn-2023/pa1-the-matrix-opcodeoutlaws/assets/142027995/453cd3e1-2aa7-4f40-9a56-c65c55b43b52)


#### Analysis


#### Limitations
- Because of some loop unrolling and the way we have accessed elements using `_mm_storeu_pd`, the matrix multiplication works for dimensions which are multiples of 4.

<br>

---
## Task 3: Software Prefetching

<br>

---
## Bonus Task 1: Blocked Matrix Multiplication + SIMD instructions
- This implementation takes advantage of both: blocking and simd instructions. We aim to reduce the number of multiplication operations and also reduce the miss rate with this implementation.

#### Implementation
- We have started with the approach that we used for blocking
- And we changed the actual matrix multiplication taking place in the innermost for loop of blocking implementation with SIMD based multiplication.
- Some amount of loop unrolling is also done to acheive greater performance.

#### Speedup and performance


#### Observations

![image](https://github.com/cs683-iitb-autumn-2023/pa1-the-matrix-opcodeoutlaws/assets/142027995/36665387-0a41-425c-b558-934f60e58384)


#### Limitations
- Here we have limitations of SIMD implementation intersected with limitations of blocking implementation.
- The dimension of matrix should be divisible by 4 and also be divisible by `BLOCK_SIZE`. In order to satisfy the requirement of optimizing matrix multiplication of dimensions 100, 200 and 800, we have decided the `BLOCK_SIZE` to be 20.
<br>

---
## Bonus Task 2: Blocked Matrix Multiplication + Software Prefetching

<br>

---
## Bonus Task 3: SIMD instructions + Software Prefetching

<br>

---
## Bonus Task 4: Bloced Matrix Multiplication + SIMD instructions + Software Prefetching

<br>

---
All the best! :smile:
