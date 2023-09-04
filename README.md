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
- We have divided the matrices into small block and trying to reuse the blocks fetched in the cache
- To implement it we are using 6 for loops, where first three loops select which blocks to multiply and last three loops use normal matrix multiplication to multiply the selected blocks.
#### VS Normal Matrix Multiplication
- In normal matrix multiplication, while performing C = A * B, we access B in column-major order. If the size of a column is greater than the cache size whatever block of matrix B that will be brought into the cache while fetching elements of matrix B will be replaced and we will be facing a cache miss for every element access of matrix B
  
  ![image](https://github.com/cs683-iitb-autumn-2023/pa1-the-matrix-opcodeoutlaws/assets/142027995/ede6508f-0696-43a7-9d86-9fdf92642ae7)

- The block that were brought into the cache while accessing B are not used expect for the 1st element. So the idea of Blocked matrix multiplication is to use the other elements of that block before that block is evicted from the cache.  
#### Miss rate reduction using Blocking
- Now in blocking the martix will be divided into small blocks (lets say of size B) and these will blocks will be multiplied with each other instead of whole matrix.

  ![image](https://github.com/cs683-iitb-autumn-2023/pa1-the-matrix-opcodeoutlaws/assets/142027995/45c4b7ea-f108-49e5-9944-ad765929dd9b)


<br>

---
## Task 2: SIMD instructions

<br>

---
## Task 3: Software Prefetching

<br>

---
## Bonus Task 1: Blocked Matrix Multiplication + SIMD instructions

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
