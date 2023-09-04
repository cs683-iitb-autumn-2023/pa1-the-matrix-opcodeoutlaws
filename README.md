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

<br>

---
## Task 2: SIMD instructions
### Specifics
#### SIMD Registers
We are using 128 bits wide registers for implementing SIMD instructions. One double is of 64 bits in size. Hence we can store two doubles adjacent to each other in a 64 bit register. Here's how it looks:

![image](https://github.com/cs683-iitb-autumn-2023/pa1-the-matrix-opcodeoutlaws/assets/48720143/d8dc7b55-bf40-43b4-ac1e-56314aa375ab)

#### SIMD Instructions
Here we'll see how SIMD multiplication works when we have two `_m128d` registers containing a total of four doubles.

![image](https://github.com/cs683-iitb-autumn-2023/pa1-the-matrix-opcodeoutlaws/assets/48720143/e6c2c930-b961-4868-b713-b122b7387529)

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
- We load 16 elements from matrix B into 8 `_128d` registers.
- We need to shuffle the registers to align them for multiplication as shown in above figure. We use  `_mm_shuffle_pd` for the same.
- Next step is to multiply the aligned registers. This gives us multiplication of `A11` × `B11` and `A12` × `B12`, `A11` × `B21` and `A12` × `B22` and so on.
- We realign the registers using `_mm_shuffle_pd` for addition so that we can get `A11 × B11`+ `A12 × B12` and so on.
- We have applied loop unrolling to get some additional boost in performance.
- Finally we store the result to C11, C12, C13 & C14 as two registers using `_mm_storeu_pd`

#### Execution time & Speedup
We have seen speedup ranging from 3x to 6x on different executions. On average the speed up is about `3.5`.

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
