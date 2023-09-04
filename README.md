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

![image](https://github.com/cs683-iitb-autumn-2023/pa1-the-matrix-opcodeoutlaws/assets/48720143/2e8beddc-b533-4f3c-a41c-73689dd41c6f)


Here is the list of SIMD functions we are using for our implementation:
1. `_mm_setzero_pd`
2. `_mm_loadu_pd`
3. `_mm_mul_pd`
4. `_mm_add_pd`
5. `_mm_shuffle_pd`

Informaton about each of them can be found here: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html

#### Implementation
![image](https://github.com/cs683-iitb-autumn-2023/pa1-the-matrix-opcodeoutlaws/assets/48720143/2a102fcc-6553-4ebd-b08c-792f8733c9ee)


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
