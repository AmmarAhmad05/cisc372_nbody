#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "vector.h"
#include "config.h"

// kernel 1: each thread fills one cell of the accels matrix
__global__ void compute_accels_kernel(vector3 *accels, vector3 *pos, double *mass) {
int i = blockIdx.y * blockDim.y + threadIdx.y;
int j = blockIdx.x * blockDim.x + threadIdx.x;

if (i >= NUMENTITIES || j >= NUMENTITIES) return;

int idx = i * NUMENTITIES + j;

if (i == j) {
accels[idx][0] = 0;
accels[idx][1] = 0;
accels[idx][2] = 0;
} else {
vector3 distance;
for (int k = 0; k < 3; k++) distance[k] = pos[i][k] - pos[j][k];
double magnitude_sq = distance[0]*distance[0] + distance[1]*distance[1] + distance[2]*distance[2];
double magnitude = sqrt(magnitude_sq);
double accelmag = -1 * GRAV_CONSTANT * mass[j] / magnitude_sq;
accels[idx][0] = accelmag * distance[0] / magnitude;
accels[idx][1] = accelmag * distance[1] / magnitude;
accels[idx][2] = accelmag * distance[2] / magnitude;
}
}

// kernel 2: each thread sums one row and updates pos/vel for that entity
__global__ void update_pos_vel_kernel(vector3 *accels, vector3 *pos, vector3 *vel) {
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i >= NUMENTITIES) return;

vector3 accel_sum = {0, 0, 0};
for (int j = 0; j < NUMENTITIES; j++) {
int idx = i * NUMENTITIES + j;
accel_sum[0] += accels[idx][0];
accel_sum[1] += accels[idx][1];
accel_sum[2] += accels[idx][2];
}

for (int k = 0; k < 3; k++) {
vel[i][k] += accel_sum[k] * INTERVAL;
pos[i][k] += vel[i][k] * INTERVAL;
}
}

// host-side compute: launches the two kernels
void compute() {
dim3 blockDim(16, 16);
dim3 gridDim((NUMENTITIES + 15) / 16, (NUMENTITIES + 15) / 16);
compute_accels_kernel<<<gridDim, blockDim>>>(d_accels, d_hPos, d_mass);

int threads = 256;
int blocks = (NUMENTITIES + threads - 1) / threads;
update_pos_vel_kernel<<<blocks, threads>>>(d_accels, d_hPos, d_hVel);
}
