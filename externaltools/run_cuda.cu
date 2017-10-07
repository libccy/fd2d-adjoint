#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "ArduinoJson.h"

#define devij int i = blockIdx.x, j = threadIdx.x + blockIdx.y * blockDim.x

const float pi = 3.1415927;
const int nbt = 1;
__constant__ float d_pi = 3.1415927;

namespace dat{
    int nx;
    int nz;
    int nt;
    float dx;
    float dz;
    float dt;
    float Lx;
    float Lz;

    int sfe;
    int nsfe;
    int order;
    int model_type;
    int wave_propagation_sh;
    int wave_propagation_psv;
    int simulation_mode;
    int use_given_model;
    int use_given_stf;
    float source_amplitude;

    int absorb_left;
    int absorb_right;
    int absorb_top;
    int absorb_bottom;
    float absorb_width;

    int isrc;
    int nsrc;
    int nrec;
    int obs_type;

    float *tw;

    int *stf_type;  // host
    float *stf_PSV_x;  // host
    float *stf_PSV_z;  // host
    float *tauw_0;  // host
    float *tauw;  // host
    float *tee_0;  // host
    float *f_min;  // host
    float *f_max;  // host

    float *src_x;
    float *src_z;
    float *rec_x;
    float *rec_z;

    int *src_x_id;
    int *src_z_id;
    int *rec_x_id;
    int *rec_z_id;

    float **stf_x;
    float **stf_y;
    float **stf_z;
    float **adstf_x;
    float **adstf_y;
    float **adstf_z;

    float **lambda;
    float **mu;
    float **rho;
    float **absbound;

    float **ux;
    float **uy;
    float **uz;
    float **vx;
    float **vy;
    float **vz;

    float **sxx;
    float **sxy;
    float **sxz;
    float **szy;
    float **szz;

    float **dsx;
    float **dsy;
    float **dsz;
    float **dvxdx;
    float **dvxdz;
    float **dvydx;
    float **dvydz;
    float **dvzdx;
    float **dvzdz;

    float **dvxdx_fw;
    float **dvxdz_fw;
    float **dvydx_fw;
    float **dvydz_fw;
    float **dvzdx_fw;
    float **dvzdz_fw;

    float **K_lambda;
    float **K_mu;
    float **K_rho;

    float **v_rec_x;
    float **v_rec_y;
    float **v_rec_z;

    float ***u_obs_x;
    float ***u_obs_y;
    float ***u_obs_z;

    float ***ux_forward;  // host
    float ***uy_forward;  // host
    float ***uz_forward;  // host
    float ***vx_forward;  // host
    float ***vy_forward;  // host
    float ***vz_forward;  // host

    int sigma;
    float **gsum;
    float **gtemp;

    float misfit_init;
    float **lambda_start;
    float **mu_start;
    float **rho_start;
}
namespace mat{
    __global__ void _setValue(float *mat, const float init){
        int i = blockIdx.x;
        mat[i] = init;
    }
    __global__ void _setValue(float **mat, const float init){
        devij;
        mat[i][j] = init;
    }
    __global__ void _setValue(float ***mat, const float init, const int p){
        devij;
        mat[p][i][j] = init;
    }
    __global__ void _setPointerValue(float **mat, float *data, const int n){
        int i = blockIdx.x;
        mat[i] = data + n * i;
    }
    __global__ void _setPointerValue(float ***mat, float **data, const int i){
        mat[i] = data;
    }
    __global__ void _copy(float **mat, float **init){
        devij;
        mat[i][j] = init[i][j];
    }


    float *init(float *mat, const int m, const float init){
        mat::_setValue<<<m, 1>>>(mat, init);
        return mat;
    }
    float **init(float **mat, const int m, const int n, const float init){
        dim3 dimGrid(m, nbt);
        mat::_setValue<<<dimGrid, n / nbt>>>(mat, init);
        return mat;
    }
    float ***init(float ***mat, const int p, const int m, const int n, const float init){
        dim3 dimGrid(m, nbt);
        for(int i = 0; i < p; i++){
            mat::_setValue<<<dimGrid, n / nbt>>>(mat, init, i);
        }
        return mat;
    }
    float *initHost(float *mat, const int m, const float init){
        for(int i = 0; i < m; i++){
            mat[i] = init;
        }
        return mat;
    }
    float **initHost(float **mat, const int m, const int n, const float init){
        for(int i = 0; i < m; i++){
            for(int j = 0; j < n; j++){
                mat[i][j] = init;
            }
        }
        return mat;
    }
    float ***initHost(float ***mat, const int p, const int m, const int n, float init){
        for(int k = 0; k < p; k++){
            for(int i = 0; i < m; i++){
                for(int j = 0; j < n; j++){
                    mat[k][i][j] = init;
                }
            }
        }
        return mat;
    }

    float *create(const int m) {
    	float *data;
    	cudaMalloc((void **)&data, m * sizeof(float));
    	return data;
    }
    float **create(const int m, const int n){
    	float *data = mat::create(m * n);
        float **mat;
        cudaMalloc((void **)&mat, m * sizeof(float *));
        mat::_setPointerValue<<<m, 1>>>(mat, data, n);
    	return mat;
    }
    float ***create(const int p, const int m, const int n){
        float ***mat;
        cudaMalloc((void **)&mat, p * sizeof(float **));
        for(int i = 0; i < p; i++){
            mat::_setPointerValue<<<1,1>>>(mat, mat::create(m, n), i);
        }
        return mat;
    }
    float *createHost(const int m) {
    	return (float *)malloc(m * sizeof(float));
    }
    float **createHost(const int m, const int n){
        float *data = mat::createHost(m * n);
    	float **mat = (float **)malloc(m * sizeof(float *));
    	for(int i  =0; i < m; i++){
    		mat[i] = data + n * i;
    	}
    	return mat;
    }
    float ***createHost(const int p, const int m, const int n){
        float ***mat = (float ***)malloc(p * sizeof(float **));
        for(int i = 0; i < p; i++){
            mat[i] = mat::createHost(m, n);
        }
        return mat;
    }
    int *createInt(const int m){
        int *a;
    	cudaMalloc((void**)&a, m * sizeof(int));
    	return a;
    }
    int *createIntHost(const int m) {
    	return (int *)malloc(m * sizeof(int));
    }

    void copy(float **mat, float **init, const int m, const int n){
        dim3 dimGrid(m, nbt);
        mat::_copy<<<dimGrid, n / nbt>>>(mat, init);
    }
    void copyHostToDevice(float *d_a, const float *a, const int m){
        cudaMemcpy(d_a, a , m * sizeof(float), cudaMemcpyHostToDevice);
    }
    void copyHostToDevice(float **pd_a, float *pa, const int m, const int n){
        float **phd_a=(float **)malloc(sizeof(float *));
        cudaMemcpy(phd_a, pd_a , sizeof(float *), cudaMemcpyDeviceToHost);
        cudaMemcpy(*phd_a, pa , m * n * sizeof(float), cudaMemcpyHostToDevice);
    }
    void copyHostToDevice(float **pd_a, float **pa, const int m, const int n){
        float **phd_a=(float **)malloc(sizeof(float *));
        cudaMemcpy(phd_a, pd_a , sizeof(float *), cudaMemcpyDeviceToHost);
        cudaMemcpy(*phd_a, *pa , m * n * sizeof(float), cudaMemcpyHostToDevice);
    }
    void copyHostToDevice(float ***pd_a, float ***pa, const int p, const int m, const int n){
        float ***phd_a=(float ***)malloc(p * sizeof(float **));
        cudaMemcpy(phd_a, pd_a, p * sizeof(float **), cudaMemcpyDeviceToHost);
        for(int i = 0; i < p; i++){
            mat::copyHostToDevice(phd_a[i], pa[i], m, n);
        }
    }
    void copyDeviceToHost(float *a, const float *d_a, const int m){
        cudaMemcpy(a, d_a , m * sizeof(float), cudaMemcpyDeviceToHost);
    }
    void copyDeviceToHost(float *pa, float **pd_a, const int m, const int n){
        float **phd_a=(float **)malloc(sizeof(float *));
        cudaMemcpy(phd_a, pd_a , sizeof(float *), cudaMemcpyDeviceToHost);
        cudaMemcpy(pa, *phd_a , m * n * sizeof(float), cudaMemcpyDeviceToHost);
    }
    void copyDeviceToHost(float **pa, float **pd_a, const int m, const int n){
        float **phd_a=(float **)malloc(sizeof(float *));
        cudaMemcpy(phd_a, pd_a , sizeof(float *), cudaMemcpyDeviceToHost);
        cudaMemcpy(*pa, *phd_a , m * n * sizeof(float), cudaMemcpyDeviceToHost);
    }
    void copyDeviceToHost(float ***pa, float ***pd_a, const int p, const int m, const int n){
        float ***phd_a=(float ***)malloc(p * sizeof(float **));
        cudaMemcpy(phd_a, pd_a, p * sizeof(float **), cudaMemcpyDeviceToHost);
        for(int i = 0; i < p; i++){
            mat::copyDeviceToHost(pa[i], phd_a[i], m, n);
        }
    }

    void read(float *data, int n, char *fname){
        char buffer[50] = "externaltools/";
        strcat(buffer, fname);
        FILE *file = fopen(buffer, "rb");
        fwrite(data, sizeof(float), n, file);
        fclose(file);
    }
    void write(float *data, int n, char *fname){
        char buffer[50] = "externaltools/";
        strcat(buffer, fname);
        FILE *file = fopen(buffer, "wb");
        fwrite(data, sizeof(float), n, file);
        fclose(file);
    }
    void write(float **data, int m, int n, char *fname){
        char buffer[50] = "externaltools/";
        strcat(buffer, fname);
        FILE *file = fopen(buffer, "wb");
        for(int i = 0; i < m; i++){
            fwrite(data[i], sizeof(float), n, file);
        }
        fclose(file);
    }
    void write(float ***data, int p, int m, int n, char *fname){
        char buffer[50] = "externaltools/";
        strcat(buffer, fname);
        FILE *file = fopen(buffer, "wb");
        for(int k = 0; k < p; k++){
            for(int i = 0; i < m; i++){
                fwrite(data[k][i], sizeof(float), n, file);
            }
        }
        fclose(file);
    }
}

__global__ void divSY(float **dsy, float **sxy, float **szy, float dx, float dz, int nx, int nz){
    devij;
    if(i >= 2 && i < nx - 2){
        dsy[i][j] = 9*(sxy[i][j]-sxy[i-1][j])/(8*dx)-(sxy[i+1][j]-sxy[i-2][j])/(24*dx);
    }
    else{
        dsy[i][j] = 0;
    }
    if(j >= 2 && j < nz - 2){
        dsy[i][j] += 9*(szy[i][j]-szy[i][j-1])/(8*dz)-(szy[i][j+1]-szy[i][j-2])/(24*dz);
    }
}
__global__ void divSXZ(float **dsx, float **dsz, float **sxx, float **szz, float **sxz, float dx, float dz, int nx, int nz){
    devij;
    if(i >= 2 && i < nx - 2){
        dsx[i][j] = 9*(sxx[i][j]-sxx[i-1][j])/(8*dx)-(sxx[i+1][j]-sxx[i-2][j])/(24*dx);
        dsz[i][j] = 9*(sxz[i][j]-sxz[i-1][j])/(8*dx)-(sxz[i+1][j]-sxz[i-2][j])/(24*dx);
    }
    else{
        dsx[i][j] = 0;
        dsz[i][j] = 0;
    }
    if(j >= 2 && j < nz - 2){
        dsx[i][j] += 9*(sxz[i][j]-sxz[i][j-1])/(8*dz)-(sxz[i][j+1]-sxz[i][j-2])/(24*dz);
        dsz[i][j] += 9*(szz[i][j]-szz[i][j-1])/(8*dz)-(szz[i][j+1]-szz[i][j-2])/(24*dz);
    }
}
__global__ void divVY(float **dvydx, float **dvydz, float **vy, float dx, float dz, int nx, int nz){
    devij;
    if(i >= 1 && i < nx - 2){
        dvydx[i][j] = 9*(vy[i+1][j]-vy[i][j])/(8*dx)-(vy[i+2][j]-vy[i-1][j])/(24*dx);
    }
    else{
        dvydx[i][j] = 0;
    }
    if(j >= 1 && j < nz - 2){
        dvydz[i][j] = 9*(vy[i][j+1]-vy[i][j])/(8*dz)-(vy[i][j+2]-vy[i][j-1])/(24*dz);
    }
    else{
        dvydz[i][j] = 0;
    }
}
__global__ void divVXZ(float **dvxdx, float **dvxdz, float **dvzdx, float **dvzdz, float **vx, float **vz, float dx, float dz, int nx, int nz){
    devij;
    if(i >= 1 && i < nx - 2){
        dvxdx[i][j] = 9*(vx[i+1][j]-vx[i][j])/(8*dx)-(vx[i+2][j]-vx[i-1][j])/(24*dx);
        dvzdx[i][j] = 9*(vz[i+1][j]-vz[i][j])/(8*dx)-(vz[i+2][j]-vz[i-1][j])/(24*dx);
    }
    else{
        dvxdx[i][j] = 0;
        dvzdx[i][j] = 0;
    }
    if(j >= 1 && j < nz - 2){
        dvxdz[i][j] = 9*(vx[i][j+1]-vx[i][j])/(8*dz)-(vx[i][j+2]-vx[i][j-1])/(24*dz);
        dvzdz[i][j] = 9*(vz[i][j+1]-vz[i][j])/(8*dz)-(vz[i][j+2]-vz[i][j-1])/(24*dz);
    }
    else{
        dvxdz[i][j] = 0;
        dvzdz[i][j] = 0;
    }
}

__global__ void addSTF(float **dsx, float **dsy, float **dsz, float **stf_x, float **stf_y, float **stf_z,
    int *src_x_id, int *src_z_id, int isrc, int sh, int psv, int it){
    int is = blockIdx.x;
    int xs = src_x_id[is];
    int zs = src_z_id[is];
    if(isrc < 0 || isrc == is){
        if(sh){
            dsy[xs][zs] += stf_y[is][it];
        }
        if(psv){
            dsx[xs][zs] += stf_x[is][it];
            dsz[xs][zs] += stf_z[is][it];
        }
    }
}
__global__ void saveV(float **v_rec_x, float **v_rec_y, float **v_rec_z, float **vx, float **vy, float **vz,
    int *rec_x_id, int *rec_z_id, int sh, int psv, int it){
    int ir = blockIdx.x;
    int xr = rec_x_id[ir];
    int zr = rec_z_id[ir];
    if(sh){
        v_rec_y[ir][it] = vy[xr][zr];
    }
    if(psv){
        v_rec_x[ir][it] = vx[xr][zr];
        v_rec_z[ir][it] = vz[xr][zr];
    }
}
__global__ void saveV(float ***v_rec_x, float ***v_rec_y, float ***v_rec_z, float **vx, float **vy, float **vz,
    int *rec_x_id, int *rec_z_id, int isrc, int sh, int psv, int it){
    int ir = blockIdx.x;
    int xr = rec_x_id[ir];
    int zr = rec_z_id[ir];
    if(sh){
        v_rec_y[isrc][ir][it] = vy[xr][zr];
    }
    if(psv){
        v_rec_x[isrc][ir][it] = vx[xr][zr];
        v_rec_z[isrc][ir][it] = vz[xr][zr];
    }
}
__global__ void updateV(float **v, float **ds, float **rho, float **absbound, float dt){
    devij;
    v[i][j] = absbound[i][j] * (v[i][j] + dt * ds[i][j] / rho[i][j]);
}
__global__ void updateSY(float **sxy, float **szy, float **dvydx, float **dvydz, float **mu, float dt){
    devij;
    sxy[i][j] += dt * mu[i][j] * dvydx[i][j];
    szy[i][j] += dt * mu[i][j] * dvydz[i][j];
}
__global__ void updateSXZ(float **sxx, float **szz, float **sxz, float **dvxdx, float **dvxdz, float **dvzdx, float **dvzdz,
    float **lambda, float **mu, float dt){
    devij;
    sxx[i][j] += dt * ((lambda[i][j] + 2 * mu[i][j]) * dvxdx[i][j] + lambda[i][j] * dvzdz[i][j]);
    szz[i][j] += dt * ((lambda[i][j] + 2 * mu[i][j]) * dvzdz[i][j] + lambda[i][j] * dvxdx[i][j]);
    sxz[i][j] += dt * (mu[i][j] * (dvxdz[i][j] + dvzdx[i][j]));
}
__global__ void updateU(float **u, float **v, float dt){
    devij;
    u[i][j] += v[i][j] * dt;
}
__global__ void interactionRhoY(float **K_rho, float **vy, float **vy_fw, float tsfe){
    devij;
    K_rho[i][j] -= vy_fw[i][j] * vy[i][j] * tsfe;
}
__global__ void interactionRhoXZ(float **K_rho, float **vx, float **vx_fw, float **vz, float **vz_fw, float tsfe){
    devij;
    K_rho[i][j] -= (vx_fw[i][j] * vx[i][j] + vz_fw[i][j] * vz[i][j]) * tsfe;
}
__global__ void interactionMuY(float **K_mu, float **dvydx, float **dvydx_fw, float **dvydz, float **dvydz_fw, float tsfe){
    devij;
    K_mu[i][j] -= (dvydx[i][j] * dvydx_fw[i][j] + dvydz[i][j] * dvydz_fw[i][j]) * tsfe;
}
__global__ void interactionMuXZ(float **K_mu, float **dvxdx, float **dvxdx_fw, float **dvxdz, float **dvxdz_fw,
    float **dvzdx, float **dvzdx_fw, float **dvzdz, float **dvzdz_fw, float tsfe){
    devij;
    K_mu[i][j] -= (2 * dvxdx[i][j] * dvxdx_fw[i][j] + 2 * dvzdz[i][j] * dvzdz_fw[i][j] +
        (dvxdz[i][j] + dvzdx[i][j]) * (dvzdx_fw[i][j] + dvxdz_fw[i][j])) * tsfe;
}
__global__ void interactionLambdaXZ(float **K_lambda, float **dvxdx, float **dvxdx_fw, float **dvzdz, float **dvzdz_fw, float tsfe){
    devij;
    K_lambda[i][j] -= ((dvxdx[i][j] + dvzdz[i][j]) * (dvxdx_fw[i][j] + dvzdz_fw[i][j])) * tsfe;
}

__global__ void computeIndices(int *coord_n_id, float *coord_n, float Ln, float n){
    int i = blockIdx.x;
    coord_n_id[i] = (int)(coord_n[i] / Ln * (n - 1) + 0.5);
}
__global__ void initialiseAbsorbingBoundaries(float **absbound, float width,
    int absorb_left, int absorb_right, int absorb_bottom, int absorb_top,
    float Lx, float Lz, float dx, float dz){
    devij;
    absbound[i][j] = 1;

    float X = i * dx;
    float Z = j * dz;
    if(absorb_left){
        if(X < width){
            absbound[i][j] *= exp(-pow((X - width) / (2 * width), 2));
        }
    }
    if(absorb_right){
        if(X > Lx - width){
            absbound[i][j] *= exp(-pow((X - (Lx - width)) / (2 * width), 2));
        }
    }
    if(absorb_bottom){
        if(Z < width){
            absbound[i][j] *= exp(-pow((Z - width) / (2 * width), 2));
        }
    }
    if(absorb_top){
        if(Z > Lz - width){
            absbound[i][j] *= exp(-pow((Z - (Lz - width)) / (2 * width), 2));
        }
    }
}
__global__ void prepareAdjointSTF(float **adstf, float **u_syn, float ***u_obs, float *tw, int nt, int isrc){
    int it = blockIdx.x;
    int irec = threadIdx.x;
    adstf[irec][nt - it - 1] = (u_syn[irec][it] - u_obs[isrc][irec][it]) * tw[it] * 2;
}
__global__ void normKernel(float **model, float **model_start, float misfit_init){
    devij;
    model[i][j] *= model_start[i][j] / misfit_init;
}
__device__ float gaussian(int x, int sigma){
    float xf = (float)x;
    float sigmaf = (float)sigma;
    return (1 / (sqrtf(2 * d_pi) * sigmaf)) * expf(-xf * xf / (2 * sigmaf * sigmaf));
}
__global__ void initialiseGaussian(float **model, int nx, int nz, int sigma){
    devij;
    float sumx = 0;
    for(int n = 0; n < nx; n++){
        sumx += gaussian(i - n, sigma);
    }
    float sumz = 0;
    for(int n = 0; n < nz; n++){
        sumz += gaussian(j - n, sigma);
    }
    model[i][j] = sumx * sumz;
}
__global__ void filterKernelX(float **model, float **gtemp, int nx, int sigma){
    devij;
    float sumx = 0;
    for(int n = 0; n < nx; n++){
        sumx += gaussian(i - n, sigma) * model[n][j];
    }
    gtemp[i][j] = sumx;
}
__global__ void filterKernelZ(float **model, float **gtemp, float **gsum, int nz, int sigma){
    devij;
    float sumz = 0;
    for(int n = 0; n < nz; n++){
        sumz += gaussian(j - n, sigma) * gtemp[i][n];
    }
    model[i][j] = sumz / gsum[i][j];
}
__global__ void updateModel(float **model, float **kernel, float step){
    devij;
    model[i][j] *= (1 - step * kernel[i][j]);
}
__global__ void getTaperWeights(float *tw, float dt, int nt){
    int it = blockIdx.x;

    float t_end = (nt - 1) * dt;
    float taper_width = t_end / 10;
    float t_min = taper_width;
    float t_max = t_end - taper_width;

    float t = it * dt;
    if(t <= t_min){
        tw[it] = 0.5 + 0.5 * cosf(d_pi * (t_min - t) / (taper_width));
    }
    else if(t >= t_max){
        tw[it] = 0.5 + 0.5 * cosf(d_pi * (t_max - t) / (taper_width));
    }
    else{
        tw[it] = 1;
    }
}
__global__ void calculateMisfit(float *misfit, float **u_syn, float ***u_obs, float *tw, float dt, int isrc, int irec){
    int it = blockIdx.x;
    float wavedif = (u_syn[irec][it] - u_obs[isrc][irec][it]) * tw[it];
    misfit[it] += wavedif * wavedif * dt;
}

static void importData(){
    FILE *datfile = fopen("externaltools/config","r");

    char *buffer = 0;
    long length;

    fseek (datfile, 0, SEEK_END);
    length = ftell (datfile);
    fseek (datfile, 0, SEEK_SET);
    buffer = (char *)malloc (length + 1);
    fread (buffer, 1, length, datfile);
    buffer[length] = '\0';

    fclose(datfile);

    if (buffer){
        DynamicJsonBuffer jsonBuffer;
        JsonObject& root = jsonBuffer.parseObject(buffer);
        if (!root.success()){
            printf("parseObject() failed\n");
        }
        else{
            dat::nx = root["nx"];
            dat::nz = root["nz"];
            dat::nt = root["nt"];
            dat::dt = root["dt"];
            dat::Lx = root["Lx"];
            dat::Lz = root["Lz"];
            dat::sfe = root["sfe"];

            dat::model_type = root["model_type"];
            dat::use_given_model = root["use_given_model"];
            dat::use_given_stf = root["use_given_stf"];
            dat::source_amplitude = root["source_amplitude"];
            dat::order = root["order"]; // order = 2: later
            dat::obs_type = root["obs_type"];

            dat::absorb_left = root["absorb_left"];
            dat::absorb_right = root["absorb_right"];
            dat::absorb_top = root["absorb_top"];
            dat::absorb_bottom = root["absorb_bottom"];
            dat::absorb_width = root["width"];

            {
                const char* wave_propagation_type = root["wave_propagation_type"].as<char*>();
                if(strcmp(wave_propagation_type,"SH") == 0){
                    dat::wave_propagation_sh = 1;
                    dat::wave_propagation_psv = 0;
                }
                else if(strcmp(wave_propagation_type,"PSV") == 0){
                    dat::wave_propagation_sh = 0;
                    dat::wave_propagation_psv = 1;
                }
                else if(strcmp(wave_propagation_type,"both") == 0){
                    dat::wave_propagation_sh = 1;
                    dat::wave_propagation_psv = 1;
                }
                else{
                    dat::wave_propagation_sh = 0;
                    dat::wave_propagation_psv = 0;
                }
            }

            {
                int single_src = root["src_info"].is<JsonObject>();
                dat::nsrc = single_src?1:root["src_info"].size();

                float *src_x = mat::createHost(dat::nsrc);
                float *src_z = mat::createHost(dat::nsrc);

                dat::stf_type = mat::createIntHost(dat::nsrc);
                dat::stf_PSV_x = mat::createHost(dat::nsrc);
                dat::stf_PSV_z = mat::createHost(dat::nsrc);
                dat::tauw_0 = mat::createHost(dat::nsrc);
                dat::tauw = mat::createHost(dat::nsrc);
                dat::tee_0 = mat::createHost(dat::nsrc);
                dat::f_min = mat::createHost(dat::nsrc);
                dat::f_max = mat::createHost(dat::nsrc);

                for(int isrc = 0; isrc < dat::nsrc; isrc++){
                    JsonObject& src = single_src?root["src_info"]:((JsonArray&)root["src_info"]).get<JsonObject>(isrc);
                    src_x[isrc] = src["loc_x"];
                    src_z[isrc] = src["loc_z"];

                    dat::stf_PSV_x[isrc] = src["stf_PSV"][0];
                    dat::stf_PSV_z[isrc] = src["stf_PSV"][1];
                    dat::tauw_0[isrc] = src["tauw_0"];
                    dat::tauw[isrc] = src["tauw"];
                    dat::tee_0[isrc] = src["tee_0"];
                    dat::f_min[isrc] = src["f_min"];
                    dat::f_max[isrc] = src["f_max"];

                    const char* stf_type_str = src["stf_type"].as<char*>();
                    if(strcmp(stf_type_str,"delta") == 0){
                        dat::stf_type[isrc] = 0;
                    }
                    else if(strcmp(stf_type_str,"delta_bp") == 0){
                        dat::stf_type[isrc] = 1;
                    }
                    else if(strcmp(stf_type_str,"ricker") == 0){
                        dat::stf_type[isrc] = 2;
                    }
                    else if(strcmp(stf_type_str,"heaviside_bp") == 0){
                        dat::stf_type[isrc] = 3;
                    }
                    else{
                        dat::stf_type[isrc] = -1;
                    }
                }

                dat::src_x = mat::create(dat::nsrc);
                dat::src_z = mat::create(dat::nsrc);

                mat::copyHostToDevice(dat::src_x, src_x, dat::nsrc);
                mat::copyHostToDevice(dat::src_z, src_z, dat::nsrc);

                free(src_x);
                free(src_z);
            }

            {
                int single_rec = root["rec_x"].is<float>();
                dat::nrec = single_rec?1:root["rec_x"].size();

                float *rec_x = mat::createHost(dat::nrec);
                float *rec_z = mat::createHost(dat::nrec);

                for(int irec = 0; irec < dat::nrec; irec++){
                    rec_x[irec] = single_rec?root["rec_x"]:((JsonArray&)root["rec_x"]).get<float>(irec);
                    rec_z[irec] = single_rec?root["rec_z"]:((JsonArray&)root["rec_z"]).get<float>(irec);
                }

                dat::rec_x = mat::create(dat::nrec);
                dat::rec_z = mat::create(dat::nrec);

                mat::copyHostToDevice(dat::rec_x, rec_x, dat::nrec);
                mat::copyHostToDevice(dat::rec_z, rec_z, dat::nrec);

                free(rec_x);
                free(rec_z);
            }
        }
        jsonBuffer.clear();
    }
}
static void checkMemoryUsage(){
    size_t free_byte ;
    size_t total_byte ;
    cudaMemGetInfo( &free_byte, &total_byte ) ;
    float free_db = (float)free_byte ;
    float total_db = (float)total_byte ;
    float used_db = total_db - free_db ;

    printf("memory usage: %.1fMB / %.1fMB\n", used_db / 1024.0 / 1024.0, total_db / 1024.0 / 1024.0);
}
static void makeSourceTimeFunction(float *stf, int index){
    float max = 0;
    float alfa = 2 * dat::tauw_0[index] / dat::tauw[index];
    for(int it = 0; it < dat::nt; it++){
        float t = it * dat::dt;
        switch(dat::stf_type[index]){
            case 2:{
                stf[it] = (-2 * pow(alfa, 3) / pi) * (t - dat::tee_0[index]) * exp(-pow(alfa, 2) * pow(t - dat::tee_0[index], 2));
                break;
            }
            // other stf: later
        }

        if(fabs(stf[it]) > max){
            max = fabs(stf[it]);
        }
    }
    if(max > 0){
        for(int it = 0; it < dat::nt; it++){
            stf[it] /= max;
        }
    }
}
static void prepareSTF(){
    int &nt = dat::nt;
    float amp = dat::source_amplitude / dat::dx / dat::dz;
    float **stf_x = mat::createHost(dat::nsrc, dat::nt);
    float **stf_y = mat::createHost(dat::nsrc, dat::nt);
    float **stf_z = mat::createHost(dat::nsrc, dat::nt);
    float *stfn = mat::createHost(dat::nt);

    for(int isrc = 0; isrc < dat::nsrc; isrc++){
        makeSourceTimeFunction(stfn, isrc);
        float px = dat::stf_PSV_x[isrc];
        float pz = dat::stf_PSV_z[isrc];
        float norm = sqrt(pow(px,2) + pow(pz,2));
        for(int it = 0; it < nt; it++){
            stf_x[isrc][it] = amp * stfn[it] * px / norm;
            stf_y[isrc][it] = amp * stfn[it];
            stf_z[isrc][it] = amp * stfn[it] * pz / norm;
        }
    }

    mat::copyHostToDevice(dat::stf_x, stf_x, dat::nsrc, dat::nt);
    mat::copyHostToDevice(dat::stf_y, stf_y, dat::nsrc, dat::nt);
    mat::copyHostToDevice(dat::stf_z, stf_z, dat::nsrc, dat::nt);

    free(*stf_x);
    free(*stf_y);
    free(*stf_z);
    free(stf_x);
    free(stf_y);
    free(stf_z);
    free(stfn);
}
static void defineMaterialParameters(){
    // other model_type: later
    int &nx = dat::nx;
    int &nz = dat::nz;
    switch(dat::model_type){
        case 1:{
            mat::init(dat::rho, nx, nz, 3000);
            mat::init(dat::mu, nx, nz, 4.8e10);
            mat::init(dat::lambda, nx, nz, 4.8e10);
            break;
        }
        case 10:{
            mat::init(dat::rho, nx, nz, 2600);
            mat::init(dat::mu, nx, nz, 2.66e10);
            mat::init(dat::lambda, nx, nz, 3.42e10);
            break;
        }
        case 13:{
            mat::init(dat::mu, nx, nz, 2.66e10);
            mat::init(dat::lambda, nx, nz, 3.42e10);

            float rho = 2600;
            float mu = 2.66e10;
            float lambda = 3.42e10;
            float vp = sqrt((lambda + 2*mu) / rho);
            float vs = sqrt(mu / rho);
            int left = (int)((float)nx / 2 - (float)nx / 20 + 0.5);
            int right = (int)((float)nx / 2 + (float)nx / 20 + 0.5);
            int bottom = (int)((float)nz / 2 - (float)nz / 20 + 0.5);
            int top = (int)((float)nz / 2 + (float)nz / 20 + 0.5);

            float **rho2 = mat::createHost(nx, nz);
            mat::initHost(rho2, nx, nz, 2600);
            for(int i = left; i < right; i++){
                for(int j = bottom; j < top; j++){
                    rho2[i][j] = 3600;
                }
            }
            mat::copyHostToDevice(dat::rho, rho2, nx, nz);
            free(*rho2);
            free(rho2);
        }
    }
}
static void initialiseDynamicFields(){
    int &nx = dat::nx;
    int &nz = dat::nz;
    if(dat::wave_propagation_sh){
        mat::init(dat::vy, nx, nz, 0);
        mat::init(dat::uy, nx, nz, 0);
        mat::init(dat::sxy, nx, nz, 0);
        mat::init(dat::szy, nx, nz, 0);
    }
    if(dat::wave_propagation_psv){
        mat::init(dat::vx, nx, nz, 0);
        mat::init(dat::vz, nx, nz, 0);
        mat::init(dat::ux, nx, nz, 0);
        mat::init(dat::uz, nx, nz, 0);
        mat::init(dat::sxx, nx, nz, 0);
        mat::init(dat::szz, nx, nz, 0);
        mat::init(dat::sxz, nx, nz, 0);
    }
}
static void initialiseKernels(){
    int &nx = dat::nx;
    int &nz = dat::nz;
    mat::init(dat::K_lambda, nx, nz, 0);
    mat::init(dat::K_mu, nx, nz, 0);
    mat::init(dat::K_rho, nx, nz, 0);
}
static void runWaveFieldPropagation(){
    int &sh = dat::wave_propagation_sh;
    int &psv = dat::wave_propagation_psv;
    int &mode = dat::simulation_mode;

    int &nx = dat::nx;
    int &nz = dat::nz;
    float &dx = dat::dx;
    float &dz = dat::dz;
    float &dt = dat::dt;

    dim3 dimGrid(nx, nbt);
    dim3 dimBlock(nz / nbt);

    initialiseDynamicFields();

    for(int it = 0; it < dat::nt; it++){
        if(mode == 0){
            if((it + 1) % dat::sfe == 0){
                int isfe = dat::nsfe - (it + 1) / dat::sfe;
                if(sh){
                    mat::copyDeviceToHost(dat::uy_forward[isfe], dat::uy, nx, nz);
                }
                if(psv){
                    mat::copyDeviceToHost(dat::ux_forward[isfe], dat::ux, nx, nz);
                    mat::copyDeviceToHost(dat::uz_forward[isfe], dat::uz, nx, nz);
                }
            }
        }

        if(sh){
            divSY<<<dimGrid, dimBlock>>>(dat::dsy, dat::sxy, dat::szy, dx, dz, nx, nz);
        }
        if(psv){
            divSXZ<<<dimGrid, dimBlock>>>(dat::dsx, dat::dsz, dat::sxx, dat::szz, dat::sxz, dx, dz, nx, nz);
        }
        if(mode == 0){
            addSTF<<<dat::nsrc, 1>>>(
                dat::dsx, dat::dsy, dat::dsz, dat::stf_x, dat::stf_y, dat::stf_z,
                dat::src_x_id, dat::src_z_id, dat::isrc, sh, psv, it
            );
        }
        else if(mode == 1){
            addSTF<<<dat::nrec, 1>>>(
                dat::dsx, dat::dsy, dat::dsz, dat::adstf_x, dat::adstf_y, dat::adstf_z,
                dat::rec_x_id, dat::rec_z_id, -1, sh, psv, it
            );
        }
        if(sh){
            updateV<<<dimGrid, dimBlock>>>(dat::vy, dat::dsy, dat::rho, dat::absbound, dt);
            divVY<<<dimGrid, dimBlock>>>(dat::dvydx, dat::dvydz, dat::vy, dx, dz, nx, nz);
            updateSY<<<dimGrid, dimBlock>>>(dat::sxy, dat::szy, dat::dvydx, dat::dvydz, dat::mu, dt);
            updateU<<<dimGrid, dimBlock>>>(dat::uy, dat::vy, dt);
        }
        if(psv){
            updateV<<<dimGrid, dimBlock>>>(dat::vx, dat::dsx, dat::rho, dat::absbound, dt);
            updateV<<<dimGrid, dimBlock>>>(dat::vz, dat::dsz, dat::rho, dat::absbound, dt);
            divVXZ<<<dimGrid, dimBlock>>>(dat::dvxdx, dat::dvxdz, dat::dvzdx, dat::dvzdz, dat::vx, dat::vz, dx, dz, nx, nz);
            updateSXZ<<<dimGrid, dimBlock>>>(dat::sxx, dat::szz, dat::sxz, dat::dvxdx, dat::dvxdz, dat::dvzdx, dat::dvzdz, dat::lambda, dat::mu, dt);
            updateU<<<dimGrid, dimBlock>>>(dat::ux, dat::vx, dt);
            updateU<<<dimGrid, dimBlock>>>(dat::uz, dat::vz, dt);
        }
        if(mode == 0){
            if(dat::obs_type == 0){
                saveV<<<dat::nrec, 1>>>(
                    dat::v_rec_x, dat::v_rec_y, dat::v_rec_z, dat::vx, dat::vy, dat::vz,
                    dat::rec_x_id, dat::rec_z_id, sh, psv, it
                );
            }
            else if(dat::obs_type == 1){
                saveV<<<dat::nrec, 1>>>(
                    dat::v_rec_x, dat::v_rec_y, dat::v_rec_z, dat::ux, dat::uy, dat::uz,
                    dat::rec_x_id, dat::rec_z_id, sh, psv, it
                );
            }
            else if(dat::obs_type == 2 && dat::isrc >= 0){
                saveV<<<dat::nrec, 1>>>(
                    dat::u_obs_x, dat::u_obs_y, dat::u_obs_z, dat::ux, dat::uy, dat::uz,
                    dat::rec_x_id, dat::rec_z_id, dat::isrc, sh, psv, it
                );
            }
            if((it + 1) % dat::sfe == 0){
                int isfe = dat::nsfe - (it + 1) / dat::sfe;
                if(sh){
                    mat::copyDeviceToHost(dat::vy_forward[isfe], dat::vy, nx, nz);
                }
                if(psv){
                    mat::copyDeviceToHost(dat::vx_forward[isfe], dat::vx, nx, nz);
                    mat::copyDeviceToHost(dat::vz_forward[isfe], dat::vz, nx, nz);
                }
            }
        }
        else if(mode == 1){
            if((it + dat::sfe) % dat::sfe == 0){
                // dsi -> ui_fw -> vi_fw
                int isfe = (it + dat::sfe) / dat::sfe - 1;
                float tsfe = dat::sfe * dt;
                if(sh){
                    mat::copyHostToDevice(dat::dsy, dat::uy_forward[isfe], nx, nz);
                    divVY<<<dimGrid, dimBlock>>>(dat::dvydx, dat::dvydz, dat::uy, dx, dz, nx, nz);
                    divVY<<<dimGrid, dimBlock>>>(dat::dvydx_fw, dat::dvydz_fw, dat::dsy, dx, dz, nx, nz);
                    mat::copyHostToDevice(dat::dsy, dat::vy_forward[isfe], nx, nz);
                    interactionRhoY<<<dimGrid, dimBlock>>>(dat::K_rho, dat::vy, dat::dsy, tsfe);
                    interactionMuY<<<dimGrid, dimBlock>>>(dat::K_mu, dat::dvydx, dat::dvydx_fw, dat::dvydz, dat::dvydz_fw, tsfe);
                }
                if(psv){
                    mat::copyHostToDevice(dat::dsx, dat::ux_forward[isfe], nx, nz);
                    mat::copyHostToDevice(dat::dsz, dat::uz_forward[isfe], nx, nz);
                    divVXZ<<<dimGrid, dimBlock>>>(
                        dat::dvxdx, dat::dvxdz, dat::dvzdx, dat::dvzdz,
                        dat::ux, dat::uz, dx, dz, nx, nz
                    );
                    divVXZ<<<dimGrid, dimBlock>>>(
                        dat::dvxdx_fw, dat::dvxdz_fw, dat::dvzdx_fw, dat::dvzdz_fw,
                        dat::dsx, dat::dsz, dx, dz, nx, nz
                    );

                    mat::copyHostToDevice(dat::dsx, dat::vx_forward[isfe], nx, nz);
                    mat::copyHostToDevice(dat::dsz, dat::vz_forward[isfe], nx, nz);
                    interactionRhoXZ<<<dimGrid, dimBlock>>>(dat::K_rho, dat::vx, dat::dsx, dat::vz, dat::dsz, tsfe);
                    interactionMuXZ<<<dimGrid, dimBlock>>>(
                        dat::K_mu, dat::dvxdx, dat::dvxdx_fw, dat::dvxdz, dat::dvxdz_fw,
                        dat::dvzdx, dat::dvzdx_fw, dat::dvzdz, dat::dvzdz_fw, tsfe
                    );
                    interactionLambdaXZ<<<dimGrid, dimBlock>>>(dat::K_lambda, dat::dvxdx, dat::dvxdx_fw, dat::dvzdz, dat::dvzdz_fw, tsfe);
                }
            }
        }
    }
}
static void checkArgs(int adjoint){
    int &sh = dat::wave_propagation_sh;
    int &psv = dat::wave_propagation_psv;

    int &nx = dat::nx;
    int &nz = dat::nz;

    if(dat::nt % dat::sfe != 0){
        dat::nt = dat::sfe * (int)((float)dat::nt / dat::sfe + 0.5);
    }
    dat::nsfe = dat::nt / dat::sfe;
    dat::dx = dat::Lx / (nx - 1);
    dat::dz = dat::Lz / (nz - 1);

    if(sh){
        dat::vy = mat::create(nx, nz);
        dat::uy = mat::create(nx, nz);
        dat::sxy = mat::create(nx, nz);
        dat::szy = mat::create(nx, nz);
        dat::dsy = mat::create(nx, nz);
        dat::dvydx = mat::create(nx, nz);
        dat::dvydz = mat::create(nx, nz);

        dat::v_rec_y = mat::create(dat::nrec, dat::nt);
        dat::uy_forward = mat::createHost(dat::nsfe, nx, nz);
        dat::vy_forward = mat::createHost(dat::nsfe, nx, nz);
    }
    if(psv){
        dat::vx = mat::create(nx, nz);
        dat::vz = mat::create(nx, nz);
        dat::ux = mat::create(nx, nz);
        dat::uz = mat::create(nx, nz);
        dat::sxx = mat::create(nx, nz);
        dat::szz = mat::create(nx, nz);
        dat::sxz = mat::create(nx, nz);
        dat::dsx = mat::create(nx, nz);
        dat::dsz = mat::create(nx, nz);
        dat::dvxdx = mat::create(nx, nz);
        dat::dvxdz = mat::create(nx, nz);
        dat::dvzdx = mat::create(nx, nz);
        dat::dvzdz = mat::create(nx, nz);

        dat::v_rec_x = mat::create(dat::nrec, dat::nt);
        dat::v_rec_z = mat::create(dat::nrec, dat::nt);
        dat::ux_forward = mat::createHost(dat::nsfe, nx, nz);
        dat::uz_forward = mat::createHost(dat::nsfe, nx, nz);
        dat::vx_forward = mat::createHost(dat::nsfe, nx, nz);
        dat::vz_forward = mat::createHost(dat::nsfe, nx, nz);
    }

    dat::absbound = mat::create(nx, nz);
    dat::lambda = mat::create(nx, nz);
    dat::rho = mat::create(nx, nz);
    dat::mu = mat::create(nx, nz);

    dat::stf_x = mat::create(dat::nsrc, dat::nt);
    dat::stf_y = mat::create(dat::nsrc, dat::nt);
    dat::stf_z = mat::create(dat::nsrc, dat::nt);

    if(adjoint){
        if(sh){
            dat::dvydx_fw = mat::create(nx, nz);
            dat::dvydz_fw = mat::create(nx, nz);
        }
        if(psv){
            dat::dvxdx_fw = mat::create(nx, nz);
            dat::dvxdz_fw = mat::create(nx, nz);
            dat::dvzdx_fw = mat::create(nx, nz);
            dat::dvzdz_fw = mat::create(nx, nz);
        }

        dat::K_lambda = mat::create(nx, nz);
        dat::K_mu = mat::create(nx, nz);
        dat::K_rho = mat::create(nx, nz);

        dat::adstf_x = mat::create(dat::nrec, dat::nt);
        dat::adstf_y = mat::create(dat::nrec, dat::nt);
        dat::adstf_z = mat::create(dat::nrec, dat::nt);
    }

    dat::src_x_id = mat::createInt(dat::nsrc);
    dat::src_z_id = mat::createInt(dat::nsrc);
    dat::rec_x_id = mat::createInt(dat::nrec);
    dat::rec_z_id = mat::createInt(dat::nrec);

    computeIndices<<<dat::nsrc, 1>>>(dat::src_x_id, dat::src_x, dat::Lx, dat::nx);
    computeIndices<<<dat::nsrc, 1>>>(dat::src_z_id, dat::src_z, dat::Lz, dat::nz);
    computeIndices<<<dat::nrec, 1>>>(dat::rec_x_id, dat::rec_x, dat::Lx, dat::nx);
    computeIndices<<<dat::nrec, 1>>>(dat::rec_z_id, dat::rec_z, dat::Lz, dat::nz);

    dim3 dimGrid(nx, nbt);
    dim3 dimBlock(nz / nbt);
    initialiseAbsorbingBoundaries<<<dimGrid, dimBlock>>>(
        dat::absbound, dat::absorb_width,
        dat::absorb_left, dat::absorb_right, dat::absorb_bottom, dat::absorb_top,
        dat::Lx, dat::Lz, dat::dx, dat::dz
    );

    float *t = mat::createHost(dat::nt);
    for(int it = 0; it < dat::nt; it++){
        t[it] = it * dat::dt;
    }
    mat::write(t, dat::nt, "t");
}
static void runForward(int isrc){
    dat::simulation_mode = 0;
    dat::isrc = isrc;
    runWaveFieldPropagation();

    // float **v_rec_x=mat::createHost(dat::nrec, dat::nt);
    // float **v_rec_z=mat::createHost(dat::nrec, dat::nt);
    // mat::copyDeviceToHost(v_rec_x, dat::v_rec_x, dat::nrec, dat::nt);
    // mat::copyDeviceToHost(v_rec_z, dat::v_rec_z, dat::nrec, dat::nt);
    // mat::write(v_rec_x, dat::nrec, dat::nt, "vx_rec");
    // mat::write(v_rec_z, dat::nrec, dat::nt, "vz_rec");
    // mat::write(dat::vx_forward, dat::nsfe, dat::nx, dat::nz, "vx");
    // mat::write(dat::vz_forward, dat::nsfe, dat::nx, dat::nz, "vz");
}
static void runAdjoint(int init_kernel){
    dat::simulation_mode = 1;
    if(init_kernel){
        initialiseKernels();
    }
    runWaveFieldPropagation();

    // float **rho = mat::createHost(dat::nx, dat::nz);
    // float **mu = mat::createHost(dat::nx, dat::nz);
    // float **lambda = mat::createHost(dat::nx, dat::nz);
    // mat::copyDeviceToHost(rho, dat::K_rho, dat::nx, dat::nz);
    // mat::copyDeviceToHost(mu, dat::K_mu, dat::nx, dat::nz);
    // mat::copyDeviceToHost(lambda, dat::K_lambda, dat::nx, dat::nz);
    // mat::write(rho, dat::nx, dat::nz, "rho");
    // mat::write(mu, dat::nx, dat::nz, "mu");
    // mat::write(lambda, dat::nx, dat::nz, "lambda");
    // mat::write(dat::vx_forward, dat::nsfe, dat::nx, dat::nz, "vx");
    // mat::write(dat::vz_forward, dat::nsfe, dat::nx, dat::nz, "vz");
}
static float computeKernels(int kernel){
    int &nsrc = dat::nsrc;
    int &nrec = dat::nrec;
    int &nx = dat::nx;
    int &nz = dat::nz;
    int &nt = dat::nt;
    float &dt = dat::dt;

    float *d_misfit = mat::create(nt);
    float *h_misfit = mat::createHost(nt);
    mat::init(d_misfit, nt, 0);

    initialiseKernels();
    for(int isrc = 0; isrc < nsrc; isrc++){
        runForward(isrc);
        for(int irec = 0; irec < nrec; irec++){
            calculateMisfit<<<nt, 1>>>(d_misfit, dat::v_rec_x, dat::u_obs_x, dat::tw, dt, isrc, irec);
            calculateMisfit<<<nt, 1>>>(d_misfit, dat::v_rec_z, dat::u_obs_z, dat::tw, dt, isrc, irec);
        }
        if(kernel){
            prepareAdjointSTF<<<nt, nrec>>>(dat::adstf_x, dat::v_rec_x, dat::u_obs_x, dat::tw, nt, isrc);
            prepareAdjointSTF<<<nt, nrec>>>(dat::adstf_z, dat::v_rec_z, dat::u_obs_z, dat::tw, nt, isrc);
            mat::init(dat::adstf_y, nrec, nt, 0);
            runAdjoint(0);
        }
    }

    mat::copyDeviceToHost(h_misfit, d_misfit, nt);

    float misfit = 0;
    for(int i = 0; i< nt; i++){
        misfit += h_misfit[i];
    }
    free(h_misfit);
    cudaFree(d_misfit);

    if(kernel){
        if(dat::misfit_init < 0){
            dat::misfit_init = misfit;
        }
        dim3 dimGrid(nx, nbt);
        dim3 dimBlock(nz / nbt);
        normKernel<<<dimGrid, dimBlock>>>(dat::K_rho, dat::rho_start, dat::misfit_init);
        normKernel<<<dimGrid, dimBlock>>>(dat::K_mu, dat::mu_start, dat::misfit_init);
        normKernel<<<dimGrid, dimBlock>>>(dat::K_lambda, dat::lambda_start, dat::misfit_init);
        filterKernelX<<<dimGrid, dimBlock>>>(dat::K_rho, dat::gtemp, nx, dat::sigma);
        filterKernelZ<<<dimGrid, dimBlock>>>(dat::K_rho, dat::gtemp, dat::gsum, nz, dat::sigma);
        filterKernelX<<<dimGrid, dimBlock>>>(dat::K_mu, dat::gtemp, nx, dat::sigma);
        filterKernelZ<<<dimGrid, dimBlock>>>(dat::K_mu, dat::gtemp, dat::gsum, nz, dat::sigma);
        filterKernelX<<<dimGrid, dimBlock>>>(dat::K_lambda, dat::gtemp, nx, dat::sigma);
        filterKernelZ<<<dimGrid, dimBlock>>>(dat::K_lambda, dat::gtemp, dat::gsum, nz, dat::sigma);
    }
    printf("misfit = %e\n", misfit); // later

    return misfit / dat::misfit_init;
}
static float computeKernels(){
    return computeKernels(1);
}
static void inversionRoutine(){
    int &nx = dat::nx;
    int &nz = dat::nz;
    int &nt = dat::nt;
    float &dt = dat::dt;

    // model start
    dat::rho_start = mat::create(nx, nz);
    dat::mu_start = mat::create(nx, nz);
    dat::lambda_start = mat::create(nx, nz);
    mat::copy(dat::rho_start, dat::rho, nx, nz);
    mat::copy(dat::mu_start, dat::mu, nx, nz);
    mat::copy(dat::lambda_start, dat::lambda, nx, nz);

    // taper weights
    dat::tw = mat::create(nt);
    getTaperWeights<<<nt, 1>>>(dat::tw, dt, nt);

    // gaussian filter
    dat::sigma = 2;
    dat::gsum = mat::create(nx, nz);
    dat::gtemp = mat::create(nx, nz);
    dim3 dimGrid(nx, nbt);
    dim3 dimBlock(nz / nbt);
    initialiseGaussian<<<dimGrid, dimBlock>>>(dat::gsum, nx, nz, dat::sigma);

    // adjoint related parameters
    dat::obs_type = 1;
    dat::misfit_init = -1;

    float misfit = computeKernels();


    float **lambda = mat::createHost(nx,nz);
    float **mu = mat::createHost(nx,nz);
    float **rho = mat::createHost(nx,nz);
    mat::copyDeviceToHost(rho, dat::K_rho, dat::nx, dat::nz);
    mat::copyDeviceToHost(mu, dat::K_mu, dat::nx, dat::nz);
    mat::copyDeviceToHost(lambda, dat::K_lambda, dat::nx, dat::nz);
    mat::write(rho, dat::nx, dat::nz, "rho");
    mat::write(mu, dat::nx, dat::nz, "mu");
    mat::write(lambda, dat::nx, dat::nz, "lambda");

    printf("misfit_normed = %f\n", misfit);
}
static void runSyntheticInvertion(){
    int &nsrc = dat::nsrc;
    int &nrec = dat::nrec;
    int &nt = dat::nt;

    checkArgs(1);
    dat::obs_type = 2; // save displacement persouce
    dat::model_type = 13; // true model: later
    prepareSTF(); // dat::use_given_stf, sObsPerFreq: later
    defineMaterialParameters(); // dat::use_given_model: later
    dat::u_obs_x = mat::create(nsrc, nrec, nt);
    dat::u_obs_z = mat::create(nsrc, nrec, nt);
    for(int isrc = 0; isrc < nsrc; isrc++){
        runForward(isrc);
    }

    dat::model_type = 10;
    defineMaterialParameters();
    inversionRoutine();
}

int main(int argc , char *argv[]){
    importData();
    if(argc == 1){
        runSyntheticInvertion();
    }
    else{
        for(int i = 1; i< argc; i++){
            if(strcmp(argv[i],"run_forward") == 0){
                checkArgs(0);
                prepareSTF();
                defineMaterialParameters();
                runForward(-1);
            }
        }
    }
    checkMemoryUsage();

    return 0;
}
