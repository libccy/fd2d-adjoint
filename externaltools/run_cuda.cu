#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "ArduinoJson.h"

#define devij int i = blockIdx.x, j = threadIdx.x + blockIdx.y * blockDim.x

const float pi = 3.1415927;
const int nbt = 2;

typedef struct{
    int nx;
    int nz;
    int nt;
    float dx;
    float dz;
    float dt;
    float Lx;
    float Lz;

    int sfe;
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

    int nsrc;
    int nrec;

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

    float **v_rec_x;
    float **v_rec_y;
    float **v_rec_z;

    float ***ux_forward;  // host
    float ***uy_forward;  // host
    float ***uz_forward;  // host
    float ***vx_forward;  // host
    float ***vy_forward;  // host
    float ***vz_forward;  // host
} fdat;

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

    void copyHostToDevice(float *d_a, const float *a, const int m){
        cudaMemcpy(d_a, a , m * sizeof(float), cudaMemcpyHostToDevice);
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

    void read(float *data, int len, char *fname){
        char buffer[50] = "externaltools/";
        strcat(buffer, fname);
        FILE *file = fopen(buffer, "rb");
        fwrite(data, sizeof(float), len, file);
        fclose(file);
    }
    float *read(int len, char *fname){
        float *data = (float *)malloc(len * sizeof(float));
        mat::read(data, len, fname);
        return data;
    }
    void write(float *data, int len, char *fname){
        char buffer[50] = "externaltools/";
        strcat(buffer, fname);
        FILE *file = fopen(buffer, "wb");
        fwrite(data, sizeof(float), len, file);
        fclose(file);
    }
}

void printMat(float **a, int m, int n){
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            printf("%f ", a[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}
void copyMat(float **a, float **b, int nx, int nz){
    // replace with copyDeviceToHost: later
    for(int i = 0; i < nx; i++){
        for(int j = 0; j < nz; j++){
            a[i][j] = b[i][j];
        }
    }
}

__global__ void divSY(float **out, float **sxy, float **szy, float dx, float dz, int nx, int nz){
    devij;
    if(i >= 2 && i < nx - 2){
        out[i][j] = 9*(sxy[i][j]-sxy[i-1][j])/(8*dx)-(sxy[i+1][j]-sxy[i-2][j])/(24*dx);
    }
    else{
        out[i][j] = 0;
    }
    if(j >= 2 && j < nz - 2){
        out[i][j] += 9*(szy[i][j]-szy[i][j-1])/(8*dz)-(szy[i][j+1]-szy[i][j-2])/(24*dz);
    }
}
__global__ void divSXZ(float **outx, float **outz, float **sxx, float **szz, float **sxz, float dx, float dz, int nx, int nz){
    devij;
    if(i >= 2 && i < nx - 2){
        outx[i][j] = 9*(sxx[i][j]-sxx[i-1][j])/(8*dx)-(sxx[i+1][j]-sxx[i-2][j])/(24*dx);
        outz[i][j] = 9*(sxz[i][j]-sxz[i-1][j])/(8*dx)-(sxz[i+1][j]-sxz[i-2][j])/(24*dx);
    }
    else{
        outx[i][j] = 0;
        outz[i][j] = 0;
    }
    if(j >= 2 && j < nz - 2){
        outx[i][j] += 9*(sxz[i][j]-sxz[i][j-1])/(8*dz)-(sxz[i][j+1]-sxz[i][j-2])/(24*dz);
        outz[i][j] += 9*(szz[i][j]-szz[i][j-1])/(8*dz)-(szz[i][j+1]-szz[i][j-2])/(24*dz);
    }
}
__global__ void divVY(float **outx, float **outz, float **vy, float dx, float dz, int nx, int nz){
    devij;
    if(i >= 1 && i < nx - 2){
        outx[i][j] = 9*(vy[i+1][j]-vy[i][j])/(8*dx)-(vy[i+2][j]-vy[i-1][j])/(24*dx);
    }
    else{
        outx[i][j] = 0;
    }
    if(j >= 1 && j < nz - 2){
        outz[i][j] = 9*(vy[i][j+1]-vy[i][j])/(8*dz)-(vy[i][j+2]-vy[i][j-1])/(24*dz);
    }
    else{
        outz[i][j] = 0;
    }
}
__global__ void divVXZ(float **outxx, float **outxz, float **outzx, float **outzz, float **vx, float **vz, float dx, float dz, int nx, int nz){
    devij;
    if(i >= 1 && i < nx - 2){
        outxx[i][j] = 9*(vx[i+1][j]-vx[i][j])/(8*dx)-(vx[i+2][j]-vx[i-1][j])/(24*dx);
        outzx[i][j] = 9*(vz[i+1][j]-vz[i][j])/(8*dx)-(vz[i+2][j]-vz[i-1][j])/(24*dx);
    }
    else{
        outxx[i][j] = 0;
        outzx[i][j] = 0;
    }
    if(j >= 1 && j < nz - 2){
        outxz[i][j] = 9*(vx[i][j+1]-vx[i][j])/(8*dz)-(vx[i][j+2]-vx[i][j-1])/(24*dz);
        outzz[i][j] = 9*(vz[i][j+1]-vz[i][j])/(8*dz)-(vz[i][j+2]-vz[i][j-1])/(24*dz);
    }
    else{
        outxz[i][j] = 0;
        outzz[i][j] = 0;
    }
}

__global__ void addSTF(float **dsx, float **dsy, float **dsz, float **stf_x, float **stf_y, float **stf_z,
    int *src_x_id, int *src_z_id, int sh, int psv, int n){
    int is = blockIdx.x;
    int xs = src_x_id[is];
    int zs = src_z_id[is];
    if(sh){
        dsy[xs][zs] += stf_y[is][n];
    }
    if(psv){
        dsx[xs][zs] += stf_x[is][n];
        dsz[xs][zs] += stf_z[is][n];
    }
}
__global__ void saveV(float **v_rec_x, float **v_rec_y, float **v_rec_z, float **vx, float **vy, float **vz,
    int *rec_x_id, int *rec_z_id, int sh, int psv, int n){
    int ir = blockIdx.x;
    int xr = rec_x_id[ir];
    int zr = rec_z_id[ir];
    if(sh){
        v_rec_y[ir][n] = vy[xr][zr];
    }
    if(psv){
        v_rec_x[ir][n] = vx[xr][zr];
        v_rec_z[ir][n] = vz[xr][zr];
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

fdat *importData(void){
    fdat *dat = new fdat;
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
            dat->nx = root["nx"];
            dat->nz = root["nz"];
            dat->nt = root["nt"];
            dat->dt = root["dt"];
            dat->Lx = root["Lx"];
            dat->Lz = root["Lz"];
            dat->sfe = root["sfe"];
            if(dat->nt % dat->sfe != 0){
                dat->nt = dat->sfe * (int)((float)dat->nt / dat->sfe + 0.5);
            }

            dat->model_type = root["model_type"];
            dat->use_given_model = root["use_given_model"];
            dat->use_given_stf = root["use_given_stf"];
            dat->source_amplitude = root["source_amplitude"];
            dat->order = root["order"]; // order = 2: later

            dat->absorb_left = root["absorb_left"];
            dat->absorb_right = root["absorb_right"];
            dat->absorb_top = root["absorb_top"];
            dat->absorb_bottom = root["absorb_bottom"];
            dat->absorb_width = root["width"];

            {
                const char* wave_propagation_type = root["wave_propagation_type"].as<char*>();
                if(strcmp(wave_propagation_type,"SH") == 0){
                    dat->wave_propagation_sh = 1;
                    dat->wave_propagation_psv = 0;
                }
                else if(strcmp(wave_propagation_type,"PSV") == 0){
                    dat->wave_propagation_sh = 0;
                    dat->wave_propagation_psv = 1;
                }
                else if(strcmp(wave_propagation_type,"both") == 0){
                    dat->wave_propagation_sh = 1;
                    dat->wave_propagation_psv = 1;
                }
                else{
                    dat->wave_propagation_sh = 0;
                    dat->wave_propagation_psv = 0;
                }
            }

            {
                const char* simulation_mode = root["simulation_mode"].as<char*>();
                if(strcmp(simulation_mode,"forward") == 0){
                    dat->simulation_mode = 0;
                }
                else if(strcmp(simulation_mode,"adjoint") == 0){
                    dat->simulation_mode = 1;
                }
                else{
                    dat->simulation_mode = -1;
                }
            }

            {
                int single_src = root["src_info"].is<JsonObject>();
                dat->nsrc = single_src?1:root["src_info"].size();

                float *src_x = mat::createHost(dat->nsrc);
                float *src_z = mat::createHost(dat->nsrc);

                dat->stf_type = mat::createIntHost(dat->nsrc);
                dat->stf_PSV_x = mat::createHost(dat->nsrc);
                dat->stf_PSV_z = mat::createHost(dat->nsrc);
                dat->tauw_0 = mat::createHost(dat->nsrc);
                dat->tauw = mat::createHost(dat->nsrc);
                dat->tee_0 = mat::createHost(dat->nsrc);
                dat->f_min = mat::createHost(dat->nsrc);
                dat->f_max = mat::createHost(dat->nsrc);

                for(int i = 0; i < dat->nsrc; i++){
                    JsonObject& src = single_src?root["src_info"]:((JsonArray&)root["src_info"]).get<JsonObject>(i);
                    src_x[i] = src["loc_x"];
                    src_z[i] = src["loc_z"];

                    dat->stf_PSV_x[i] = src["stf_PSV"][0];
                    dat->stf_PSV_z[i] = src["stf_PSV"][1];
                    dat->tauw_0[i] = src["tauw_0"];
                    dat->tauw[i] = src["tauw"];
                    dat->tee_0[i] = src["tee_0"];
                    dat->f_min[i] = src["f_min"];
                    dat->f_max[i] = src["f_max"];

                    const char* stf_type_str = src["stf_type"].as<char*>();
                    if(strcmp(stf_type_str,"delta") == 0){
                        dat->stf_type[i] = 0;
                    }
                    else if(strcmp(stf_type_str,"delta_bp") == 0){
                        dat->stf_type[i] = 1;
                    }
                    else if(strcmp(stf_type_str,"ricker") == 0){
                        dat->stf_type[i] = 2;
                    }
                    else if(strcmp(stf_type_str,"heaviside_bp") == 0){
                        dat->stf_type[i] = 3;
                    }
                    else{
                        dat->stf_type[i] = -1;
                    }
                }

                dat->src_x = mat::create(dat->nsrc);
                dat->src_z = mat::create(dat->nsrc);

                mat::copyHostToDevice(dat->src_x, src_x, dat->nsrc);
                mat::copyHostToDevice(dat->src_z, src_z, dat->nsrc);

                free(src_x);
                free(src_z);
            }

            {
                int single_rec = root["rec_x"].is<float>();
                dat->nrec = single_rec?1:root["rec_x"].size();

                float *rec_x = mat::createHost(dat->nrec);
                float *rec_z = mat::createHost(dat->nrec);

                for(int i = 0; i < dat->nrec; i++){
                    rec_x[i] = single_rec?root["rec_x"]:((JsonArray&)root["rec_x"]).get<float>(i);
                    rec_z[i] = single_rec?root["rec_z"]:((JsonArray&)root["rec_z"]).get<float>(i);
                }

                dat->rec_x = mat::create(dat->nrec);
                dat->rec_z = mat::create(dat->nrec);

                mat::copyHostToDevice(dat->rec_x, rec_x, dat->nrec);
                mat::copyHostToDevice(dat->rec_z, rec_z, dat->nrec);

                free(rec_x);
                free(rec_z);
            }
        }
        jsonBuffer.clear();
    }
    return dat;
}
void makeSourceTimeFunction(fdat *dat, float *stf, int index){
    float max = 0;
    float alfa = 2 * dat->tauw_0[index] / dat->tauw[index];
    for(int i = 0; i < dat->nt; i++){
        float t = i * dat->dt;
        switch(dat -> stf_type[index]){
            case 2:{
                stf[i] = (-2 * pow(alfa, 3) / pi) * (t - dat->tee_0[index]) * exp(-pow(alfa, 2) * pow(t - dat->tee_0[index], 2));
                break;
            }
            // other stf: later
        }

        if(fabs(stf[i]) > max){
            max = fabs(stf[i]);
        }
    }
    if(max > 0){
        for(int i = 0; i < dat->nt; i++){
            stf[i] /= max;
        }
    }
}
void prepareSTF(fdat *dat){
    int nt = dat->nt;
    float amp = dat->source_amplitude / dat->dx / dat->dz;
    float **stf_x = mat::createHost(dat->nsrc, dat->nt);
    float **stf_y = mat::createHost(dat->nsrc, dat->nt);
    float **stf_z = mat::createHost(dat->nsrc, dat->nt);
    float *stfn = mat::createHost(dat->nt);

    for(int i=0; i < dat->nsrc; i++){
        makeSourceTimeFunction(dat, stfn, i);
        float px = dat->stf_PSV_x[i];
        float pz = dat->stf_PSV_z[i];
        float norm = sqrt(pow(px,2) + pow(pz,2));
        for(int j = 0; j < nt; j++){
            stf_x[i][j] = amp * stfn[j] * px / norm;
            stf_y[i][j] = amp * stfn[j];
            stf_z[i][j] = amp * stfn[j] * pz / norm;
        }
    }

    mat::copyHostToDevice(dat->stf_x, stf_x, dat->nsrc, dat->nt);
    mat::copyHostToDevice(dat->stf_y, stf_y, dat->nsrc, dat->nt);
    mat::copyHostToDevice(dat->stf_z, stf_z, dat->nsrc, dat->nt);

    free(*stf_x);
    free(*stf_y);
    free(*stf_z);
    free(stf_x);
    free(stf_y);
    free(stf_z);
    free(stfn);
}
void defineMaterialParameters(fdat *dat){
    // other model_type: later
    int nx = dat->nx;
    int nz = dat->nz;
    switch(dat->model_type){
        case 1:{
            mat::init(dat->rho, nx, nz, 3000);
            mat::init(dat->mu, nx, nz, 4.8e10);
            mat::init(dat->lambda, nx, nz, 4.8e10);
            break;
        }
        case 10:{
            mat::init(dat->rho, nx, nz, 2600);
            mat::init(dat->mu, nx, nz, 2.66e10);
            mat::init(dat->lambda, nx, nz, 3.42e10);
            break;
        }
    }
}
void initialiseDynamicFields(fdat *dat){
    int nx = dat->nx;
    int nz = dat->nz;
    if(dat->wave_propagation_sh){
        mat::init(dat->vy, nx, nz, 0);
        mat::init(dat->uy, nx, nz, 0);
        mat::init(dat->sxy, nx, nz, 0);
        mat::init(dat->szy, nx, nz, 0);
    }
    if(dat->wave_propagation_psv){
        mat::init(dat->vx, nx, nz, 0);
        mat::init(dat->vz, nx, nz, 0);
        mat::init(dat->ux, nx, nz, 0);
        mat::init(dat->uz, nx, nz, 0);
        mat::init(dat->sxx, nx, nz, 0);
        mat::init(dat->szz, nx, nz, 0);
        mat::init(dat->sxz, nx, nz, 0);
    }
    // initialise kernels: later
}
void runWaveFieldPropagation(fdat *dat){
    int sh = dat->wave_propagation_sh;
    int psv = dat->wave_propagation_psv;
    int mode = dat->simulation_mode;

    int nx = dat->nx;
    int nz = dat->nz;
    int nt = dat->nt;
    float dx = dat->dx;
    float dz = dat->dz;
    float dt = dat->dt;


    dim3 dimGrid(nx, nbt);
    dim3 dimBlock(nz / nbt);
    initialiseDynamicFields(dat);

    for(int n = 0; n < dat->nt; n++){
        if((n + 1) % dat->sfe == 0){
            int isfe = (int)(nt / dat->sfe) - (n + 1) / dat->sfe;
            if(sh){
                mat::copyDeviceToHost(dat->uy_forward[isfe], dat->uy, nx, nz);
            }
            if(psv){
                mat::copyDeviceToHost(dat->ux_forward[isfe], dat->ux, nx, nz);
                mat::copyDeviceToHost(dat->uz_forward[isfe], dat->uz, nx, nz);
            }
        }
        if(sh){
            divSY<<<dimGrid, dimBlock>>>(dat->dsy, dat->sxy, dat->szy, dx, dz, nx, nz);
        }
        if(psv){
            divSXZ<<<dimGrid, dimBlock>>>(dat->dsx, dat->dsz, dat->sxx, dat->szz, dat->sxz, dx, dz, nx, nz);
        }
        if(mode == 0 || mode == 1){
            addSTF<<<dat->nsrc, 1>>>(
                dat->dsx, dat->dsy, dat->dsz, dat->stf_x, dat->stf_y, dat->stf_z,
                dat->src_x_id, dat->src_z_id, dat->wave_propagation_sh, dat->wave_propagation_psv, n
            );
        }
        if(sh){
            updateV<<<dimGrid, dimBlock>>>(dat->vy, dat->dsy, dat->rho, dat->absbound, dt);
            divVY<<<dimGrid, dimBlock>>>(dat->dvydx, dat->dvydz, dat->vy, dx, dz, nx, nz);
            updateSY<<<dimGrid, dimBlock>>>(dat->sxy, dat->szy, dat->dvydx, dat->dvydz, dat->mu, dt);
            updateU<<<dimGrid, dimBlock>>>(dat->uy, dat->vy, dt);
        }
        if(psv){
            updateV<<<dimGrid, dimBlock>>>(dat->vx, dat->dsx, dat->rho, dat->absbound, dt);
            updateV<<<dimGrid, dimBlock>>>(dat->vz, dat->dsz, dat->rho, dat->absbound, dt);
            divVXZ<<<dimGrid, dimBlock>>>(dat->dvxdx, dat->dvxdz, dat->dvzdx, dat->dvzdz, dat->vx, dat->vz, dx, dz, nx, nz);
            updateSXZ<<<dimGrid, dimBlock>>>(dat->sxx, dat->szz, dat->sxz, dat->dvxdx, dat->dvxdz, dat->dvzdx, dat->dvzdz, dat->lambda, dat->mu, dt);
            updateU<<<dimGrid, dimBlock>>>(dat->ux, dat->vx, dt);
            updateU<<<dimGrid, dimBlock>>>(dat->uz, dat->vz, dt);
        }
        if(mode == 0){
            saveV<<<dat->nrec, 1>>>(
                dat->v_rec_x, dat->v_rec_y, dat->v_rec_z, dat->vx, dat->vy, dat->vz,
                dat->rec_x_id, dat->rec_z_id, dat->wave_propagation_sh, dat->wave_propagation_psv, n
            );
            if((n + 1) % dat->sfe == 0){
                int isfe = (int)(nt / dat->sfe) - (n + 1) / dat->sfe;
                if(sh){
                    mat::copyDeviceToHost(dat->vy_forward[isfe], dat->vy, nx, nz);
                }
                if(psv){
                    mat::copyDeviceToHost(dat->vx_forward[isfe], dat->vx, nx, nz);
                    mat::copyDeviceToHost(dat->vz_forward[isfe], dat->vz, nx, nz);
                }
            }
        }
        else if(mode == 1){
            // adjoint: later
        }
    }
}
void checkArgs(fdat *dat){
    int nx = dat->nx;
    int nz = dat->nz;
    int nsfe = (int)(dat->nt / dat->sfe);

    dat->dx = dat->Lx / (nx - 1);
    dat->dz = dat->Lz / (nz - 1);

    if(dat->wave_propagation_sh){
        dat->vy = mat::create(nx, nz);
        dat->uy = mat::create(nx, nz);
        dat->sxy = mat::create(nx, nz);
        dat->szy = mat::create(nx, nz);
        dat->dsy = mat::create(nx, nz);
        dat->dvydx = mat::create(nx, nz);
        dat->dvydz = mat::create(nx, nz);

        dat->v_rec_y = mat::create(dat->nrec, dat->nt);
        dat->uy_forward = mat::createHost(nsfe, nx, nz);
        dat->vy_forward = mat::createHost(nsfe, nx, nz);
    }
    if(dat->wave_propagation_psv){
        dat->vx = mat::create(nx, nz);
        dat->vz = mat::create(nx, nz);
        dat->ux = mat::create(nx, nz);
        dat->uz = mat::create(nx, nz);
        dat->sxx = mat::create(nx, nz);
        dat->szz = mat::create(nx, nz);
        dat->sxz = mat::create(nx, nz);
        dat->dsx = mat::create(nx, nz);
        dat->dsz = mat::create(nx, nz);
        dat->dvxdx = mat::create(nx, nz);
        dat->dvxdz = mat::create(nx, nz);
        dat->dvzdx = mat::create(nx, nz);
        dat->dvzdz = mat::create(nx, nz);

        dat->v_rec_x = mat::create(dat->nrec, dat->nt);
        dat->v_rec_z = mat::create(dat->nrec, dat->nt);
        dat->ux_forward = mat::createHost(nsfe, nx, nz);
        dat->uz_forward = mat::createHost(nsfe, nx, nz);
        dat->vx_forward = mat::createHost(nsfe, nx, nz);
        dat->vz_forward = mat::createHost(nsfe, nx, nz);
    }

    dat->absbound = mat::create(nx, nz);
    dat->lambda = mat::create(nx, nz);
    dat->rho = mat::create(nx, nz);
    dat->mu = mat::create(nx, nz);

    dat->stf_x = mat::create(dat->nsrc, dat->nt);
    dat->stf_y = mat::create(dat->nsrc, dat->nt);
    dat->stf_z = mat::create(dat->nsrc, dat->nt);

    dat->src_x_id = mat::createInt(dat->nsrc);
    dat->src_z_id = mat::createInt(dat->nsrc);
    dat->rec_x_id = mat::createInt(dat->nrec);
    dat->rec_z_id = mat::createInt(dat->nrec);

    if(dat->use_given_model){
        // GivenModel: later
    }
    else{
        defineMaterialParameters(dat);
    }
    if(dat->use_given_stf){
        // GivenSTF: later
    }
    else{
        prepareSTF(dat);
    }

    computeIndices<<<dat->nsrc, 1>>>(dat->src_x_id, dat->src_x, dat->Lx, dat->nx);
    computeIndices<<<dat->nsrc, 1>>>(dat->src_z_id, dat->src_z, dat->Lz, dat->nz);
    computeIndices<<<dat->nrec, 1>>>(dat->rec_x_id, dat->rec_x, dat->Lx, dat->nx);
    computeIndices<<<dat->nrec, 1>>>(dat->rec_z_id, dat->rec_z, dat->Lz, dat->nz);

    dim3 dimGrid(nx, nbt);
    dim3 dimBlock(nz / nbt);
    initialiseAbsorbingBoundaries<<<dimGrid, dimBlock>>>(
        dat->absbound, dat->absorb_width,
        dat->absorb_left, dat->absorb_right, dat->absorb_bottom, dat->absorb_top,
        dat->Lx, dat->Lz, dat->dx, dat->dz
    );

    float *t = mat::createHost(dat->nt);
    for(int i = 0; i < dat->nt; i++){
        t[i] = i * dat->dt;
    }
    mat::write(t, dat->nt, "t");
}
void runForward(void){
    fdat *dat = importData();
    checkArgs(dat);
    runWaveFieldPropagation(dat);

    char oname[50];
    float **v_rec_x = mat::createHost(dat->nrec, dat->nt);
    float **v_rec_z = mat::createHost(dat->nrec, dat->nt);
    mat::copyDeviceToHost(v_rec_x, dat->v_rec_x, dat->nrec, dat->nt);
    mat::copyDeviceToHost(v_rec_z, dat->v_rec_z, dat->nrec, dat->nt);
    for(int i = 0; i < dat->nrec; i++){
        sprintf(oname, "vx%d", i);
        mat::write(v_rec_x[i], dat->nt, oname);
        sprintf(oname, "vz%d", i);
        mat::write(v_rec_z[i], dat->nt, oname);
    }
}

int main(int argc , char *argv[]){
    for(int i = 0; i< argc; i++){
        if(strcmp(argv[i],"runForward") == 0){
            runForward();
        }
    }

    return 0;
}
