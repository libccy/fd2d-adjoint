#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "ArduinoJson.h"

#define devij(dimx, dimy) \
int i = blockIdx.x % dimx; \
int j = threadIdx.x + (blockIdx.x - i) / dimx * dimy / d_nbt;

const float pi = 3.1415927;
const int nbt = 1;
__constant__ int d_nbt = 1;

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
    int *stf_type;
    float *src_x;
    float *src_z;
    float *stf_PSV_x;
    float *stf_PSV_z;
    float *tauw_0;
    float *tauw;
    float *tee_0;
    float *f_min;
    float *f_max;
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

    float ***ux_forward;
    float ***uy_forward;
    float ***uz_forward;
    float ***vx_forward;
    float ***vy_forward;
    float ***vz_forward;
} fdat;

namespace mat{
    __global__ void _setValue(float *mat, const float init, const int m){
        int i = threadIdx.x;
        mat[i] = init;
    }
    __global__ void _setValue(float **mat, const float init, const int m, const int n){
        devij(m, n);
        mat[i][j] = init;
    }
    __global__ void _setValue(float ***mat, const float init, const int m, const int n, const int p){
        devij(m, n);
        mat[p][i][j] = init;
    }
    __global__ void _setPointerValue(float **mat, float *data, const int n){
        int i = threadIdx.x;
        mat[i] = data + n * i;
    }
    __global__ void _setPointerValue(float ***mat, float **data, const int i){
        mat[i] = data;
    }


    float *init(float *mat, const int m, const float init){
        mat::_setValue<<<1, m>>>(mat, init, m);
        return mat;
    }
    float **init(float **mat, const int m, const int n, const float init){
        mat::_setValue<<<m * nbt, n / nbt>>>(mat, init, m, n);
        return mat;
    }
    float ***init(float ***mat, const int p, const int m, const int n, const float init){
        for(int i = 0; i < p; i++){
            mat::_setValue<<<m * nbt, n / nbt>>>(mat, init, m, n, i);
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
        mat::_setPointerValue<<<1, m>>>(mat, data, n);
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
    for(int i=2;i<m-2;i++){
        for(int j=2;j<n-2;j++){
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

void divSY(float **out, float **sxy, float **szy, float dx, float dz, int nx, int nz, int order){
    // order = 2: later
    for(int i = 0; i < nx; i++){
        for(int j = 0; j < nz; j++){
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
    }
}
void divSXZ(float **outx, float **outz, float **sxx, float **szz, float **sxz, float dx, float dz, int nx, int nz, int order){
    // order = 2: later
    for(int i = 0; i < nx; i++){
        for(int j = 0; j < nz; j++){
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
    }
}
void divVY(float **outx, float **outz, float **vy, float dx, float dz, int nx, int nz, int order){
    for(int i = 0; i < nx; i++){
        for(int j = 0; j < nz; j++){
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
    }
}
void divVXZ(float **outxx, float **outxz, float **outzx, float **outzz, float **vx, float **vz, float dx, float dz, int nx, int nz, int order){
    for(int i = 0; i < nx; i++){
        for(int j = 0; j < nz; j++){
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
    }
}
void updateV(float **v, float **ds, float **rho, float **absbound, float dt, int nx, int nz){
    for(int i = 0; i < nx; i++){
        for(int j = 0; j < nz; j++){
            v[i][j] = absbound[i][j] * (v[i][j] + dt * ds[i][j] / rho[i][j]);
        }
    }
}
void updateSY(float **sxy, float **szy, float **dvydx, float **dvydz, float **mu, float dt, int nx, int nz){
    for(int i = 0; i < nx; i++){
        for(int j = 0; j < nz; j++){
            sxy[i][j] += dt * mu[i][j] * dvydx[i][j];
            szy[i][j] += dt * mu[i][j] * dvydz[i][j];
        }
    }
}
void updateSXZ(float **sxx, float **szz, float **sxz, float **dvxdx, float **dvxdz, float **dvzdx, float **dvzdz, float **lambda, float **mu, float dt, int nx, int nz){
    for(int i = 0; i < nx; i++){
        for(int j = 0; j < nz; j++){
            sxx[i][j] += dt * ((lambda[i][j] + 2 * mu[i][j]) * dvxdx[i][j] + lambda[i][j] * dvzdz[i][j]);
            szz[i][j] += dt * ((lambda[i][j] + 2 * mu[i][j]) * dvzdz[i][j] + lambda[i][j] * dvxdx[i][j]);
            sxz[i][j] += dt * (mu[i][j] * (dvxdz[i][j] + dvzdx[i][j]));
        }
    }
}
void updateU(float **u, float **v, float dt, int nx, int nz){
    for(int i = 0; i < nx; i++){
        for(int j = 0; j < nz; j++){
            u[i][j] += v[i][j] * dt;
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
            dat->order = root["order"];

            int single_src = root["src_info"].is<JsonObject>();
            dat->nsrc = single_src?1:root["src_info"].size();
            dat->stf_type = mat::createIntHost(dat->nsrc);
            dat->src_x = mat::createHost(dat->nsrc);
            dat->src_z = mat::createHost(dat->nsrc);
            dat->stf_PSV_x = mat::createHost(dat->nsrc);
            dat->stf_PSV_z = mat::createHost(dat->nsrc);
            dat->tauw_0 = mat::createHost(dat->nsrc);
            dat->tauw = mat::createHost(dat->nsrc);
            dat->tee_0 = mat::createHost(dat->nsrc);
            dat->f_min = mat::createHost(dat->nsrc);
            dat->f_max = mat::createHost(dat->nsrc);

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

            dat->absorb_left = root["absorb_left"];
            dat->absorb_right = root["absorb_right"];
            dat->absorb_top = root["absorb_top"];
            dat->absorb_bottom = root["absorb_bottom"];
            dat->absorb_width = root["width"];

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

            for(int i = 0; i < dat->nsrc; i++){
                JsonObject& src = single_src?root["src_info"]:((JsonArray&)root["src_info"]).get<JsonObject>(i);
                dat->src_x[i] = src["loc_x"];
                dat->src_z[i] = src["loc_z"];
                dat->stf_PSV_x[i] = src["stf_PSV"][0];
                dat->stf_PSV_z[i] = src["stf_PSV"][1];
                dat->tauw_0[i] = src["tauw_0"];
                dat->tauw[i] = src["tauw"];
                dat->tee_0[i] = src["tee_0"];
                dat->f_min[i] = src["f_min"];
                dat->f_max[i] = src["f_max"];

                const char* stf_type = src["stf_type"].as<char*>();
                if(strcmp(stf_type,"delta") == 0){
                    dat->stf_type[i] = 0;
                }
                else if(strcmp(stf_type,"delta_bp") == 0){
                    dat->stf_type[i] = 1;
                }
                else if(strcmp(stf_type,"ricker") == 0){
                    dat->stf_type[i] = 2;
                }
                else if(strcmp(stf_type,"heaviside_bp") == 0){
                    dat->stf_type[i] = 3;
                }
                else{
                    dat->stf_type[i] = -1;
                }
            }

            int single_rec = root["rec_x"].is<float>();
            dat->nrec = single_rec?1:root["rec_x"].size();
            dat->rec_x = mat::createHost(dat->nrec);
            dat->rec_z = mat::createHost(dat->nrec);
            for(int i = 0; i < dat->nrec; i++){
                dat->rec_x[i] = single_rec?root["rec_x"]:((JsonArray&)root["rec_x"]).get<float>(i);
                dat->rec_z[i] = single_rec?root["rec_z"]:((JsonArray&)root["rec_z"]).get<float>(i);
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
    dat->stf_x = mat::createHost(dat->nsrc, nt);
    dat->stf_y = mat::createHost(dat->nsrc, nt);
    dat->stf_z = mat::createHost(dat->nsrc, nt);
    float amp = dat->source_amplitude / dat->dx / dat->dz;
    float *stfn = mat::createHost(dat->nt);
    for(int i=0; i < dat->nsrc; i++){
        makeSourceTimeFunction(dat, stfn, i);
        float px = dat->stf_PSV_x[i];
        float pz = dat->stf_PSV_z[i];
        float norm = sqrt(pow(px,2) + pow(pz,2));
        for(int j = 0; j < nt; j++){
            dat->stf_x[i][j] = amp * stfn[j] * px / norm;
            dat->stf_y[i][j] = amp * stfn[j];
            dat->stf_z[i][j] = amp * stfn[j] * pz / norm;
        }
    }
    free(stfn);
}
void defineMaterialParameters(fdat *dat){
    // other model_type: later
    int nx = dat->nx;
    int nz = dat->nz;
    switch(dat->model_type){
        case 1:{
            mat::initHost(dat->rho, nx, nz, 3000);
            mat::initHost(dat->mu, nx, nz, 4.8e10);
            mat::initHost(dat->lambda, nx, nz, 4.8e10);
            break;
        }
        case 10:{
            mat::initHost(dat->rho, nx, nz, 2600);
            mat::initHost(dat->mu, nx, nz, 2.66e10);
            mat::initHost(dat->lambda, nx, nz, 3.42e10);
            break;
        }
    }
}
void computeIndices(int *coord_n_id, float *coord_n, float Ln, float n, int nthings){
    for(int i = 0; i < nthings;i++){
        coord_n_id[i] = (int)(coord_n[i] / Ln * (n - 1) + 0.5);
    }
}
void initialiseDynamicFields(fdat *dat){
    int nx = dat->nx;
    int nz = dat->nz;
    if(dat->wave_propagation_sh){
        mat::initHost(dat->vy, nx, nz, 0);
        mat::initHost(dat->uy, nx, nz, 0);
        mat::initHost(dat->sxy, nx, nz, 0);
        mat::initHost(dat->szy, nx, nz, 0);
    }
    if(dat->wave_propagation_psv){
        mat::initHost(dat->vx, nx, nz, 0);
        mat::initHost(dat->vz, nx, nz, 0);
        mat::initHost(dat->ux, nx, nz, 0);
        mat::initHost(dat->uz, nx, nz, 0);
        mat::initHost(dat->sxx, nx, nz, 0);
        mat::initHost(dat->szz, nx, nz, 0);
        mat::initHost(dat->sxz, nx, nz, 0);
    }
    // initialise kernels: later
}
void initialiseAbsorbingBoundaries(fdat *dat){
    mat::initHost(dat->absbound, dat->nx, dat->nz, 1);
    float width = dat->absorb_width;
    for(int i = 0; i < dat->nx; i++){
        for(int j = 0; j < dat->nz; j++){
            float X = i * dat->dx;
            float Z = j * dat->dz;
            if(dat->absorb_left){
                if(X < width){
                    dat->absbound[i][j] *= exp(-pow((X - width) / (2 * width), 2));
                }
            }
            if(dat->absorb_right){
                if(X > dat->Lx - width){
                    dat->absbound[i][j] *= exp(-pow((X - (dat->Lx - width)) / (2 * width), 2));
                }
            }
            if(dat->absorb_bottom){
                if(Z < width){
                    dat->absbound[i][j] *= exp(-pow((Z - width) / (2 * width), 2));
                }
            }
            if(dat->absorb_top){
                if(Z > dat->Lz - width){
                    dat->absbound[i][j] *= exp(-pow((Z - (dat->Lz - width)) / (2 * width), 2));
                }
            }
        }
    }
}
void runWaveFieldPropagation(fdat *dat){
    initialiseDynamicFields(dat);
    initialiseAbsorbingBoundaries(dat);

    int sh = dat->wave_propagation_sh;
    int psv = dat->wave_propagation_psv;
    int order = dat->order;
    int mode = dat->simulation_mode;

    int nx = dat->nx;
    int nz = dat->nz;
    int nt = dat->nt;
    float dx = dat->dx;
    float dz = dat->dz;
    float dt = dat->dt;

    for(int n = 0; n < dat->nt; n++){
        if((n + 1) % dat->sfe == 0){
            int isfe = (int)(nt / dat->sfe) - (n + 1) / dat->sfe;
            if(sh){
                copyMat(dat->uy_forward[isfe], dat->uy, nx, nz);
            }
            if(psv){
                copyMat(dat->ux_forward[isfe], dat->ux, nx, nz);
                copyMat(dat->uz_forward[isfe], dat->uz, nx, nz);
            }
        }
        if(sh){
            divSY(dat->dsy, dat->sxy, dat->szy, dx, dz, nx, nz, order);
        }
        if(psv){
            divSXZ(dat->dsx, dat->dsz, dat->sxx, dat->szz, dat->sxz, dx, dz, nx, nz, order);
        }
        if(mode == 0 || mode == 1){
            for(int is = 0; is < dat->nsrc; is++){
                int xs = dat->src_x_id[is];
                int zs = dat->src_z_id[is];
                if(sh){
                    dat->dsy[xs][zs] += dat->stf_y[is][n];
                }
                if(psv){
                    dat->dsx[xs][zs] += dat->stf_x[is][n];
                    dat->dsz[xs][zs] += dat->stf_z[is][n];
                }
            }
        }
        if(sh){
            updateV(dat->vy, dat->dsy, dat->rho, dat->absbound, dt, nx, nz);
            divVY(dat->dvydx, dat->dvydz, dat->vy, dx, dz, nx, nz, dat->order);
            updateSY(dat->sxy, dat->szy, dat->dvydx, dat->dvydz, dat->mu, dt, nx, nz);
            updateU(dat->uy, dat->vy, nx, nz, dt);
        }
        if(psv){
            updateV(dat->vx, dat->dsx, dat->rho, dat->absbound, dt, nx, nz);
            updateV(dat->vz, dat->dsz, dat->rho, dat->absbound, dt, nx, nz);
            divVXZ(dat->dvxdx, dat->dvxdz, dat->dvzdx, dat->dvzdz, dat->vx, dat->vz, dx, dz, nx, nz, order);
            updateSXZ(dat->sxx, dat->szz, dat->sxz, dat->dvxdx, dat->dvxdz, dat->dvzdx, dat->dvzdz, dat->lambda, dat->mu, dt, nx, nz);
            updateU(dat->ux, dat->vx, dt, nx, nz);
            updateU(dat->uz, dat->vz, dt, nx, nz);
        }
        if(mode == 0){
            for(int ir = 0; ir < dat->nrec; ir++){
                int xr = dat->rec_x_id[ir];
                int zr = dat->rec_z_id[ir];
                if(sh){
                    dat->v_rec_y[ir][n] = dat->vy[xr][zr];
                }
                if(psv){
                    dat->v_rec_x[ir][n] = dat->vx[xr][zr];
                    dat->v_rec_z[ir][n] = dat->vz[xr][zr];
                }
            }
            if((n + 1) % dat->sfe == 0){
                int isfe = (int)(nt / dat->sfe) - (n + 1) / dat->sfe;
                if(sh){
                    copyMat(dat->vy_forward[isfe], dat->vy, nx, nz);
                }
                if(psv){
                    copyMat(dat->vx_forward[isfe], dat->vx, nx, nz);
                    copyMat(dat->vz_forward[isfe], dat->vz, nx, nz);
                }
            }
        }
        else if(mode == 1){
            // adjoint: later
        }
    }
    char oname[50];
    for(int i = 0; i < dat->nrec; i++){
        sprintf(oname, "vx%d", i);
        mat::write(dat->v_rec_x[i], nt, oname);
        sprintf(oname, "vz%d", i);
        mat::write(dat->v_rec_z[i], nt, oname);
    }
}
void checkArgs(fdat *dat){
    int nx = dat->nx;
    int nz = dat->nz;
    int nsfe = (int)(dat->nt / dat->sfe);

    dat->dx = dat->Lx / (nx - 1);
    dat->dz = dat->Lz / (nz - 1);

    if(dat->wave_propagation_sh){
        dat->vy = mat::createHost(nx, nz);
        dat->uy = mat::createHost(nx, nz);
        dat->sxy = mat::createHost(nx, nz);
        dat->szy = mat::createHost(nx, nz);
        dat->dsy = mat::createHost(nx, nz);
        dat->dvydx = mat::createHost(nx, nz);
        dat->dvydz = mat::createHost(nx, nz);
        dat->v_rec_y = mat::createHost(dat->nrec, dat->nt);
        dat->uy_forward = mat::createHost(nsfe, nx, nz);
        dat->vy_forward = mat::createHost(nsfe, nx, nz);
    }
    if(dat->wave_propagation_psv){
        dat->vx = mat::createHost(nx, nz);
        dat->vz = mat::createHost(nx, nz);
        dat->ux = mat::createHost(nx, nz);
        dat->uz = mat::createHost(nx, nz);
        dat->sxx = mat::createHost(nx, nz);
        dat->szz = mat::createHost(nx, nz);
        dat->sxz = mat::createHost(nx, nz);
        dat->dsx = mat::createHost(nx, nz);
        dat->dsz = mat::createHost(nx, nz);
        dat->dvxdx = mat::createHost(nx, nz);
        dat->dvxdz = mat::createHost(nx, nz);
        dat->dvzdx = mat::createHost(nx, nz);
        dat->dvzdz = mat::createHost(nx, nz);
        dat->v_rec_x = mat::createHost(dat->nrec, dat->nt);
        dat->v_rec_z = mat::createHost(dat->nrec, dat->nt);
        dat->ux_forward = mat::createHost(nsfe, nx, nz);
        dat->uz_forward = mat::createHost(nsfe, nx, nz);
        dat->vx_forward = mat::createHost(nsfe, nx, nz);
        dat->vz_forward = mat::createHost(nsfe, nx, nz);
    }

    dat->absbound = mat::createHost(nx, nz);
    dat->lambda = mat::createHost(nx, nz);
    dat->rho = mat::createHost(nx, nz);
    dat->mu = mat::createHost(nx, nz);

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

    dat->src_x_id = mat::createIntHost(dat->nsrc);
    dat->src_z_id = mat::createIntHost(dat->nsrc);
    dat->rec_x_id = mat::createIntHost(dat->nrec);
    dat->rec_z_id = mat::createIntHost(dat->nrec);
    computeIndices(dat->src_x_id, dat->src_x, dat->Lx, dat->nx, dat->nsrc);
    computeIndices(dat->src_z_id, dat->src_z, dat->Lz, dat->nz, dat->nsrc);
    computeIndices(dat->rec_x_id, dat->rec_x, dat->Lx, dat->nx, dat->nrec);
    computeIndices(dat->rec_z_id, dat->rec_z, dat->Lz, dat->nz, dat->nrec);

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
}

int main(int argc , char *argv[]){
    for(int i = 0; i< argc; i++){
        if(strcmp(argv[i],"runForward") == 0){
            runForward();
        }
    }
    // float **a=mat::createHost(8,8);
    // float **b=mat::createHost(8,8);
    // float **e=mat::createHost(8,8);
    // float **c=mat::createHost(8,8);
    // float **d=mat::createHost(8,8);
    // float **cc=mat::createHost(8,8);
    // float **dd=mat::createHost(8,8);
    //
    // mat::initHost(c,8,8,0);
    // for(int i=0;i<8;i++){
    //     for(int j=0;j<8;j++){
    //         a[i][j]=(i+5)*(j+7)-(float)(i+2)/(j+6);
    //         b[i][j]=(i+1)*(j+9)+(float)(i+3)/(j+4);
    //         e[i][j]=(i+11)*(j+19)+(float)(i+13)/(j+14);
    //     }
    // }
    // divVXZ(c, d,cc,dd, b,a, 1, 1, 8, 8, 4);
    // printMat(cc,8,8);
    // printMat(dd,8,8);

    return 0;
}
