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

    float ***ux_forward;
    float ***uy_forward;
    float ***uz_forward;
    float ***vx_forward;
    float ***vy_forward;
    float ***vz_forward;

    float **absbound;
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
            dat->nt = root["nt"]; // nt = sfe*round(nt/sfe): later
            dat->dt = root["dt"];
            dat->Lx = root["Lx"];
            dat->Lz = root["Lz"];

            dat->sfe = root["sfe"];
            dat->model_type = root["model_type"];
            dat->use_given_model = root["use_given_model"];
            dat->use_given_stf = root["use_given_stf"];
            dat->source_amplitude = root["source_amplitude"];

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

void copyMat(float **a, float **b, int nx, int nz){
    // replace with copyDeviceToHost: later
    for(int i = 0; i < nx; i++){
        for(int j = 0; j < nz; j++){
            a[i][j] = b[i][j];
        }
    }
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
    int nsfe = (int)(dat->nt / dat->sfe);
    if(dat->wave_propagation_sh){
        mat::init(dat->vy, nx, nz, 0);
        mat::init(dat->uy, nx, nz, 0);
        mat::init(dat->sxy, nx, nz, 0);
        mat::init(dat->szy, nx, nz, 0);
        mat::init(dat->vy_forward, nsfe, nx, nz, 0);
    }
    if(dat->wave_propagation_psv){
        mat::init(dat->vx, nx, nz, 0);
        mat::init(dat->vz, nx, nz, 0);
        mat::init(dat->ux, nx, nz, 0);
        mat::init(dat->uz, nx, nz, 0);
        mat::init(dat->sxx, nx, nz, 0);
        mat::init(dat->szz, nx, nz, 0);
        mat::init(dat->sxz, nx, nz, 0);
        mat::init(dat->vx_forward, nsfe, nx, nz, 0);
        mat::init(dat->vz_forward, nsfe, nx, nz, 0);
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
    // later
    int nz=dat->nz;
    float *abs=mat::createHost(nz);
    for(int i=0;i<nz;i++){
        abs[i]=dat->absbound[i][i];
    }
    mat::write(abs,nz,"abs");
}
void runWaveFieldPropagation(fdat *dat){
    initialiseDynamicFields(dat);
    initialiseAbsorbingBoundaries(dat);
    printf("iterating...\n");

    int sh = dat->wave_propagation_sh;
    int psv = dat->wave_propagation_psv;
    int nsfe = (int)(dat->nt / dat->sfe);
    for(int n = 0; n < dat->nt; n++){
        if((n + 1) % dat->sfe == 0){
            if(sh){
                copyMat(dat->uy_forward[nsfe - n / dat->sfe], dat->uy, dat->nx, dat->nz);
            }
            if(psv){
                copyMat(dat->ux_forward[nsfe - n / dat->sfe], dat->ux, dat->nx, dat->nz);
                copyMat(dat->uz_forward[nsfe - n / dat->sfe], dat->uz, dat->nx, dat->nz);
            }
        }
    }
    // next: dsy dsx dsz copyMat
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
        dat->szy = mat::createHost(nx, nz);
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

    mat::write(dat->stf_z[0],dat->nt,"stf_z"); // later
}

int main(int argc , char *argv[]){
    for(int i = 0; i< argc; i++){
        if(strcmp(argv[i],"runForward") == 0){
            runForward();
        }
    }

    return 0;
}
