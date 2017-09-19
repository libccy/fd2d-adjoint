#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "ArduinoJson.h"

#define devij(dimx, dimy) \
int i = blockIdx.x % dimx; \
int j = threadIdx.x + (blockIdx.x - i) / dimx * dimy / d_nbt; \
int ij = i * dimy + j

const float pi = 3.14159265;

typedef struct{
    int nx;
    int nz;
    int nt;
    float dt;
    float Lx;
    float Lz;

    int sfe;
    int updateParams;
    int model_type;

    int nsrc;
    int *stf_type;
    float *stf_PSV_x;
    float *stf_PSV_z;
    float *src_x;
    float *src_z;
    float *tauw_0;
    float *tauw;
    float *tee_0;
    float *f_min;
    float *f_max;

    float *stf_x;
    float *stf_y;
    float *stf_z;
} fdat;

const int nbt = 8;
__constant__ int d_nbt = 8;

namespace mat{
    float *create(const int m) {
    	float *a;
    	cudaMalloc((void**)&a, m * sizeof(float));
    	return a;
    }
    float *createHost(const int m) {
    	return (float *)malloc(m * sizeof(float));
    }
    int *createInt(const int m){
        int *a;
    	cudaMalloc((void**)&a, m * sizeof(int));
    	return a;
    }
    int *createIntHost(const int m) {
    	return (int *)malloc(m * sizeof(int));
    }

    float maxHost(float *a, const int m) {
        // whether needed: modify later
        float max = fabs(a[0]);
        for(int i = 1; i < m; i++){
            if(fabs(a[i]) > max){
                max = fabs(a[i]);
            }
        }
        return max;
    }

    __global__ void setDevice(float *a, const float init, const int m, const int n){
        devij(m, n);
        a[ij] = init;
    }
    void set(float *a, const float init, const int m, const int n){
        mat::setDevice<<<m * nbt, n / nbt>>>(a, init, m, n);
    }
    void copyHostToDevice(float *d_a, const float *a, const int m){
        cudaMemcpy(d_a, a , m * sizeof(float), cudaMemcpyHostToDevice);
    }
    void copyDeviceToHost(float *a, const float *d_a, const int m){
        cudaMemcpy(a, d_a , m * sizeof(float), cudaMemcpyDeviceToHost);
    }
    void write(FILE *file, float *d_a, float *a, const int m){
        mat::copyDeviceToHost(a, d_a, m);
        fwrite(a, sizeof(float), m, file);
    }
    void read(FILE *file, float *a, const int m){
        fread(a, sizeof(float), m, file);
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
            dat->model_type = root["model_type"];

            if(root["src_info"].is<JsonObject>()){
                JsonObject& src = root["src_info"];
                DynamicJsonBuffer jsonBufferSrc;
                JsonArray& src_info = jsonBufferSrc.createArray();
                src_info.add(src);
                root.set("src_info",src_info);
            }

            JsonArray& src_info = root["src_info"];
            dat->nsrc = src_info.size();
            dat->stf_type = mat::createIntHost(dat->nsrc * sizeof(int));
            dat->src_x = mat::createHost(dat->nsrc * sizeof(float));
            dat->src_z = mat::createHost(dat->nsrc * sizeof(float));
            dat->stf_PSV_x = mat::createHost(dat->nsrc * sizeof(float));
            dat->stf_PSV_z = mat::createHost(dat->nsrc * sizeof(float));
            dat->tauw_0 = mat::createHost(dat->nsrc * sizeof(float));
            dat->tauw = mat::createHost(dat->nsrc * sizeof(float));
            dat->tee_0 = mat::createHost(dat->nsrc * sizeof(float));
            dat->f_min = mat::createHost(dat->nsrc * sizeof(float));
            dat->f_max = mat::createHost(dat->nsrc * sizeof(float));

            for(int i = 0; i < dat->nsrc; i++){
                JsonObject& src = src_info.get<JsonObject>(i);
                dat->src_x[i] = src["loc_x"];
                dat->src_z[i] = src["loc_z"];
                dat->stf_type[i] = 2; // ricker: modify later
                dat->stf_PSV_x[i] = src["stf_PSV"][0];
                dat->stf_PSV_z[i] = src["stf_PSV"][1];
                dat->tauw_0[i] = src["tauw_0"];
                dat->tauw[i] = src["tauw"];
                dat->tee_0[i] = src["tee_0"];
                dat->f_min[i] = src["f_min"];
                dat->f_max[i] = src["f_max"];
            }
        }
    }
    return dat;
}
float *importData(char *path, int *len){
    char fpath[50] = "externaltools/";
    strcat(fpath, path);
    *len = 0;
    float *data = 0;
    FILE *datafile = fopen(fpath,"r");
    if(datafile){
        while(!feof(datafile)){
            float datavalue;
            fscanf(datafile, "%f\n", &datavalue);
            *len = *len + 1;
        }
        fclose(datafile);

        datafile = fopen(fpath,"r");
        data = mat::createHost(*len);
        for(int i=0; i<*len; i++){
            fscanf(datafile, "%f\n", data + i);
        }
        fclose(datafile);
    }
    return data;
}
void exportData(float *data, int len, char *fname){
    char buffer[50] = "externaltools/";
    strcat(buffer, fname);
    FILE *file = fopen(buffer, "w");
    for(int i = 0; i < len; i++){
        fprintf(file, "%f\n", data[i]);
    }
    fclose(file);
}
void defineComputationalDomain(float Lx, float Lz, int nx, int nz, float *dx, float *dz){
    *dx = Lx / (nx - 1);
    *dz = Lz / (nz - 1);
}
float *makeSourceTimeFunction(fdat *dat, int index){
    float *stf = mat::createHost(dat->nt);
    float max = 0;
    float alfa = 2 * dat->tauw_0[index] / dat->tauw[index];
    for(int i = 0; i < dat->nt; i++){
        float t = i * dat->dt;
        switch(dat -> stf_type[index]){
            case 2:{
                stf[i] = (-2 * pow(alfa, 3) / pi) * (t - dat->tee_0[index]) * exp(-pow(alfa, 2) * pow(t - dat->tee_0[index], 2));
                break;
            }
            // other stf: modify later
        }

        if(fabs(stf[i]) > max){
            max = fabs(stf[i]);
        }
    }
    if(max > 0){
        for(int i = 0; i > dat->nt; i++){
            stf[i] /= max;
        }
    }
    return stf;
}
void prepareSTF(fdat *dat){
    float dx, dz;
    defineComputationalDomain(dat->Lx, dat->Lz, dat->nx, dat->nz, &dx, &dz);
    float *t = mat::createHost(dat->nt);
    int nt = dat->nt;
    for(int i = 0; i < nt; i++){
        t[i] = i * dat->dt;
    }

    dat->stf_x = mat::createHost(nt * dat->nsrc);
    dat->stf_y = mat::createHost(nt * dat->nsrc);
    dat->stf_z = mat::createHost(nt * dat->nsrc);
    for(int i=0; i < dat->nsrc; i++){
        float *stfn = makeSourceTimeFunction(dat, i);
        float px = dat->stf_PSV_x[i];
        float pz = dat->stf_PSV_z[i];
        float norm = sqrt(pow(px,2) + pow(pz,2));
        for(int j = 0; j < nt; j++){
            dat->stf_x[i * nt + j] = stfn[j] * px / norm;
            dat->stf_y[i * nt + j] = stfn[j];
            dat->stf_z[i * nt + j] = stfn[j] * pz / norm;
        }
    }
}
void checkArgs(fdat *dat){
    // int len;
    // add input file option: modify later
    // if updateParams == 1  defineMaterialParameters here
    // float *stfall = importData("stf", &len);
    // if(len > 0){
    //     stf.stf_x = stfall;
    //     return stf;
    // }
    dat -> updateParams = 0;
    prepareSTF(dat);
}
void defineMaterialParameters(fdat *dat){
    // from here
}
void runWaveFieldPropagation(void){

}
void runForward(void){
    printf("initialising...\n");
    fdat *dat = importData();
    checkArgs(dat);
    exportData(dat->stf_z,dat->nt,"stf_z");
    if(!dat->updateParams){
        defineMaterialParameters(dat);
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
