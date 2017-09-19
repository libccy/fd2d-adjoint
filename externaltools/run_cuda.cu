#include <stdio.h>
#include <stdlib.h>
#include "ArduinoJson.h"

#define devij(dimx, dimy) \
int i = blockIdx.x % dimx; \
int j = threadIdx.x + (blockIdx.x - i) / dimx * dimy / d_nbt; \
int ij = i * dimy + j

typedef struct{
    int nx;
    int nz;
    int nt;
    float dt;
    float Lx;
    float Lz;

    int nsrc;
    int *stf_type;
    int *stf_PSV;
    float *src_x;
    float *src_z;
    float *tauw_0;
    float *tauw;
    float *tee_0;
    float *f_min;
    float *f_max;

} fwdcfg;
typedef struct{
    float *stf_x;
    float *stf_y;
    float *stf_z;
} fwdarg;
typedef struct{
    float *vx;
    float *vz;
} fwddat;

const int nbt = 8;
__constant__ int d_nbt = 8;

namespace mat{
    float *create(const int m) {
    	float *a;
    	cudaMalloc((void**)&a, m * sizeof(float));
    	return a;
    }
    float *create_h(const int m) {
    	return (float *)malloc(m * sizeof(float));
    }
    int *create_i(const int m){
        int *a;
    	cudaMalloc((void**)&a, m * sizeof(int));
    	return a;
    }
    int *create_hi(const int m) {
    	return (int *)malloc(m * sizeof(int));
    }

    __global__ void set_d(float *a, const float init, const int m, const int n){
        devij(m, n);
        a[ij] = init;
    }
    void set(float *a, const float init, const int m, const int n){
        mat::set_d<<<m * nbt, n / nbt>>>(a, init, m, n);
    }
    void copyhd(float *d_a, const float *a, const int m){
        cudaMemcpy(d_a, a , m * sizeof(float), cudaMemcpyHostToDevice);
    }
    void copydh(float *a, const float *d_a, const int m){
        cudaMemcpy(a, d_a , m * sizeof(float), cudaMemcpyDeviceToHost);
    }
    void write(FILE *file, float *d_a, float *a, const int m){
        mat::copydh(a, d_a, m);
        fwrite(a, sizeof(float), m, file);
    }
    void read(FILE *file, float *a, const int m){
        fread(a, sizeof(float), m, file);
    }
}

fwdcfg import_data(void){
    fwdcfg cfg;
    FILE *cfgfile = fopen("externaltools/config","r");

    char *buffer = 0;
    long length;

    fseek (cfgfile, 0, SEEK_END);
    length = ftell (cfgfile);
    fseek (cfgfile, 0, SEEK_SET);
    buffer = (char *)malloc (length + 1);
    fread (buffer, 1, length, cfgfile);
    buffer[length] = '\0';

    fclose(cfgfile);

    if (buffer){
        DynamicJsonBuffer jsonBuffer;
        JsonObject& root = jsonBuffer.parseObject(buffer);
        if (!root.success()){
            printf("parseObject() failed\n");
        }
        else{
            cfg.nx = root["nx"];
            cfg.nz = root["nz"];
            cfg.nt = root["nt"];
            cfg.dt = root["dt"];
            cfg.Lx = root["Lx"];
            cfg.Lz = root["Lz"];

            if(root["src_info"].is<JsonObject>()){
                JsonObject& src = root["src_info"];
                DynamicJsonBuffer jsonBufferSrc;
                JsonArray& src_info = jsonBufferSrc.createArray();
                src_info.add(src);
                root.set("src_info",src_info);
            }

            JsonArray& src_info = root["src_info"];
            cfg.nsrc = src_info.size();
            cfg.stf_type = mat::create_hi(cfg.nsrc * sizeof(int));
            cfg.stf_PSV = mat::create_hi(cfg.nsrc * sizeof(int));
            cfg.src_x = mat::create_h(cfg.nsrc * sizeof(int));
            cfg.src_z = mat::create_h(cfg.nsrc * sizeof(int));
            cfg.tauw_0 = mat::create_h(cfg.nsrc * sizeof(float));
            cfg.tauw = mat::create_h(cfg.nsrc * sizeof(float));
            cfg.tee_0 = mat::create_h(cfg.nsrc * sizeof(float));
            cfg.f_min = mat::create_h(cfg.nsrc * sizeof(float));
            cfg.f_max = mat::create_h(cfg.nsrc * sizeof(float));

            for(int i = 0; i < cfg.nsrc; i++){
                JsonObject& src = src_info.get<JsonObject>(i);
                cfg.src_x[i] = src["loc_x"];
                cfg.src_z[i] = src["loc_z"];
                cfg.stf_type[i] = 1; // ricker modify later
                cfg.stf_PSV[i] = 0; // radiate modify later
                cfg.tauw_0[i] = src["tauw_0"];
                cfg.tauw[i] = src["tauw"];
                cfg.tee_0[i] = src["tee_0"];
                cfg.f_min[i] = src["f_min"];
                cfg.f_max[i] = src["f_max"];
            }

            printf("src %f %f %f\n", cfg.src_x[0], cfg.src_z[0],cfg.tauw_0[0]);
        }
    }
    return cfg;
}
float *import_data(char *path, int *len){
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
        data = mat::create_h(*len);
        for(int i=0; i<*len; i++){
            fscanf(datafile, "%f\n", data + i);
        }
        fclose(datafile);
    }
    return data;
}
void define_computational_domain(float Lx, float Lz, int nx, int nz, float *dx, float *dz){
    *dx = Lx / (nx - 1);
    *dz = Lz / (nz - 1);
}
fwdarg prepare_stf(fwdcfg cfg){
    float dx, dz;
    define_computational_domain(cfg.Lx, cfg.Lz, cfg.nx, cfg.nz, &dx, &dz);
    float *t = mat::create_h(cfg.nt);
    for(int i = 0; i < cfg.nt; i++){
        t[i] = i * cfg.dt;
    }
    fwdarg stf;// from here
    float *data = mat::create_h(cfg.nt);
    for(int i=0;i<cfg.nt;i++){
        data[i]=i;
        // from here
    }
    stf.stf_x = data;
    printf("lxz %f %f\n",dx,dz);
    return stf;
}
fwdarg checkstf(fwdcfg cfg){
    int len;
    float *stfall = import_data("stf", &len);
    if(len > 0){
        fwdarg stf;
        stf.stf_x = stfall;
        return stf;
    }
    else{
        return prepare_stf(cfg);
    }
}
void run_wavefield_propagation(void){

}
void run_forward(void){
    fwdcfg cfg = import_data();
    fwdarg stf = checkstf(cfg);
    printf("nx: %d\nnt: %d\n", cfg.nx, cfg.nt);
    printf("stf: %f %f %f\n",stf.stf_x[0],stf.stf_x[1],stf.stf_x[2]);
}

int main(int argc , char *argv[]){
    for(int i = 0; i< argc; i++){
        if(strcmp(argv[i],"run_forward") == 0){
            run_forward();
        }
    }
    return 0;
}
