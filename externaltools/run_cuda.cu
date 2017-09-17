#include <stdio.h>
#include <stdlib.h>

#define devij(dimx, dimy) \
int i = blockIdx.x % dimx; \
int j = threadIdx.x + (blockIdx.x - i) / dimx * dimy / d_nbt; \
int ij = i * dimy + j

typedef struct{
    int nx;
    int nz;
    int nt;
    int set(char *key, float value){
        switch(key[0]){
            case 'n':{
                switch(key[1]){
                    case 'x': nx = (int)value; return 1;
                    case 'z': nz = (int)value; return 1;
                    case 't': nt = (int)value; return 1;
                    default: return 0;
                }
            }
            default: return 0;
        }
    }
} config;
typedef struct{
    float *x;
    float *y;
    float *z;
} stfunc;

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

config import_data(void){
    config cfg;
    FILE *cfgfile = fopen("externaltools/config","r");
    char cfgkey[50];
    while(fgets(cfgkey, 50, cfgfile)){
        float cfgvalue;
        fscanf(cfgfile, "%f\n", &cfgvalue);
        cfg.set(cfgkey, cfgvalue);
    }
    fclose(cfgfile);
    return cfg;
}
void import_data(char *path, float **data, int *len){
    char fpath[50] = "externaltools/";
    strcat(fpath, path);
    *len = 0;
    FILE *datafile = fopen(fpath,"r");
    if(datafile){
        while(!feof(datafile)){
            float datavalue;
            fscanf(datafile, "%f\n", &datavalue);
            *len = *len + 1;
        }
        fclose(datafile);

        datafile = fopen(fpath,"r");
        *data = mat::create_h(*len);
        for(int i=0; i<*len; i++){
            fscanf(datafile, "%f\n", *data + i);
        }
        fclose(datafile);
    }
}
stfunc prepare_stf(config cfg){
    stfunc stf;
    float *data = mat::create_h(cfg.nt);
    for(int i=0;i<cfg.nt;i++){
        data[i]=i;
        // from here
    }
    stf.x = data;
    return stf;
}
stfunc checkstf(config cfg){
    int len;
    float *stfall;
    import_data("stf", &stfall, &len);
    if(len > 0){
        stfunc stf;
        stf.x = stfall;
        return stf;
    }
    else{
        return prepare_stf(cfg);
    }
}
void run_wavefield_propagation(void){

}
void run_forward(void){
    config cfg = import_data();
    stfunc stf = checkstf(cfg);
    printf("nx: %d\nnt: %d\n", cfg.nx, cfg.nt);
    printf("stf: %f %f %f\n",stf.x[0],stf.x[1],stf.x[2]);
}

int main(int argc , char *argv[]){
    for(int i = 0; i< argc; i++){
        if(strcmp(argv[i],"run_forward") == 0){
            run_forward();
        }
    }
    return 0;
}
