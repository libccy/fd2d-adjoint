#include <stdio.h>
#include <stdlib.h>

using namespace std;

struct config{
    float nx;
    float nz;
    float nt;
    int set(char *key, float value){
        switch(key[0]){
            case 'n':{
                switch(key[1]){
                    case 'x': nx = value; return 1;
                    case 'z': nz = value; return 1;
                    case 't': nt = value; return 1;
                    default: return 0;
                }
            }
            default: return 0;
        }
    }
};

struct config import_data(void){
    struct config cfg;
    FILE *cfgfile = fopen("externaltools/config","r");
    char cfgkey[50];
    while(fgets(cfgkey, 50, cfgfile)){
        float cfgvalue;
        fscanf(cfgfile, "%f\n", &cfgvalue);
        cfg.set(cfgkey, cfgvalue);
    }
    return cfg;
}
void run_wavefield_propagation(void){

}
void run_forward(void){
    struct config cfg = import_data();
    printf("nx: %f\nnt: %f\n", cfg.nx, cfg.nt);
}

int main(int argc , char *argv[]){
    for(int i = 0; i< argc; i++){
        if(strcmp(argv[i],"run_forward") == 0){
            run_forward();
        }
    }
    return 0;
}
