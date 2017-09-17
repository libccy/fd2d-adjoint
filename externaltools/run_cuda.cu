#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <map>

using namespace std;

map<string, float> import_data(void){
    map<string, float> config;
    FILE *cfgfile = fopen("externaltools/config","r");
    char buf[50];
    while(fgets(buf, 50, cfgfile)){
        int len = strlen(buf);
        char *cfgkey = (char *) malloc(len);
        strncpy(cfgkey, buf, len-1);
        cfgkey[len-1] = '\0';
        float cfgvalue;
        fscanf(cfgfile, "%f\n", &cfgvalue);
        config[cfgkey] = cfgvalue;
    }
    return config;
}
void run_wavefield_propagation(void){

}
void run_forward(void){
    map<string, float> config = import_data();
    printf("%f %f\n",config["nx"],config["nz"]);
}

int main(int argc , char *argv[]){
    for(int i = 0; i< argc; i++){
        if(strcmp(argv[i],"run_forward") == 0){
            run_forward();
        }
    }
    return 0;
}
