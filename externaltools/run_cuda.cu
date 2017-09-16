#include <stdio.h>
#include <stdlib.h>

void run_wavefield_propagation(void){
    printf("prsdfsdfo\n");
}

int main(int argc , char *argv[]){
    for(int i = 0; i< argc; i++){
        if(strcmp(argv[i],"run_wavefield_propagation") == 0){
            run_wavefield_propagation();
        }
    }
    return 0;
}
