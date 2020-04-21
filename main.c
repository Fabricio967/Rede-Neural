#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define NUMENTRADAS 2
#define NEUOCULTA 4
#define NEUSAIDA 1
#define NUMAMOSTRAS 4
#define EPOCAS 10000

const float eta = 0.7;
const float erroMin = 0.1;

float desejado[NUMAMOSTRAS] = { 0, 1, 1, 0};
float entrada[NUMENTRADAS][1];
float pesosOculta[NEUOCULTA][NUMENTRADAS] = { {0.5, 0.7}, {0.75, 0.9} };
float oculta[NUMAMOSTRAS][NEUOCULTA];
float pesosSaida[NEUSAIDA][NEUOCULTA] = {0.45, 0.8};
float saida[NUMAMOSTRAS][NEUSAIDA];
float erroOculta[NUMAMOSTRAS][NEUSAIDA];
float gradienteOculta[NEUOCULTA][1];

float sigmoide(float z);
float dsig(float z);
float rnd();
FILE *fp;

float sigmoide(float z){

    return (1/(1+exp(-z)));

}

float dsig(float z){

    return (z*(1-z));
}

void treinar(){

    int i, j, k, p, numEpocas = 0;
    float soma = 0;
    float amostras[4][2] = { {0, 0}, {0, 1}, {1, 0}, {1, 1} };

    for(numEpocas = 0; numEpocas < EPOCAS; numEpocas++){
        p = 0;
        for(j = 0; j < NUMAMOSTRAS; j++){   //indica a amostra atual
            //printf("Amostra %d\n\n", j+1);
            //FEEDFORWARD
            // OCULTA
            for(k = 0; k < NUMENTRADAS; k++){   // atribuindo as entradas
                entrada[k][0] = amostras[j][k];
                //printf("%f\n", entrada[k][0]);
            }
            //printf("%f\n", entrada[0][0]);
            //printf("%f\n", entrada[1][0]);

            for(k = 0; k < NEUOCULTA; k++){     // fazendo a multiplicação da entrada pelos pesos da oculta
                soma = 0;
                for(i = 0; i < NUMENTRADAS; i++){
                    soma += pesosOculta[k][i] * entrada[i][0];
                    //printf("\nPeso[%d][%d] = %f | entrada[%d][0] = %f\n", k, i, pesosOculta[k][i], i, entrada[i][0]);
                }
                oculta[k][0] = soma;
                oculta[k][0] = sigmoide(oculta[k][0]);
                //printf("oculta = %f\n", oculta[k][0]);
                //system("pause");

            }

            // SAIDA
            for(k = 0; k < NEUSAIDA; k++){     // fazendo a multiplicação da oculta pelos pesos da saida
                soma = 0;
                for(i = 0; i < NEUOCULTA; i++){
                    soma += pesosSaida[k][i] * oculta[i][0];
                    //printf("\pesosSaida[%d][%d] = %f | oculta[%d][0] = %f\n", k, i, pesosSaida[k][i], i, oculta[i][0]); //erro aqui [problema com a oculta]
                    //getchar();
                }
                saida[j][0] = soma;
                saida[j][0] = sigmoide(saida[j][0]);
                //printf("saida = %f\n", saida[j][0]);
            }
            //system("pause");

            erroOculta[j][0] = (desejado[j] - saida[j][0]) * dsig(saida[j][0]);  //calculando o erro
            //printf("\nerro[%d][0] = %f | saida[%d][0] = %f | desejado[%d] = %f\n", j, erroOculta[j][0], j, saida[j][0], j, desejado[j]);
            //getchar();


            //BACKPROPAGATION

            //SAIDA
            for(k = 0; k < NEUOCULTA; k++){     //atualizando os pesos da saida
                //printf("\npesosSaida[0][%d] = %f", k, pesosSaida[0][k]);
                pesosSaida[0][k] += eta * erroOculta[j][0] * oculta[k][0];
                //printf("\npesosSaida[0][%d] = %f | erroOculta[%d][0] = %f | oculta[%d][0] = %f", k, pesosSaida[0][k], j, erroOculta[j][0], j, oculta[k][0]);
                //getchar();
            }
            //printf("\n--\n");
            //OCULTA
            for(k = 0; k < NEUOCULTA; k++){     //atualizano os pesos da oculta
                gradienteOculta[k][0] = erroOculta[j][0] * pesosSaida[0][k] * dsig(oculta[k][0]);
                //printf("\ngradienteOculta[%d][0] = %f | erroOculta[%d][0] = %f | pesosSaida[0][%d] = %f | dsigoculta[%d][0] = %f", k, gradienteOculta[k][0], j, erroOculta[j][0], k, pesosSaida[0][k], k, dsig(oculta[k][0]));
                for(i = 0; i < NUMENTRADAS; i++){
                    pesosOculta[k][i] += eta * gradienteOculta[k][0] * entrada[i][0];
                    //printf("\nentrada[%d][0] = %f | pesosOculta[%d][%d] = %f", i, entrada[i][0], k, i, pesosOculta[k][i]);
                    //getchar();
                }
            }
        }
        //printf("\n\n");

        if(numEpocas == EPOCAS-1){
            for(i = 0; i < NUMAMOSTRAS; i++){
                for(k = 0; k < NUMENTRADAS; k++){
                    printf("\nEntrada[%d] = \t%f", k, amostras[i][k]);
                }
                printf("\n");

                printf("\nSaida Obtida = \t%f", saida[i][0]);
                printf("\nSaida Desejado = \t%f", desejado[i]);
                printf("\nErro = \t%f", erroOculta[i][0]);
                printf("\n===================\n");

            }
        }
        //printf("\n===================\n");
    }// fim do for epocas
}// fim da funcao treinar

int main(){
    /*
    printf("dsig = %f\n", dsig(0.1));
    system("pause");*/



    treinar();


    return 1;
}
