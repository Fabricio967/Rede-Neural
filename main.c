/**
    Autor: Fabrício Henrique da Silva
    Aluno de Analise e Desenvolvimento de Sistemas
    Rede Neural Perceptron
    Data: 20/03/2020
    Atualização: 21/04/2020
**/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define NUMENTRADAS 3
#define NEUOCULTA 4
#define NEUSAIDA 1
#define NUMAMOSTRAS 6
#define EPOCAS 10000
#define MIN -0.5
#define MAX 0.5
#define randn() (((double)rand()/((double)RAND_MAX + 1)) * (MAX - MIN)) + MIN

const float eta = 0.1;
FILE *arqPesoOculta, *arqPesoSaida, *arqAmostras;

float desejado[NUMAMOSTRAS] = { 0, 0, 1, 0, 1, 1};
float entrada[NUMENTRADAS][1];
float pesosOculta[NEUOCULTA][NUMENTRADAS];
float oculta[NUMAMOSTRAS][NEUOCULTA];
float pesosSaida[NEUSAIDA][NEUOCULTA];
float saida[NUMAMOSTRAS][NEUSAIDA];
float erroOculta[NUMAMOSTRAS][NEUSAIDA];
float gradienteOculta[NEUOCULTA][1];
float amostras[NUMAMOSTRAS][NUMENTRADAS];

float sigmoide(float z);
float dsig(float z);
void preencherMatrizes();
int salvarPesos();
void previsao();

void preencherMatrizes(){

    int i, j;

    for(i = 0; i < NEUOCULTA; i++){
        for(j = 0; j < NUMENTRADAS; j++){
            pesosOculta[i][j] = randn();
        }
    }

    for(i = 0; i < NEUSAIDA; i++){
        for(j = 0; j < NEUOCULTA; j++){
            pesosSaida[i][j] = randn();
        }
    }
}

float sigmoide(float z){

    return (1/(1+exp(-z)));

}

float dsig(float z){

    return (z*(1-z));
}

void treinar(){

    int i, j, k, numEpocas = 0;
    float soma = 0;

    arqAmostras = fopen("amostras.txt", "r");
    if(!arqAmostras){
        printf("\nErro ao abrir arquivo de amostras.\n");
        return 1;
    }

    for(i = 0; i < NUMAMOSTRAS; i++){
        for(j = 0; j < NUMENTRADAS; j++){
            fscanf(arqAmostras, "%f ", &amostras[i][j]);
            printf("[%d][%d] = %f\t", i, j, amostras[i][j]);
        }
        printf("\n");
    }
    //getchar();

    preencherMatrizes();

    for(numEpocas = 0; numEpocas < EPOCAS; numEpocas++){
        for(j = 0; j < NUMAMOSTRAS; j++){   // Indica a amostra atual

            /* FEEDFORWARD */
            // OCULTA
            for(k = 0; k < NUMENTRADAS; k++){   // Recebendo as entradas das amostras
                entrada[k][0] = amostras[j][k];
                //printf("%f\n", entrada[k][0]);
            }

            for(k = 0; k < NEUOCULTA; k++){     // Multiplicação de pesos da oculta pela entrada
                soma = 0;
                for(i = 0; i < NUMENTRADAS; i++){
                    soma += pesosOculta[k][i] * entrada[i][0];
                    //printf("\nPeso[%d][%d] = %f | entrada[%d][0] = %f\n", k, i, pesosOculta[k][i], i, entrada[i][0]);
                }
                oculta[k][0] = soma;
                oculta[k][0] = sigmoide(oculta[k][0]);
                //printf("oculta = %f\n", oculta[k][0]);
            }

            // SAIDA
            for(k = 0; k < NEUSAIDA; k++){     // Multiplicação de pesos da saida pela oculta
                soma = 0;
                for(i = 0; i < NEUOCULTA; i++){
                    soma += pesosSaida[k][i] * oculta[i][0];
                    //printf("\pesosSaida[%d][%d] = %f | oculta[%d][0] = %f\n", k, i, pesosSaida[k][i], i, oculta[i][0]);
                }
                saida[j][0] = soma;
                saida[j][0] = sigmoide(saida[j][0]);
                //printf("saida = %f\n", saida[j][0]);
            }

            erroOculta[j][0] = (desejado[j] - saida[j][0]) * dsig(saida[j][0]);  //calculando o erro
            //printf("\nerro[%d][0] = %f | saida[%d][0] = %f | desejado[%d] = %f\n", j, erroOculta[j][0], j, saida[j][0], j, desejado[j]);

            /* BACKPROPAGATION */

            //SAIDA
            for(k = 0; k < NEUOCULTA; k++){     // Atualiza pesos da Saida
                //printf("\npesosSaida[0][%d] = %f", k, pesosSaida[0][k]);
                pesosSaida[0][k] += eta * erroOculta[j][0] * oculta[k][0];
                //printf("\npesosSaida[0][%d] = %f | erroOculta[%d][0] = %f | oculta[%d][0] = %f", k, pesosSaida[0][k], j, erroOculta[j][0], j, oculta[k][0]);
            }
            //OCULTA
            for(k = 0; k < NEUOCULTA; k++){     // Atualiza pesos da Oculta
                gradienteOculta[k][0] = erroOculta[j][0] * pesosSaida[0][k] * dsig(oculta[k][0]);
                //printf("\ngradienteOculta[%d][0] = %f | erroOculta[%d][0] = %f | pesosSaida[0][%d] = %f | dsigoculta[%d][0] = %f", k, gradienteOculta[k][0], j, erroOculta[j][0], k, pesosSaida[0][k], k, dsig(oculta[k][0]));
                for(i = 0; i < NUMENTRADAS; i++){
                    pesosOculta[k][i] += eta * gradienteOculta[k][0] * entrada[i][0];
                    //printf("\nentrada[%d][0] = %f | pesosOculta[%d][%d] = %f", i, entrada[i][0], k, i, pesosOculta[k][i]);
                }
            }
        }

        if(numEpocas == EPOCAS-1){
            for(i = 0; i < NUMAMOSTRAS; i++){
                for(k = 0; k < NUMENTRADAS; k++){
                    printf("\nEntrada[%d] = \t%f", k, amostras[i][k]);
                }
                printf("\n\nSaida Obtida = \t%f", saida[i][0]);
                printf("\nSaida Desejado = \t%f", desejado[i]);
                //printf("\nErro = \t%f", erroOculta[i][0]);
                printf("\n===================\n");
            }
        }
    }// fim do for epocas
}// fim da funcao treinar

int salvarPesos(){

    int i, j;

    //Salvando os pesos da Oculta
    arqPesoOculta = fopen("pesosOculta.txt", "r+");
    if(!arqPesoOculta){
        printf("\nErro ao abrir arquivo.\n");
        return 1;
    }

    for(i = 0; i < NEUOCULTA; i++){
        for(j = 0; j < NUMENTRADAS; j++){
            fprintf(arqPesoOculta, "%f ", pesosOculta[i][j]);
        }
        fprintf(arqPesoOculta, "\n");
    }

    arqPesoSaida = fopen("pesosSaida.txt", "r+");
    if(!arqPesoSaida){
        printf("\nErro ao abrir arquivo.\n");
        return 1;
    }

    for(i = 0; i < NEUSAIDA; i++){
        for(j = 0; j < NEUOCULTA; j++){
            fprintf(arqPesoSaida, "%f ", pesosSaida[i][j]);
        }
        fprintf(arqPesoSaida, "\n");
    }
    fclose(arqPesoOculta);
    fclose(arqPesoSaida);

    return 0;
}

void previsao(){

    int i, j, k;
    float soma;
    FILE *arqPrevisao;

    arqPrevisao = fopen("previsao.txt", "r");
    if(!arqPrevisao){
        printf("\nEro ao abrir arquivo de previsao.\n");
        return 1;
    }

    for(i = 0; i < 1; i++){
        for(j = 0; j < NUMENTRADAS; j++){
            fscanf(arqPrevisao, "%f ", &entrada[i][j]);
            printf("[%d][%d] = %f\t", i, j, entrada[i][j]);
        }
        printf("\n");
    }

    for(k = 0; k < NEUOCULTA; k++){     // Multiplicação de pesos da oculta pela entrada
        soma = 0;
        for(i = 0; i < NUMENTRADAS; i++){
            soma += pesosOculta[k][i] * entrada[i][0];
            //printf("\nPeso[%d][%d] = %f | entrada[%d][0] = %f\n", k, i, pesosOculta[k][i], i, entrada[i][0]);
        }
        oculta[k][0] = soma;
        oculta[k][0] = sigmoide(oculta[k][0]);
        //printf("oculta = %f\n", oculta[k][0]);
    }

    // SAIDA
    for(k = 0; k < NEUSAIDA; k++){     // Multiplicação de pesos da saida pela oculta
        soma = 0;
        for(i = 0; i < NEUOCULTA; i++){
            soma += pesosSaida[k][i] * oculta[i][0];
            //printf("\pesosSaida[%d][%d] = %f | oculta[%d][0] = %f\n", k, i, pesosSaida[k][i], i, oculta[i][0]);
        }
        saida[k][0] = soma;
        saida[k][0] = sigmoide(saida[k][0]);
    }
    printf("\nSaida Obtida = %f", saida[k][0]);
    printf("\nSaida Correta = 0\n");

}

int main(){

    treinar();
    salvarPesos();
    printf("\nRede treinada com sucesso :)\n");
    printf("\nProcesso de previsao: \n");
    previsao();
    return 0;
}
