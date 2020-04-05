/**
    Autor: Fabrício Henrique da Silva
    Aluno de Analise e Desenvolvimento de Sistemas
    Rede Neural Perceptron
    Data: 20/03/2020
    Atualização: 28/03/2020
**/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "fmatriz.h"


const int numEntrada = 3;
const int numNeuronio = 2;
const int numSaida= 1;
const float taxa_aprendizado = 0.1;
const float Erro_desejado = 0.01;


void treinar(){

    //int yd[] = {0, 1, 1, 0};    //saida desejados
    float *y_d_saida = alocaMatriz(numSaida, 1);
    float *y_d_oculta = alocaMatriz(numNeuronio, 1);
    //int pesos[][] =

    for(int i = 0; i < numSaida; i++){
        *(y_d_saida+i) = 1;
    }

    for(int i = 0; i < numNeuronio; i++){
        *(y_d_oculta+i) = 1;
    }

    //FEEDFORWARD

    //Entrada -> Camada Oculta
    float *pesosOculta = preencher(alocaMatriz(numNeuronio, numEntrada), numNeuronio, numEntrada);
    float *entradas = alocaMatriz(numEntrada, 1);

    for(int i = 0; i < numEntrada; i++){
        if(i % 2 == 0){
            *(entradas+i) = 1;
        }else{
            *(entradas+i) = 1;
        }
    }

    printf("\nMatriz de pesos:\n");
    exibirMatriz(pesosOculta, numNeuronio, numEntrada);

    printf("\nMatriz de entrada:\n");
    exibirMatriz(entradas, numEntrada, 1);

    float *uOculta = produtoMatriz(pesosOculta, numNeuronio, numEntrada, entradas, numEntrada, 1);
    float *biasOculta = preencher(alocaMatriz(numNeuronio, 1), numNeuronio, 1);

    system("pause");

    printf("\nMatriz biasOculta:\n");
    exibirMatriz(biasOculta, numNeuronio, 1);

    uOculta = somaMatrizes(uOculta, numNeuronio, 1, biasOculta, numNeuronio, 1);

    printf("\nMatriz uOculta:\n");
    exibirMatriz(uOculta, numNeuronio, 1);

    float *yOculta = sigmoid(uOculta, numNeuronio);

    printf("\nY oculta\n");
    exibirMatriz(yOculta, numNeuronio, 1);

    printf("\n-------------------------------------------------------------\n");

    //Camada Oculta -> Camada de Saida
    float *camadaSaida = yOculta;
    printf("\nMatriz de entrada para neuronio de Saida: \n");
    exibirMatriz(camadaSaida, numNeuronio, numSaida);

    float *pesosSaida = preencher(alocaMatriz(numSaida, numNeuronio), numSaida, numNeuronio);
    float *biasSaida = preencher(alocaMatriz(numSaida, numSaida), numSaida, numSaida);
    float *uSaida = produtoMatriz(pesosSaida, numSaida, numNeuronio, camadaSaida, numNeuronio, 1);

    printf("\nMatriz uSaida:\n");
    exibirMatriz(uSaida, numSaida, 1);

    uSaida = somaMatrizes(uSaida, numSaida, 1, biasSaida, numSaida, numSaida);
    float *ySaida = sigmoid(uSaida, numSaida);

    printf("\nMatriz de pesos de saida:\n");
    exibirMatriz(pesosSaida, numSaida, numNeuronio);

    printf("\nMatriz biasSaida:\n");
    exibirMatriz(biasSaida, numSaida, 1);

    printf("\nMatriz uSaida:\n");
    exibirMatriz(uSaida, numSaida, 1);

    printf("\nY Saida\n");
    exibirMatriz(ySaida, numSaida, 1);

    printf("\n");

    //BACKPROPAGATION

    //Saida -> Oculta

    float *erro_saida = subtraiMatriz(y_d_saida, numSaida, 1, ySaida, numSaida, 1);
    printf("\nErro na saida\n");
    exibirMatriz(erro_saida, numSaida, 1);

    if(*(erro_saida) > Erro_desejado){
        printf("\n\t\ttrue\n");
    }

    float *gradiente_os = produtoMatriz(camadaSaida, numNeuronio, 1, erro_saida, numSaida, 1);
    printf("\ncamada gradiente_os\n");
    gradiente_os = produtoEscalar(gradiente_os, numNeuronio, 1, taxa_aprendizado);
    gradiente_os = transporMatriz(gradiente_os, numNeuronio, 1);
    printf("\n");
    exibirMatriz(gradiente_os, 1, numNeuronio);
    //gradiente_os = transporMatriz(gradiente_os, numNeuronio, 1);
    pesosSaida = somaMatrizes(pesosSaida, numSaida, numNeuronio, gradiente_os, 1, numNeuronio);
    printf("\n");
    exibirMatriz(pesosSaida, numSaida, numNeuronio);

    //Oculta -> Entrada
    printf("\n------------------------------\n");
    float *erro_oculta = produtoMatriz(entradas, numEntrada, 1, pesosSaida, 1, numNeuronio);
    exibirMatriz(erro_oculta, 3, 2);
    system("pause");

    float *gradiente_eo = transporMatriz(erro_oculta, numEntrada, numNeuronio);
    gradiente_eo = produtoEscalar(gradiente_eo, numNeuronio, numEntrada, *erro_saida);
    gradiente_eo = produtoEscalar(gradiente_eo, numNeuronio, numEntrada, taxa_aprendizado);
    pesosOculta = somaMatrizes(pesosOculta, numNeuronio, numEntrada, gradiente_eo, numNeuronio, numEntrada);

    printf("\ncamada gradiente_eo\n");
    exibirMatriz(gradiente_eo, numNeuronio, numEntrada);
    printf("\n");
    printf("\n");
    exibirMatriz(pesosOculta, numNeuronio, numEntrada);

    free(pesosOculta);
    free(biasOculta);
    free(entradas);
    free(uOculta);
    free(yOculta);
    free(pesosSaida);
    free(biasSaida);
    free(uSaida);
    free(yOculta);

}

int main(){

    srand(time(NULL));

    treinar();

    return 1;

}
