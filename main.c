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

    float *y_d_saida = alocaMatriz(numSaida, 1);
    float *y_d_oculta = alocaMatriz(numNeuronio, 1);

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

    float *uOculta = produtoMatriz(pesosOculta, numNeuronio, numEntrada, entradas, numEntrada, 1);
    float *biasOculta = preencher(alocaMatriz(numNeuronio, 1), numNeuronio, 1);

    //system("pause");

    uOculta = somaMatrizes(uOculta, numNeuronio, 1, biasOculta, numNeuronio, 1);

    float *yOculta = sigmoid(uOculta, numNeuronio);

    //Camada Oculta -> Camada de Saida
    float *camadaSaida = yOculta;

    float *pesosSaida = preencher(alocaMatriz(numSaida, numNeuronio), numSaida, numNeuronio);
    float *biasSaida = preencher(alocaMatriz(numSaida, numSaida), numSaida, numSaida);
    float *uSaida = produtoMatriz(pesosSaida, numSaida, numNeuronio, camadaSaida, numNeuronio, 1);

    uSaida = somaMatrizes(uSaida, numSaida, 1, biasSaida, numSaida, numSaida);
    float *ySaida = sigmoid(uSaida, numSaida);

    //BACKPROPAGATION

    //Saida -> Oculta

    float *erro_saida = subtraiMatriz(y_d_saida, numSaida, 1, ySaida, numSaida, 1);

/*
    if(*(erro_saida) > Erro_desejado){
        printf("\n\t\ttrue\n");
    }*/

    float *gradiente_os = produtoMatriz(camadaSaida, numNeuronio, 1, erro_saida, numSaida, 1);
    gradiente_os = produtoEscalar(gradiente_os, numNeuronio, 1, taxa_aprendizado);
    gradiente_os = transporMatriz(gradiente_os, numNeuronio, 1);
    pesosSaida = somaMatrizes(pesosSaida, numSaida, numNeuronio, gradiente_os, 1, numNeuronio);

    //Oculta -> Entrada

    float *erro_oculta = produtoMatriz(entradas, numEntrada, 1, pesosSaida, 1, numNeuronio);

    //system("pause");

    float *gradiente_eo = transporMatriz(erro_oculta, numEntrada, numNeuronio);
    gradiente_eo = produtoEscalar(gradiente_eo, numNeuronio, numEntrada, *erro_saida);
    gradiente_eo = produtoEscalar(gradiente_eo, numNeuronio, numEntrada, taxa_aprendizado);
    pesosOculta = somaMatrizes(pesosOculta, numNeuronio, numEntrada, gradiente_eo, numNeuronio, numEntrada);

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
