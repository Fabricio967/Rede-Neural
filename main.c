/**
    Autor: Fabrício Henrique da Silva
    Aluno de Analise e Desenvolvimento de Sistemas
    Rede Neural Perceptron
    Data: 20/03/2020
    Atualização: 05/04/2020
**/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "fmatriz.h"


const int numEntrada = 3;
const int numNeuronioOculta = 2;
const int numNeuronioSaida= 1;
const float taxaDeAprendizado = 0.1;
const float erroDesejado = 0.01;


void treinar(){

    float *yDesejadoSaida = alocaMatriz(numNeuronioSaida, 1);
    float *yDesejadoOculta = alocaMatriz(numNeuronioOculta, 1);

    for(int i = 0; i < numNeuronioSaida; i++){
        *(yDesejadoSaida+i) = 1;
    }

    for(int i = 0; i < numNeuronioOculta; i++){
        *(yDesejadoOculta+i) = 1;
    }

    //FEEDFORWARD

    //Entrada -> Camada Oculta
    float *pesosOculta = preencher(alocaMatriz(numNeuronioOculta, numEntrada), numNeuronioOculta, numEntrada);
    float *matrizDeEntrada = alocaMatriz(numEntrada, 1);

    for(int i = 0; i < numEntrada; i++){
        if(i % 2 == 0){
            *(matrizDeEntrada+i) = 1;
        }else{
            *(matrizDeEntrada+i) = 1;
        }
    }

    float *zOculta = produtoMatriz(pesosOculta, numNeuronioOculta, numEntrada, matrizDeEntrada, numEntrada, 1);
    float *biasOculta = preencher(alocaMatriz(numNeuronioOculta, 1), numNeuronioOculta, 1);

    //system("pause");

    zOculta = somaMatrizes(zOculta, numNeuronioOculta, 1, biasOculta, numNeuronioOculta, 1);

    float *yOculta = sigmoid(zOculta, numNeuronioOculta);

    //Camada Oculta -> Camada de Saida

    float *pesosSaida = preencher(alocaMatriz(numNeuronioSaida, numNeuronioOculta), numNeuronioSaida, numNeuronioOculta);
    float *biasSaida = preencher(alocaMatriz(numNeuronioSaida, numNeuronioSaida), numNeuronioSaida, numNeuronioSaida);
    float *zSaida = produtoMatriz(pesosSaida, numNeuronioSaida, numNeuronioOculta, yOculta, numNeuronioOculta, 1);

    zSaida = somaMatrizes(zSaida, numNeuronioSaida, 1, biasSaida, numNeuronioSaida, numNeuronioSaida);
    float *ySaida = sigmoid(zSaida, numNeuronioSaida);

    //BACKPROPAGATION

    //Saida -> Oculta

    float *erroSaida = subtraiMatriz(yDesejadoSaida, numNeuronioSaida, 1, ySaida, numNeuronioSaida, 1);

/*
    if(*(erroSaida) > erroDesejado){
        printf("\n\t\ttrue\n");
    }*/

    float *gradienteSaida = produtoMatriz(yOculta, numNeuronioOculta, 1, erroSaida, numNeuronioSaida, 1);
    gradienteSaida = produtoEscalar(gradienteSaida, numNeuronioOculta, 1, taxaDeAprendizado);
    gradienteSaida = transporMatriz(gradienteSaida, numNeuronioOculta, 1);
    pesosSaida = somaMatrizes(pesosSaida, numNeuronioSaida, numNeuronioOculta, gradienteSaida, 1, numNeuronioOculta);

    //Oculta -> Entrada

    float *erroOculta = produtoMatriz(matrizDeEntrada, numEntrada, 1, pesosSaida, 1, numNeuronioOculta);

    //system("pause");

    float *gradienteOculta = transporMatriz(erroOculta, numEntrada, numNeuronioOculta);
    gradienteOculta = produtoEscalar(gradienteOculta, numNeuronioOculta, numEntrada, *erroSaida);
    gradienteOculta = produtoEscalar(gradienteOculta, numNeuronioOculta, numEntrada, taxaDeAprendizado);
    pesosOculta = somaMatrizes(pesosOculta, numNeuronioOculta, numEntrada, gradienteOculta, numNeuronioOculta, numEntrada);

    free(pesosOculta);
    free(biasOculta);
    free(matrizDeEntrada);
    free(zOculta);
    free(yOculta);
    free(pesosSaida);
    free(biasSaida);
    free(zSaida);
    free(ySaida);
    free(gradienteOculta);
    free(gradienteSaida);
    free(erroOculta);
    free(erroSaida);

}

int main(){

    srand(time(NULL));

    treinar();

    return 1;

}
