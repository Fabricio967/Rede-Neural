/**
    Autor: Fabrício Henrique da Silva
    Aluno de Analise e Desenvolvimento de Sistemas
    Rede Neural Perceptron
    Data: 20/03/2020
    Atualização: 14/04/2020
**/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "fmatriz.h"

const int numEntrada = 2;
const int numNeuronioOculta = 2;
const int numNeuronioSaida= 1;
const float taxaDeAprendizado = 0.1;
const float erroDesejado = 0.1;
const int numeroAmostras = 4;
const int epocas = 50;

void treinar(){

    float *yDesejadoSaida = alocaMatriz(numNeuronioSaida, 1);
    float *matrizDeEntrada;
    float *pesosOculta, *biasOculta, *zOculta, *yOculta;
    float *pesosSaida, *biasSaida, *zSaida, *ySaida;
    float *erroSaida, *gradienteSaida;
    float *erroOculta, *gradienteOculta;
    float erroGlobal;

    int i, j;

    /*for(int i = 0; i < numNeuronioOculta; i++){
        *(yDesejadoOculta+i) = 1;
    }*/
    float amostras[4][2] = { {0, 0}, {0, 1}, {1, 0}, {1, 1} };
    float saidaDesejada[4] = { 0, 1, 1, 0};

    matrizDeEntrada = alocaMatriz(numEntrada, 1);
    int numEpocas = 0;

    pesosOculta = alocaMatriz(numNeuronioOculta, numEntrada);
    biasOculta = preencher(alocaMatriz(numNeuronioOculta, 1), numNeuronioOculta, 1);
    pesosSaida = alocaMatriz(numNeuronioSaida, numNeuronioOculta);
    biasSaida = preencher(alocaMatriz(numNeuronioSaida, numNeuronioSaida), numNeuronioSaida, numNeuronioSaida);

    erroSaida = alocaMatriz(numNeuronioOculta, numNeuronioSaida);
    erroOculta = alocaMatriz(numNeuronioOculta, numEntrada);

    *(pesosOculta+0) = 1;
    *(pesosOculta+1) = 1.5;
    *(pesosOculta+2) = 0.5;
    *(pesosOculta+3) = 2;

    *(pesosSaida+0) = 2;
    *(pesosSaida+1) = 0.5;



    do{ //ser referente as amostras

        for(i = 0; i < numeroAmostras; i++){ //refrente as amostras

            //printf("\n==============================Amostra %d: \n", i+1);

            for(j = 0; j < (numEntrada); j++){ //referente
                *(matrizDeEntrada+j) = amostras[i][j];
                printf("\n======Amostra atual: \t%f", amostras[i][j]);
            }

            //FEEDFORWARD

            //Entrada -> Camada Oculta

            zOculta = produtoMatriz(pesosOculta, numNeuronioOculta, numEntrada, matrizDeEntrada, numEntrada, 1);
            zOculta = somaMatrizes(zOculta, numNeuronioOculta, 1, biasOculta, numNeuronioOculta, 1);
            yOculta = sigmoid(zOculta, numNeuronioOculta);

            //Camada Oculta -> Camada de Saida

            zSaida = produtoMatriz(pesosSaida, numNeuronioSaida, numNeuronioOculta, yOculta, numNeuronioOculta, 1);
            zSaida = somaMatrizes(zSaida, numNeuronioSaida, 1, biasSaida, numNeuronioSaida, numNeuronioSaida);
            ySaida = sigmoid(zSaida, numNeuronioSaida); // = 0.7773

            //Verifica o erro na camada de saída

            printf("\n\nY obtido: \t");
            exibirMatriz(ySaida, numNeuronioSaida, 1);

            erroGlobal = ( *(ySaida) - saidaDesejada[i] );
            printf("\nErro Global = \t%f\n\n", erroGlobal);

            if(fabs(erroGlobal) > erroDesejado){
                //BACKPROPAGATION
                //Processo de atualização dos pesos

                erroSaida = produtoEscalar(yOculta, numNeuronioOculta, numNeuronioSaida, erroGlobal);
                erroSaida = produtoEscalar(yOculta, numNeuronioOculta, numNeuronioSaida, taxaDeAprendizado);
                erroSaida = transporMatriz(erroSaida, numNeuronioOculta, numNeuronioSaida);

                //Saida -> Oculta

                pesosSaida = subtraiMatriz(pesosSaida, numNeuronioSaida, numNeuronioOculta, erroSaida, numNeuronioSaida, numNeuronioOculta);

                //Oculta -> Entrada

                erroOculta = produtoMatriz(matrizDeEntrada, numEntrada, 1, pesosSaida, numNeuronioSaida, numNeuronioOculta);
                erroOculta = produtoEscalar(erroOculta, numEntrada, numNeuronioOculta, erroGlobal);
                erroOculta = produtoEscalar(erroOculta, numEntrada, numNeuronioOculta, taxaDeAprendizado);
                erroOculta = transporMatriz(erroOculta, numEntrada, numNeuronioOculta);
                pesosOculta = subtraiMatriz(pesosOculta, numNeuronioOculta, numEntrada, erroOculta, numNeuronioOculta, numEntrada);
            }

            printf("======Novos Pesos====== \nSaida:\n");
            exibirMatriz(pesosSaida, numNeuronioSaida, numNeuronioOculta);
            printf("\nOculta:\n");
            exibirMatriz(pesosOculta, numNeuronioOculta, numEntrada);
            printf("\n------------------------------\n");
        }


        numEpocas = numEpocas + 1;
        //printf("\nEpocas: %d", numEpocas);
    }while(numEpocas < epocas);

    printf("\nMatriz de entrada: ");
    exibirMatriz(matrizDeEntrada, numEntrada, 1);
    printf("\nMatriz esperada: ");
    exibirMatriz(yDesejadoSaida, numNeuronioSaida, 1);
    printf("\nMatriz obtida: ");
    exibirMatriz(ySaida, numNeuronioSaida, 1);



    printf("\n================================================\n");

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
    //free(erroSaida);

}

int main(){

    srand(time(NULL));

    treinar();

    printf("\nRede Treinada :) ");

    return 1;

}
