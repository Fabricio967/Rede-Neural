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
#include <time.h>

#include "fmatriz.h"

const int numEntrada = 2;
const int numNeuronioOculta = 2;
const int numNeuronioSaida= 1;
const float taxaDeAprendizado = 0.1;
const float erroDesejado = 0.01;
const int numeroAmostras = 4;
const int epocas = 1000;

void treinar(){

    float *yDesejadoSaida = alocaMatriz(numNeuronioSaida, 1);
    float *matrizDeEntrada;
    float *pesosOculta, *biasOculta, *zOculta, *yOculta;
    float *pesosSaida, *biasSaida, *zSaida, *ySaida;
    float *erroSaida, *gradienteSaida;
    float *erroOculta, *gradienteOculta;

    int i, j;

    /*for(int i = 0; i < numNeuronioOculta; i++){
        *(yDesejadoOculta+i) = 1;
    }*/
    float amostras[4][3] = { {0, 0, 0}, {0, 1, 1}, {1, 0, 1}, {1, 1, 0} };

    matrizDeEntrada = alocaMatriz(numEntrada, 1);
    int numEpocas = 0;

    do{

        for(i = 0; i < numeroAmostras; i++){
            printf("i = %d\n", i);
            for(j = 0; j < (numEntrada+1); j++){
                if(j == numEntrada){
                    *(yDesejadoSaida) = amostras[i][j];
                }else{
                    *(matrizDeEntrada+j) = amostras[i][j];
                }

                /*
                printf("\n");
                exibirMatriz(matrizDeEntrada, numEntrada, 1);
                printf("\n=");
                exibirMatriz(yDesejadoSaida, numNeuronioSaida, 1);*/

                //FEEDFORWARD

                //Entrada -> Camada Oculta
                pesosOculta = preencher(alocaMatriz(numNeuronioOculta, numEntrada), numNeuronioOculta, numEntrada);
                zOculta = produtoMatriz(pesosOculta, numNeuronioOculta, numEntrada, matrizDeEntrada, numEntrada, 1);
                biasOculta = preencher(alocaMatriz(numNeuronioOculta, 1), numNeuronioOculta, 1);
                zOculta = somaMatrizes(zOculta, numNeuronioOculta, 1, biasOculta, numNeuronioOculta, 1);
                yOculta = sigmoid(zOculta, numNeuronioOculta);

                //Camada Oculta -> Camada de Saida
                pesosSaida = preencher(alocaMatriz(numNeuronioSaida, numNeuronioOculta), numNeuronioSaida, numNeuronioOculta);
                biasSaida = preencher(alocaMatriz(numNeuronioSaida, numNeuronioSaida), numNeuronioSaida, numNeuronioSaida);
                zSaida = produtoMatriz(pesosSaida, numNeuronioSaida, numNeuronioOculta, yOculta, numNeuronioOculta, 1);
                zSaida = somaMatrizes(zSaida, numNeuronioSaida, 1, biasSaida, numNeuronioSaida, numNeuronioSaida);
                ySaida = sigmoid(zSaida, numNeuronioSaida);
                exibirMatriz(ySaida, numNeuronioSaida, 1);

                //Verifica o erro na camada de saída
                erroSaida = subtraiMatriz(yDesejadoSaida, numNeuronioSaida, 1, ySaida, numNeuronioSaida, 1);
                exibirMatriz(erroSaida, 1, 1);
                //system("pause");

                if(*erroSaida > erroDesejado){
                    //BACKPROPAGATION
                    //Processo de atualização dos pesos

                    //Saida -> Oculta
                    gradienteSaida = produtoMatriz(yOculta, numNeuronioOculta, 1, erroSaida, numNeuronioSaida, 1);
                    gradienteSaida = produtoEscalar(gradienteSaida, numNeuronioOculta, 1, taxaDeAprendizado);
                    gradienteSaida = transporMatriz(gradienteSaida, numNeuronioOculta, 1);
                    pesosSaida = somaMatrizes(pesosSaida, numNeuronioSaida, numNeuronioOculta, gradienteSaida, 1, numNeuronioOculta);

                    //Oculta -> Entrada

                    erroOculta = produtoMatriz(matrizDeEntrada, numEntrada, 1, pesosSaida, 1, numNeuronioOculta);
                    gradienteOculta = transporMatriz(erroOculta, numEntrada, numNeuronioOculta);
                    gradienteOculta = produtoEscalar(gradienteOculta, numNeuronioOculta, numEntrada, *erroSaida);
                    gradienteOculta = produtoEscalar(gradienteOculta, numNeuronioOculta, numEntrada, taxaDeAprendizado);
                    pesosOculta = somaMatrizes(pesosOculta, numNeuronioOculta, numEntrada, gradienteOculta, numNeuronioOculta, numEntrada);


                }
            }
        }



        numEpocas += 1;
        //system("pause");
    }while(numEpocas < epocas || *erroSaida > erroDesejado);


    printf("\nMatriz de entrada:");
    exibirMatriz(matrizDeEntrada, numEntrada, 1);
    printf("\nMatriz esperada");
    exibirMatriz(yDesejadoSaida, numNeuronioSaida, 1);
    printf("\nMatriz obtida:");
    exibirMatriz(ySaida, numNeuronioSaida, 1);



    printf("\n================================================\n");

    /*
    //FEEDFORWARD

    //Entrada -> Camada Oculta
    pesosOculta = preencher(alocaMatriz(numNeuronioOculta, numEntrada), numNeuronioOculta, numEntrada);
    zOculta = produtoMatriz(pesosOculta, numNeuronioOculta, numEntrada, matrizDeEntrada, numEntrada, 1);
    biasOculta = preencher(alocaMatriz(numNeuronioOculta, 1), numNeuronioOculta, 1);
    zOculta = somaMatrizes(zOculta, numNeuronioOculta, 1, biasOculta, numNeuronioOculta, 1);
    yOculta = sigmoid(zOculta, numNeuronioOculta);

    //Camada Oculta -> Camada de Saida
    pesosSaida = preencher(alocaMatriz(numNeuronioSaida, numNeuronioOculta), numNeuronioSaida, numNeuronioOculta);
    biasSaida = preencher(alocaMatriz(numNeuronioSaida, numNeuronioSaida), numNeuronioSaida, numNeuronioSaida);
    zSaida = produtoMatriz(pesosSaida, numNeuronioSaida, numNeuronioOculta, yOculta, numNeuronioOculta, 1);
    zSaida = somaMatrizes(zSaida, numNeuronioSaida, 1, biasSaida, numNeuronioSaida, numNeuronioSaida);
    ySaida = sigmoid(zSaida, numNeuronioSaida);

    //FIM DO FEEDFORWARD
    erroSaida = subtraiMatriz(yDesejadoSaida, numNeuronioSaida, 1, ySaida, numNeuronioSaida, 1);
    //BACKPROPAGATION

    //Saida -> Oculta
    gradienteSaida = produtoMatriz(yOculta, numNeuronioOculta, 1, erroSaida, numNeuronioSaida, 1);
    gradienteSaida = produtoEscalar(gradienteSaida, numNeuronioOculta, 1, taxaDeAprendizado);
    gradienteSaida = transporMatriz(gradienteSaida, numNeuronioOculta, 1);
    pesosSaida = somaMatrizes(pesosSaida, numNeuronioSaida, numNeuronioOculta, gradienteSaida, 1, numNeuronioOculta);

    //Oculta -> Entrada

    erroOculta = produtoMatriz(matrizDeEntrada, numEntrada, 1, pesosSaida, 1, numNeuronioOculta);
    gradienteOculta = transporMatriz(erroOculta, numEntrada, numNeuronioOculta);
    gradienteOculta = produtoEscalar(gradienteOculta, numNeuronioOculta, numEntrada, *erroSaida);
    gradienteOculta = produtoEscalar(gradienteOculta, numNeuronioOculta, numEntrada, taxaDeAprendizado);
    pesosOculta = somaMatrizes(pesosOculta, numNeuronioOculta, numEntrada, gradienteOculta, numNeuronioOculta, numEntrada);
    */

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

    printf("\nRede Treinada :) ");

    return 1;

}
