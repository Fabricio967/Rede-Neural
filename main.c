/**
    Autor: Fabrício Henrique da Silva
    Aluno de Analise e Desenvolvimento de Sistemas
    Rede Neural Perceptron
    Data: 20/03/2020
    Atualização: 18/04/2020
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
const long int epocas = 10000;

float rr(){
    float f = (rand()%10);
    f /= 10;
    if(f != 0){
        //printf("%f\n", f);
        return f;
    }else{
        rr();
    }
}

void treinar(){

    //float *yDesejadoSaida = alocaMatriz(numNeuronioSaida, 1);
    float *matrizDeEntrada;
    float *pesosOculta, *biasOculta, *yOculta;
    float *pesosSaida, *biasSaida, *ySaida;
    float *erroSaida, *pesosSaidaTransposta;
    float *erroOculta, *entradaTransposta;
    float *erroGlobal;

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

    //erroSaida = alocaMatriz(numNeuronioOculta, numNeuronioSaida);
    //erroOculta = alocaMatriz(numNeuronioOculta, numEntrada);

    srand(time(NULL));

    *(pesosOculta+0) = rr();
    *(pesosOculta+1) = rr();
    *(pesosOculta+2) = rr();
    *(pesosOculta+3) = rr();

    *(pesosSaida+0) = rr();
    *(pesosSaida+1) = rr();

    float erro = 1;

    do{ //ser referente as amostras

        for(i = 0; i < numeroAmostras; i++){ //refrente as amostras

            //printf("\n==============================Amostra %d: \n", i+1);

            for(j = 0; j < (numEntrada); j++){
                *(matrizDeEntrada+j) = amostras[i][j];
                //printf("\n======Amostra atual: \t%f", amostras[i][j]);
            }

            //FEEDFORWARD

            //Entrada -> Camada Oculta

            yOculta = produtoMatriz(pesosOculta, numNeuronioOculta, numEntrada, matrizDeEntrada, numEntrada, 1);
            yOculta = somaMatrizes(yOculta, numNeuronioOculta, 1, biasOculta, numNeuronioOculta, 1);
            yOculta = sigmoid(yOculta, numNeuronioOculta);

            //Camada Oculta -> Camada de Saida

            ySaida = produtoMatriz(pesosSaida, numNeuronioSaida, numNeuronioOculta, yOculta, numNeuronioOculta, 1);
            ySaida = somaMatrizes(ySaida, numNeuronioSaida, 1, biasSaida, numNeuronioSaida, numNeuronioSaida);
            ySaida = sigmoid(ySaida, numNeuronioSaida); // = 0.7773

            //printf("\n\nY obtido: \t");
            //exibirMatriz(ySaida, numNeuronioSaida, 1);

            erro = ( saidaDesejada[i] - *(ySaida) );
            erroGlobal = produtoEscalar(d_sigmoid(ySaida, numNeuronioSaida), numNeuronioSaida, numNeuronioSaida, erro);
            //printf("\nErro Global = \t%f\n\n", erroGlobal);
            //system("pause");

            if( fabs(erro) > erroDesejado ){
                //BACKPROPAGATION
                //Processo de atualização dos pesos

                //Saida -> Oculta

                erroSaida = transporMatriz(yOculta, numNeuronioOculta, numNeuronioSaida);
                erroSaida = produtoEscalar(erroSaida, numNeuronioSaida, numNeuronioOculta, *erroGlobal);
                erroSaida = produtoEscalar(erroSaida, numNeuronioSaida, numNeuronioOculta, taxaDeAprendizado);
                pesosSaida = somaMatrizes(pesosSaida, numNeuronioSaida, numNeuronioOculta, erroSaida, numNeuronioSaida, numNeuronioOculta);

                biasSaida = somaMatrizes(biasSaida, numNeuronioSaida, numNeuronioSaida, erroGlobal, numNeuronioSaida, numNeuronioSaida);

                //Oculta -> Entrada

                pesosSaidaTransposta = transporMatriz(pesosSaida, numNeuronioSaida, numNeuronioOculta);
                entradaTransposta = transporMatriz(matrizDeEntrada, numEntrada, 1);
                erroOculta = produtoEscalar(pesosSaidaTransposta, numNeuronioOculta, numNeuronioSaida, erro);

                erroOculta = hadamard(erroOculta, numNeuronioOculta, numNeuronioSaida, d_sigmoid(yOculta, numNeuronioOculta), numNeuronioOculta, numNeuronioSaida);
                biasOculta = somaMatrizes(biasOculta, numNeuronioOculta, numNeuronioSaida, erroOculta, numNeuronioOculta, numNeuronioSaida);
                erroOculta = produtoMatriz(erroOculta, numNeuronioOculta, numNeuronioSaida, entradaTransposta, 1, numEntrada);
                erroOculta = produtoEscalar(erroOculta, numEntrada, numNeuronioOculta, taxaDeAprendizado);
                pesosOculta = somaMatrizes(pesosOculta, numNeuronioOculta, numEntrada, erroOculta, numNeuronioOculta, numEntrada);


            }

        /*
            printf("======Novos Pesos====== \nSaida:\n");
            exibirMatriz(pesosSaida, numNeuronioSaida, numNeuronioOculta);
            printf("\nOculta:\n");
            exibirMatriz(pesosOculta, numNeuronioOculta, numEntrada);
            printf("\n------------------------------\n");
            */

            if( /*fabs(erro) < erroDesejado */ numEpocas == (epocas - 1)){
                printf("\nMatriz de entrada:\n");
                exibirMatriz(matrizDeEntrada, numEntrada, 1);
                printf("\nSaida Desejada = \t%f", saidaDesejada[i]);
                printf("\nMatriz obtida = \t");
                exibirMatriz(ySaida, numNeuronioSaida, 1);
                printf("\n-----------------------------\n");
            }
        }


        numEpocas = numEpocas + 1;
        //printf("\nEpocas: %d", numEpocas);
    }while(numEpocas < epocas);

    /*
    printf("\nMatriz de entrada: ");
    exibirMatriz(matrizDeEntrada, numEntrada, 1);
    printf("\nMatriz esperada: ");
    exibirMatriz(yDesejadoSaida, numNeuronioSaida, 1);
    printf("\nMatriz obtida: ");
    exibirMatriz(ySaida, numNeuronioSaida, 1); */



    printf("\n================================================\n");

    free(pesosOculta);
    free(biasOculta);
    free(matrizDeEntrada);
    free(yOculta);
    free(yOculta);
    free(pesosSaida);
    free(biasSaida);
    free(ySaida);
    free(ySaida);
    free(pesosSaidaTransposta);
    free(entradaTransposta);
    free(erroOculta);
    //free(erroSaida);

}

int main(){

    srand(time(NULL));

    treinar();

    printf("\nRede Treinada :) ");

    return 1;

}
