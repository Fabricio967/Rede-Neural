/**
    Autor: Fabrício Henrique da Silva
    Aluno de Analise e Desenvolvimento de Sistemas
    Rede Neural Perceptron
    Data: 20/03/2020
    Atualização: 26/03/2020

**/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void exibirMatriz(float *ptr_matriz, int nlin, int ncol);
float *alocaMatriz(int nlin, int ncol);
float *sigmoid(float *matriz, int nlin);
float *produtoMatriz(float *mat1, int nlin_mat1, int ncol_mat1, float *mat2, int nlin_mat2, int ncol_mat2);
float *preencher(float *matriz, int nlin, int ncol);
float *somaMatrizes(float *mat1, int nlin_mat1, int ncol_mat1, float *mat2, int nlin_mat2, int ncol_mat2);
void treinar();

const int numEntrada = 3;
const int numNeuronio = 2;
const int numSaida= 1;

float *sigmoid(float *matriz, int nlin){

    float *sig = alocaMatriz(nlin, 1);
    float soma;

    for(int i = 0; i < nlin; i++){
        soma = (1/(1+exp(- *(matriz + i) )));
        *(sig+i) = soma;
        printf("\nSigmoid:\t%f\n", *(sig+i));
    }
    return sig;
}

float *produtoMatriz(float *mat1, int nlin_mat1, int ncol_mat1, float *mat2, int nlin_mat2, int ncol_mat2){

    int i, j, soma = 0;
    float *prod_mat;

    prod_mat = alocaMatriz(nlin_mat1, ncol_mat2);

    if(ncol_mat1 == nlin_mat2){

        for(i = 0; i < nlin_mat1; i++){
            soma = 0;
            for(j = 0; j < nlin_mat2; j++){
                soma = soma + (*(mat1 + (i * ncol_mat1) + j) * *(mat2 + j));
            }
            *(prod_mat+i) = soma;
        }
    }
    return prod_mat;
}

float *alocaMatriz(int nlin, int ncol){

    int i;
    float *ponteiro, *q;

    if(nlin < 1 || ncol < 1)
        printf("\n2. ERRO: Parametro de tamanho invalido.\n");

    for(i = 0; i < (nlin*ncol); i++)
        ponteiro = (float*)malloc(nlin*ncol*sizeof(float));

    if(ponteiro == NULL)
        printf("\n2.1. ERRO: Memoria Insuficiente!");

    q = ponteiro;

    for (i = 0; i < (nlin*ncol); i++, q++)
		*q = 0;

    return ponteiro;

}

void exibirMatriz(float *ptr_matriz, int nlin, int ncol){

    int i, j;

    for(i = 0; i < nlin; i++){
        for(j = 0; j < ncol; j++){
            printf("%.2f\t", *(ptr_matriz + (ncol * i) + j));
        }
        printf("\n");
    }
}

float *preencher(float *matriz, int nlin, int ncol){

    for(int i = 0; i < (nlin*ncol); i++){
        *(matriz+i) = rand()%10;
    }

    return matriz;
}

float *somaMatrizes(float *mat1, int nlin_mat1, int ncol_mat1, float *mat2, int nlin_mat2, int ncol_mat2){

    int i;
    float *soma_mat;

    soma_mat = alocaMatriz(nlin_mat1, ncol_mat2);

    if(nlin_mat1 == nlin_mat2 && ncol_mat1 == ncol_mat2){
        for(i = 0; i < (nlin_mat1*ncol_mat1); i++){
            *(soma_mat+i) = ( *(mat1+i) + *(mat2+i) );
            //printf("\nveredito\n");
        }
    }
    return soma_mat;
}

void treinar(){

    int yd[] = {0, 1, 1, 0};    //saida desejados
    float erro = 0.001;         //erro desejado
    int j;

    int amostra1[] = {0, 0};
    int amostra2[] = {0, 1};
    int amostra3[] = {1, 0};
    int amostra4[] = {1, 1};
    int amostra;

    int amostras[] = {0, 0, 1, 0, 0, 1, 1, 1};

    //Entrada -> Camada Oculta
    float *camadaOculta = preencher(alocaMatriz(numNeuronio, numEntrada), numNeuronio, numEntrada);
    float *entradas = alocaMatriz(numEntrada, 1);

    for(int i = 0; i < numEntrada; i++){
        if(i % 2 == 0){
            *(entradas+i) = 1.0f;
        }else{
            *(entradas+i) = 0;
        }
    }

    printf("\nMatriz de pesos:\n");
    exibirMatriz(camadaOculta, numNeuronio, numEntrada);

    printf("\nMatriz de entrada:\n");
    exibirMatriz(entradas, numEntrada, 1);

    float *u = produtoMatriz(camadaOculta, numNeuronio, numEntrada, entradas, numEntrada, 1);
    float *bias = preencher(alocaMatriz(numNeuronio, 1), numNeuronio, 1);

    printf("\nMatriz bias:\n");
    exibirMatriz(bias, numNeuronio, 1);

    printf("\nMatriz u:\n");
    exibirMatriz(u, numNeuronio, 1);

    u = somaMatrizes(u, numNeuronio, 1, bias, numNeuronio, 1);
    //u = sigmoid(u, numNeuronio);

    printf("\nMatriz u:\n");
    exibirMatriz(u, numNeuronio, 1);

    float *y = sigmoid(u, numNeuronio);

    printf("\nY final\n");
    //exibirMatriz(y, numNeuronio, 1);

    for(int i = 0; i < numNeuronio; i++){
        for(j = 0; j < 1; j++){
            printf("%f\t", *(y + (1 * i) + j));
        }
        printf("\n");
    }

    //Camada Oculta -> Camada de Saida
    float *camadaSaida = alocaMatriz(numSaida, numNeuronio);
    printf("\nMatriz de : \n");
    exibirMatriz(camadaSaida, numSaida, numNeuronio);
    printf("\n");



    /*
    int r = 2*(rand()%4);

    printf("\nMatriz de entrada: \nr = %d\n", r);

    for(int i = r; i < r+2; i++){
        amostra[j] = amostras[i];
        printf("%d\n", amostra[i]);

    }*/

    printf("\n");

    //v = produtoMatriz(pesos, neuronio, entrada, &amostra, entrada, 1);
    //printf("\nMatriz resultado: \n");
    //exibirMatriz(numNeuronio, 1, v);
    free(camadaOculta);
    free(entradas);
    free(u);
    free(y);

}

int main(){

    srand(time(NULL));

    treinar();

    return 1;

}
