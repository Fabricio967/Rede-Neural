/**
    Autor: Fabrício Henrique da Silva
    Aluno de Analise e Desenvolvimento de Sistemas
    Rede Neural Perceptron
    Data: 20/03/2020
    Atualização: 26/03/2020

    testando Git

**/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void exibirMatriz(int nlin, int ncol, int *ptr_matriz);
int *alocaMatriz(int num_entradas, int num_neuronios);
float sigmoid(int *matriz, int nlin);
float randomico(void);
int *produtoMatriz(int *mat1, int lin_mat1, int col_mat1, int *mat2, int lin_mat2, int col_mat2);

float sigmoid(int *matriz, int nlin){

    int i;
    float soma = 0, sig;

    for(i = 0; i < nlin; i++){
        sig = (1/(1+exp(- *(matriz + i) )));
        soma = soma + sig;
        printf("\nSig: %f", soma);
    }

    return soma;
}

float randomico(void){

    srand((unsigned int)time(0));

    return ((float)(rand()/(float)RAND_MAX))*10;

}

int *produtoMatriz(int *mat1, int lin_mat1, int col_mat1, int *mat2, int lin_mat2, int col_mat2){

    int *prod_mat, i, j, soma = 0;

    prod_mat = alocaMatriz(lin_mat1, col_mat2);

    if(col_mat1 == lin_mat2){

        for(i = 0; i < lin_mat1; i++){
            soma = 0;
            for(j = 0; j < lin_mat2; j++){
                soma = soma + (*(mat1 + (i * col_mat1) + j) * *(mat2 + j));
            }
            *(prod_mat+i) = soma;
        }
    }
    return prod_mat;
}

int *alocaMatriz(int num_entradas, int num_neuronios){

    int i, *ponteiro, *q, nlin, ncol;
    nlin = num_entradas;
    ncol = num_neuronios;

    if(nlin < 1 || ncol < 1)
        printf("\n2. ERRO: Parametro de tamanho invalido.\n");

    for(i = 0; i < (nlin*ncol); i++)
        ponteiro = (int)malloc(nlin*ncol*sizeof(int));

    if(ponteiro == NULL)
        printf("\n2.1. ERRO: Memoria Insuficiente!");

    q = ponteiro;

    for (i = 0; i < (nlin*ncol); i++, q++)
		*q = 0;

    return ponteiro;

}

void exibirMatriz(int nlin, int ncol, int *ptr_matriz){

    int i, j;

    for(i = 0; i < nlin; i++){
        for(j = 0; j < ncol; j++){
            printf("%d\t", *(ptr_matriz + (ncol * i) + j));
        }
        printf("\n");
    }
}

int main(){

    char url1[] = "matrizpesos.txt";
    char url2[] = "matrizentrada.txt";
    FILE *arq, *input;
    float yk;
    int i, entrada = 5, neuronio = 4;  //entrada = dado de entrada, neuronio = numero de neuronios na rede neural
    int *weigth, *x_input, *u;

    weigth = alocaMatriz(neuronio, entrada);
    x_input = alocaMatriz(entrada, 1);

    arq = fopen(url1, "r");
    input = fopen(url2, "r");

    if(arq == NULL){
        printf("\n1. ERRO ao abrir arquivos!\n");
    }
    else{
        printf("\n1. Arquivos aberto com sucesso!\n\n");
    }

    for(i = 0; i < (entrada*neuronio); i++)
        fscanf(arq, "%d", (weigth+i));

    for(i = 0; i < entrada; i++)
        fscanf(input, "%d", (x_input+i));

    exibirMatriz(neuronio, entrada, weigth);
    printf("\n");
    exibirMatriz(entrada, 1, x_input);
    printf("\n");

    u = produtoMatriz(weigth, neuronio, entrada, x_input, entrada, 1);
    exibirMatriz(neuronio, 1, u);

    printf("\n");

    yk = sigmoid(u, neuronio);
    printf("\nSaida: %f\n", yk);

    fclose(arq);
    fclose(x_input);
    free(weigth);
    free(x_input);

    return 0;
}
