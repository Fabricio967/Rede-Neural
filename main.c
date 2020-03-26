/**
    Autor: Fabrício Henrique da Silva
    Aluno de Analise e Desenvolvimento de Sistemas
    Rede Neural Perceptron
    Data: 20/03/2020
    Atualização: 24/03/2020

**/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void exibirMatriz(int nlin, int ncol, int *ptr_matriz);
int *alocaMatriz(int nlin, int ncol);
float sigmoid(int *matriz, int nlin);
float randomico(void);
int *produtoMatriz(int *mat1, int nlin_mat1, int ncol_mat1, int *mat2, int nlin_mat2, int ncol_mat2);
int *preencher(int *matriz, int nlin, int ncol);

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



    return ((float)(rand()/(float)RAND_MAX))*10;

}

int *produtoMatriz(int *mat1, int nlin_mat1, int ncol_mat1, int *mat2, int nlin_mat2, int ncol_mat2){

    int *prod_mat, i, j, soma = 0;

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

int *alocaMatriz(int nlin, int ncol){

    int i, *ponteiro, *q;

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

int *preencher(int *matriz, int nlin, int ncol){

    for(int i = 0; i < (nlin*ncol); i++){
        *(matriz+i) = rand()%10;
    }

    return matriz;
}

void treinar(){

    int yd[] = {0, 1, 1, 0};    //valores desejados
    int neuronio = 3, entrada = 2;
    float erro = 0.001;
    float *pesos, *v;
    int j;

    int amostra1[] = {0, 0};
    int amostra2[] = {0, 1};
    int amostra3[] = {1, 0};
    int amostra4[] = {1, 1};
    int amostra;

    int amostras[] = {0, 0, 1, 0, 0, 1, 1, 1};

    pesos = alocaMatriz(neuronio, entrada);
    pesos = preencher(pesos, neuronio, entrada);

    printf("\nMatriz de pesos: \n");
    exibirMatriz(neuronio, entrada, pesos);
    printf("\n");

    int r = 2*(rand()%4);

    printf("\nMatriz de entrada: \nr = %d\n", r);
    for(int i = r; i < r+2; i++){
        amostra[j] = amostras[i];
        printf("%d\n", amostra[i]);

    }
    printf("\n");

    v = produtoMatriz(pesos, neuronio, entrada, &amostra, entrada, 1);
    printf("\nMatriz resultado: \n");
    exibirMatriz(neuronio, 1, v);




}

int main(){

    srand((unsigned int)time(NULL));

    float yo;
    int num_entradas = 2, num_neuronios = 3, num_saida = 1;
    int *weigth, *x_input, *u;

    treinar();
    exit(1);

    weigth = alocaMatriz(num_neuronios, num_entradas);
    x_input = alocaMatriz(num_entradas, 1);

    weigth = preencher(weigth, num_neuronios, num_entradas);


    exibirMatriz(num_neuronios, num_entradas, weigth);
    printf("\n");
    exibirMatriz(num_entradas, 1, x_input);
    printf("\n");

    u = produtoMatriz(weigth, num_neuronios, num_entradas, x_input, num_entradas, 1);
    exibirMatriz(num_neuronios, 1, u);

    printf("\n");

    yo = sigmoid(u, num_neuronios);
    printf("\nSaida: %f\n", yo);

    free(weigth);
    free(x_input);

    return 0;
}
