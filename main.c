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

void exibirMatriz(float *ptr_matriz, int nlin, int ncol);
float *alocaMatriz(int nlin, int ncol);
float *sigmoid(float *matriz, int nlin);
float *produtoMatriz(float *mat1, int nlin_mat1, int ncol_mat1, float *mat2, int nlin_mat2, int ncol_mat2);
float *preencher(float *matriz, int nlin, int ncol);
float *somaMatrizes(float *mat1, int nlin_mat1, int ncol_mat1, float *mat2, int nlin_mat2, int ncol_mat2);
float *subtraiMatriz(float *mat1, int nlin_mat1, int ncol_mat1, float *mat2, int nlin_mat2, int ncol_mat2);
void treinar();
float *d_sigmoid(float *matriz, int nlin);


const int numEntrada = 3;
const int numNeuronio = 2;
const int numSaida= 1;
const float taxa_aprendizado = 0.1;
const float Erro_desejado = 0.01;

float *sigmoid(float *matriz, int nlin){

    float *sig = alocaMatriz(nlin, 1);
    float soma;

    for(int i = 0; i < nlin; i++){
        soma = (1/(1+exp(- *(matriz + i) )));
        *(sig+i) = soma;
        //printf("\nSigmoid:\t%f\n", *(sig+i));
    }
    return sig;
}

float *produtoMatriz(float *mat1, int nlin_mat1, int ncol_mat1, float *mat2, int nlin_mat2, int ncol_mat2){

    int i, j;
    float *prod_mat, soma;

    prod_mat = alocaMatriz(nlin_mat1, ncol_mat2);

    if(ncol_mat1 == nlin_mat2){

        for(i = 0; i < nlin_mat1; i++){
            soma = 0.f;
            for(j = 0; j < nlin_mat2; j++){
                soma = soma + (*(mat1 + (i * ncol_mat1) + j) * *(mat2 + j));
            }
            *(prod_mat+i) = soma;
            //printf("\n---------------------------%f", *(prod_mat+i));
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
            printf("%f\t", *(ptr_matriz + (ncol * i) + j));
        }
        printf("\n");
    }
}

float *preencher(float *matriz, int nlin, int ncol){

    for(int i = 0; i < (nlin*ncol); i++){
        *(matriz+i) = rand()%10 - 5;
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

float *d_sigmoid(float *matriz, int nlin){

    float *d_sig = alocaMatriz(nlin, 1);
    //float soma;

    for(int i = 0; i < nlin; i++){
        *(d_sig) = *(matriz + i) * (1 - *(matriz + i) );
        //printf("\nSigmoid:\t%f\n", *(sig+i));
    }
    return d_sig;

}

float *subtraiMatriz(float *mat1, int nlin_mat1, int ncol_mat1, float *mat2, int nlin_mat2, int ncol_mat2){

    float *sub_mat;

    sub_mat = alocaMatriz(nlin_mat1, ncol_mat2);

    if(nlin_mat1 == nlin_mat2 && ncol_mat1 == ncol_mat2){
        for(int i = 0; i < (nlin_mat1*ncol_mat1); i++){
            *(sub_mat+i) = ( *(mat1+i) - *(mat2+i) );
            //printf("\nveredito\n");
        }
    }
    return sub_mat;

}

void treinar(){

    //int yd[] = {0, 1, 1, 0};    //saida desejados
    int j;
    float *y_d_saida = alocaMatriz(numSaida, 1);
    float *y_d_oculta = alocaMatriz(numNeuronio, 1);
    float *erro_o_saida = alocaMatriz(numSaida, 1);
    //float erro_o_oculta =

    for(int i = 0; i < numSaida; i++){
        *(y_d_saida+i) = 1;
        printf("\n\t\t\t\tyd_saida = %f", *(y_d_saida+i));
    }

    for(int i = 0; i < numNeuronio; i++){
        *(y_d_oculta+i) = 1;
        printf("\n\t\t\t\tyd_oculta = %f", *(y_d_oculta+i));
    }


    int amostra1[] = {0, 0};
    int amostra2[] = {0, 1};
    int amostra3[] = {1, 0};
    int amostra4[] = {1, 1};
    int amostra;

    int amostras[] = {0, 0, 1, 0, 0, 1, 1, 1};

    //FEEDFORWARD

    //Entrada -> Camada Oculta
    float *pesosOculta = preencher(alocaMatriz(numNeuronio, numEntrada), numNeuronio, numEntrada);
    float *entradas = alocaMatriz(numEntrada, 1);

    for(int i = 0; i < numEntrada; i++){
        if(i % 2 == 0){
            *(entradas+i) = 0.57;
        }else{
            *(entradas+i) = 0;
        }
    }

    printf("\nMatriz de pesos:\n");
    exibirMatriz(pesosOculta, numNeuronio, numEntrada);

    printf("\nMatriz de entrada:\n");
    exibirMatriz(entradas, numEntrada, 1);

    float *uOculta = produtoMatriz(pesosOculta, numNeuronio, numEntrada, entradas, numEntrada, 1);
    float *biasOculta = preencher(alocaMatriz(numNeuronio, 1), numNeuronio, 1);

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

    erro_o_saida = subtraiMatriz(y_d_saida, numSaida, 1, ySaida, numSaida, 1);
    printf("\nErro na saida");
    exibirMatriz(erro_o_saida, numSaida, 1);

    if(*(erro_o_saida) > Erro_desejado){
        printf("\n\t\ttrue\n");
    }

    float *d_sig = d_sigmoid(erro_o_saida, numSaida);
    exibirMatriz(d_sig, numSaida, 1);


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
