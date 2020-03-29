#ifndef FMATRIZ_H_INCLUDED
#define FMATRIZ_H_INCLUDED

void exibirMatriz(float *ptr_matriz, int nlin, int ncol);
float *alocaMatriz(int nlin, int ncol);
float *sigmoid(float *matriz, int nlin);
float *produtoMatriz(float *mat1, int nlin_mat1, int ncol_mat1, float *mat2, int nlin_mat2, int ncol_mat2);
float *preencher(float *matriz, int nlin, int ncol);
float *somaMatrizes(float *mat1, int nlin_mat1, int ncol_mat1, float *mat2, int nlin_mat2, int ncol_mat2);
float *subtraiMatriz(float *mat1, int nlin_mat1, int ncol_mat1, float *mat2, int nlin_mat2, int ncol_mat2);
void treinar();
float *d_sigmoid(float *matriz, int nlin);
float *hadamard(float *mat1, int nlin_mat1, int ncol_mat1, float *mat2, int nlin_mat2, int ncol_mat2);

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

float *hadamard(float *mat1, int nlin_mat1, int ncol_mat1, float *mat2, int nlin_mat2, int ncol_mat2){

    int i, j;
    float *had_mat, soma;

    had_mat = alocaMatriz(nlin_mat1, ncol_mat2);

    if(nlin_mat1 == nlin_mat2 && ncol_mat1 == ncol_mat2){

        for(i = 0; i < (nlin_mat1*ncol_mat1); i++){
                *(had_mat+i) = (*(mat1 + i)  * *(mat2 + i));
        }
    }
    return had_mat;
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


#endif // FMATRIZ_H_INCLUDED
