/*
compile this with gcc net.c 
run - ./a.out

weight should be place inside wei folder (use pre_wei.py script for this)
image should be place inside wei golder (use read.py for this)

*/
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
// add padding 0, around the image- for for 3x3 kernel 1 line padding will be added around
float* padding(float* input, int sz, int k_d, int k_sz){
    int ssz = sz+ k_sz -1;
    int kk = k_sz/2;
    int i,j,k;
    float* out = malloc(ssz*ssz*k_d*sizeof(int));
    for(i=0;i<ssz;i++){
        for(j=0;j<ssz;j++){
            for(k=0;k<k_d;k++){
                if(i<kk || j<kk || i > ssz-1-kk || j> ssz-1-kk){
                    out[i*ssz*k_d + j*k_d + k]=0;
                }
                else{
                // printf("%d\n",(i-kk)*sz*k_d + (j-kk)*k_d + k);
                    out[i*ssz*k_d + j*k_d + k]=input[(i-kk)*sz*k_d + (j-kk)*k_d + k];
                }
            }
        }
    }
    // printf("%d\n",ssz*ssz);
    //         for (k = 0; k < k_d; k++){
    // for(i=0;i<ssz;i++){
    //     for (j=0;j<ssz;j++){
    //             printf("%.2f\t",out[i*ssz*k_d + j*k_d + k]);
    //         }
    //      printf("\n");
    //     }
    //     printf("___________________\n");
    // }
    return out;
}

// 3d input to 4d kernel convolution.
//      outdepth,  out size, input_layer,  input_size,   kernel_dep,  kern_size, weight,   bias,       stride, padding
float* con(int o_d, int o_sz, float* in_layer, int i_sz, int k_d, int k_sz, float* weight, float* bias, int str, int same){
    
    float* in_pad=in_layer;
    int oz,ox,oy,kx,ky,kz;
    int ssz = i_sz; // without padding
    if(same==1){
    ssz = i_sz+ k_sz -1;
    in_pad = padding(in_layer, i_sz, k_d, k_sz);          
    }

    float* out = malloc(o_sz*o_sz*o_d*sizeof(float));
    for(oz=0; oz<o_d; oz++){ //loop over  feature maps
        for(ox=0; ox<o_sz; ox++){ // loop over  rows
            for(oy=0; oy<o_sz; oy++){ // loop over  columns
                float ans=bias[oz]; // Step 1: load bias
                for(kx=0; kx<k_sz; kx++){ //loop over 6x6 window
                    for(ky=0; ky<k_sz; ky++){ // loop over 6x6 window
                        for(kz=0;kz<k_d;kz++){
                            // printf("%d ",(ox*str+kx)*ssz*k_d + (oy*str + ky)*k_d );
                            ans += in_pad[(ox*str+kx)*ssz*k_d + (oy*str + ky)*k_d + kz] * weight[oz*k_sz*k_sz*k_d + kx*k_sz*k_d + ky*k_d +kz]; // Step 2: convolution
                            // ans += in_pad[(ox*str+kx)*ssz+oy*str+ky] * weight[oz*k_sz*k_sz+kx*k_sz+ky]; // Step 2: convolution
                        }
                    }   
                }
                // printf("%d\n",(ox*str+kx)*ssz*k_d );
                out[ox*o_sz*o_d+oy*o_d+oz]=ans>0?ans:0 ;//relu activation
            }
        }
    }
return out;
}
                // input layer, size,       depth, kernelsize, number_ker, stride
float* maxpool(float* in_layer, int i_sz, int i_d, int k_sz, int k_n, int s){
    int i,j,k,l,m;
    int o_sz=(i_sz - k_sz )/s +1 ;
    // printf("%d\n", o_sz );
    float* out = calloc(o_sz*o_sz*k_n ,sizeof(float));
        for (int i = 0; i < o_sz; ++i){
            for (int j = 0; j < o_sz; ++j){
                for (int k = 0; k < k_n; ++k){
                    float max=0;
                    // printf("\n");
                    for(int l=0;l<k_sz;l++){
                        for(int m=0;m<k_sz;m++){
                            // printf("%d ", (i*s+l)*i_sz + j*s+m);
                            int idx = (i*s+l)*i_sz*i_d + (j*s+m)*i_d + k;
                            if(max<in_layer[idx])
                                max = in_layer[idx];
                        }
                    }
                    out[i*o_sz*k_n + j*k_n + k] = max;
            }
        }
    }

    return out;
}

               // input layer, size,       depth, kernelsize, number_ker, stride
float* avgpool(float* in_layer, int i_sz, int i_d, int k_sz, int k_n, int s){
    int i,j,k,l,m;
    int o_sz=(i_sz - k_sz )/s +1 ;
    // printf("%d\n", o_sz );
    float* out = calloc(o_sz*o_sz*k_n ,sizeof(float));
        for (int i = 0; i < o_sz; ++i){
            for (int j = 0; j < o_sz; ++j){
                for (int k = 0; k < k_n; ++k){
                    float avg=0;
                    // printf("\n");
                    for(int l=0;l<k_sz;l++){
                        for(int m=0;m<k_sz;m++){
                            // printf("%d ", (i*s+l)*i_sz + j*s+m);
                            int idx = (i*s+l)*i_sz*i_d + (j*s+m)*i_d + k;
                                avg += in_layer[idx];
                        }
                    }
                    out[i*o_sz*k_n + j*k_n + k] = avg/(k_sz*k_sz);
            }
        }
    }

    return out;
}

// get weight from file (for each layer seperate files)
float* getwei(int sz, char* name){
    float* wei = calloc(sz, sizeof(float));
    FILE *ff = fopen(name, "r");
    int i;
    for (int i = 0; i < sz; ++i)
    {
        fscanf(ff,"%f",wei+i);
        // printf("%f\n",*(wei+i) );
    }
    fclose(ff);
    return wei;
}

// read input image
float* getimg(int sz){
    float* wei = calloc(sz*sz*3, sizeof(float));
    FILE *ff = fopen("wei/image", "r");
    int i;
    for (int i = 0; i < sz*sz*3; ++i)
    {
        fscanf(ff,"%f",wei+i);
        // printf("%f\n",*(wei+i) );
    }
    fclose(ff);
    return wei;
}

//for debuggin only
void test_op(){

    int i;
    int pad=1;
    int in_s=4; //for odd work like tensorflow
    int in_d= 1;
    int out_s=  2;
    int out_d=  1;
    int ker_s= 3;
    int ker_d= in_d;
    int ker_n= out_d;
    int str= 1;


    float bi[2]={0,0};
    //////////integer
    float in[in_s*in_s*in_d]; for(i =0 ;i<in_s*in_s*in_d;i++)in[i]=i;
    float ker[ker_s*ker_s*ker_d*ker_n];for(i=0;i<ker_s*ker_s*ker_d*ker_n;i++){ker[i]=1;}
    
    ////////////random
    // float in[in_s*in_s*in_d]; for(i =0 ;i<in_s*in_s*in_d;i++)in[i]=(float)rand()/rand();
    // float ker[ker_s*ker_s*ker_d*ker_n];for(i=0;i<ker_s*ker_s*ker_d*ker_n;i++){ker[i]=(float)rand()/rand();}


    // FILE *ff=fopen("in.txt","w");
    // for(i =0 ;i<in_s*in_s*in_d;i++)fprintf(ff,"%f,", in[i]); fprintf(ff,"\n");
    // for(i=0;i<ker_s*ker_s*ker_d*ker_n;i++)fprintf(ff,"%f,", ker[i]); fprintf(ff,"\n");


///average pool test
    // float* res = avgpool(in, in_s, in_d, ker_s, ker_n, str);
    // for(i = 0;i<out_s*out_s*out_d;i++){
    //     printf("%f ", res[i]);
    // }

// ///////conv test
    float* res = con(out_d,out_s,  in,in_s,   ker_d,ker_s,   ker, bi,pad,0);
    for(i = 0;i<out_s*out_s*out_d;i++){
        printf("%.3f ", res[i]);
    }
}


//concatination used after expand
float* concat(float* in1,float* in2,int sz, int d){
    float* out = malloc(sz*sz*d*2*sizeof(float));
    int i,j,k;
    for (int i = 0; i < sz; ++i){
        for (int j = 0; j < sz; ++j){
            for (int k = 0; k < d; ++k){
                // printf("%d\n", i*sz*d*2 + j*d*2 +k);
             out[i*sz*d*2 + j*d*2 +k] = in1[i*sz*d + j*d +k];
            }
            // printf("__\n");
            for (int k = 0; k < d; ++k){
                // printf("%d\n", i*sz*d*2 + j*d*2 + d +k);
             out[i*sz*d*2 + j*d*2 + d +k] = in2[i*sz*d + j*d +k];   
            }
        }
    }

    return out;
}

///test concat
void test_con(){
    int i,j,k;
    int in_s=4; //for odd work like tensorflow
    int in_d= 1;
    
    float in1[in_s*in_s*in_d]; for(i =0 ;i<in_s*in_s*in_d;i++)in1[i]=i;
    float in2[in_s*in_s*in_d]; for(i =0 ;i<in_s*in_s*in_d;i++)in2[i]=i+100;
    float* out = concat(in1,in2,in_s,in_d);

    // for (k = 0; k < in_d*2; k++){
    //     for(i=0;i<in_s;i++){
    //         for (j=0;j<in_s;j++){
    //                 printf("%.2f\t",out[i*in_s*in_d + j*in_d + k]);
    //             }
    //          printf("\n");
    //         }
    //         printf("___________________\n");
    // }
    
}
int main(){

// test_con();
// return 0;
int i;

int in_sz=227,in_dep=3;
int k_sz=3,k_d=in_dep,k_n=64;
int o_dep=k_n,o_sz=113;
int str = 2,same=0;
float* img = getimg(in_sz);
/////////////conv 1/////////////////////////////
float* ker = getwei(k_n*k_sz*k_sz*k_d,"wei/conv1_ker");
float* bias = getwei(k_n,"wei/conv1_bias");
float* conv1 = con(o_dep,o_sz,img,in_sz,   in_dep,k_sz, ker, bias, str,same);
free(ker);free(bias);
free(img);
//////////pool1
in_sz=o_sz;in_dep=o_dep;
k_sz=3;k_d=in_dep;
o_dep=in_dep;o_sz=56;
str = 2;
float*  pool1= maxpool(conv1, in_sz, in_dep, k_sz, o_dep, str);
free(conv1);
//////////fire 2////////////////////////////////////////
//////squeze
in_sz=o_sz;in_dep=o_dep;
k_sz=1;k_d=in_dep,k_n=16;
o_dep=k_n;o_sz=in_sz;
str = 1;same=0;
ker = getwei(k_n*k_sz*k_sz*k_d,"wei/fire2_squeeze1x1_ker");
bias = getwei(k_n,"wei/fire2_squeeze1x1_bias");
float* f2_squ = con(o_dep,o_sz,pool1,in_sz,   in_dep,k_sz, ker, bias, str,same);
free(ker);free(bias);
free(pool1);
////////expand 1
in_sz=o_sz;in_dep=o_dep;
k_sz=1;k_d=in_dep,k_n=64;
o_dep=k_n;o_sz=in_sz;
str = 1;same=0;
ker = getwei(k_n*k_sz*k_sz*k_d,"wei/fire2_expand1x1_ker");
bias = getwei(k_n,"wei/fire2_expand1x1_bias");
float* f2_exp1 = con(o_dep,o_sz,f2_squ,in_sz,   in_dep,k_sz, ker, bias, str,same);
free(ker);free(bias);
///////////expand 3
k_sz=3;k_n=64;
o_dep=k_n;o_sz=in_sz;
str = 1;same=1;
ker = getwei(k_n*k_sz*k_sz*k_d,"wei/fire2_expand3x3_ker");
bias = getwei(k_n,"wei/fire2_expand3x3_bias");
float* f2_exp3 = con(o_dep,o_sz,f2_squ,in_sz,   in_dep,k_sz, ker, bias, str,same);
free(ker);free(bias);

float* f2 = concat(f2_exp1,f2_exp3,o_sz,o_dep);
o_dep*=2; //concatination doubles the ouput depth
free(f2_squ);free(f2_exp3);free(f2_exp1);
//printf("fire2 out : %d %d %d\n", o_sz,o_sz,o_dep);

//////////fire 3////////////////////////////////////////
//////squeze
in_sz=o_sz;in_dep=o_dep;
k_sz=1;k_d=in_dep,k_n=16;
o_dep=k_n;o_sz=in_sz;
str = 1;same=0;
ker = getwei(k_n*k_sz*k_sz*k_d,"wei/fire3_squeeze1x1_ker");
bias = getwei(k_n,"wei/fire3_squeeze1x1_bias");
// printf("o_dep %d out_sz, %d in_si %d, in_dep %d, kers %d\n",o_dep,o_sz,in_sz, in_dep, k_sz );
float* f3_squ = con(o_dep,o_sz,f2,in_sz,   in_dep,k_sz, ker, bias, str,same);
free(ker);free(bias);
free(f2);
////////expand 1
in_sz=o_sz;in_dep=o_dep;
k_sz=1;k_d=in_dep,k_n=64;
o_dep=k_n;o_sz=in_sz;
str = 1;same=0;
ker = getwei(k_n*k_sz*k_sz*k_d,"wei/fire3_expand1x1_ker");
bias = getwei(k_n,"wei/fire3_expand1x1_bias");
float* f3_exp1 = con(o_dep,o_sz,f3_squ,in_sz,   in_dep,k_sz, ker, bias, str,same);
free(ker);free(bias);
///////////expand 3
k_sz=3;k_n=64;
o_dep=k_n;o_sz=in_sz;
str = 1;same=1;
ker = getwei(k_n*k_sz*k_sz*k_d,"wei/fire3_expand3x3_ker");
bias = getwei(k_n,"wei/fire3_expand3x3_bias");
float* f3_exp3 = con(o_dep,o_sz,f3_squ,in_sz,   in_dep,k_sz, ker, bias, str,same);
free(ker);free(bias);

float* f3 = concat(f3_exp1,f3_exp3,o_sz,o_dep);
o_dep*=2;
free(f3_squ);free(f3_exp3);free(f3_exp1);
//printf("fire3 out : %d %d %d\n", o_sz,o_sz,o_dep);


//////////pool3/////////////////
in_sz=o_sz;in_dep=o_dep;
k_sz=3;k_d=in_dep;
o_dep=in_dep;o_sz=27;
str = 2;
float*  pool3= maxpool(f3, in_sz, in_dep, k_sz, o_dep, str);
free(f3);

////////// fire 4////////////////////////////////////////
//////squeze
in_sz=o_sz;in_dep=o_dep;
k_sz=1;k_d=in_dep,k_n=32;
o_dep=k_n;o_sz=in_sz;
str = 1;same=0;
ker = getwei(k_n*k_sz*k_sz*k_d,"wei/fire4_squeeze1x1_ker");
bias = getwei(k_n,"wei/fire4_squeeze1x1_bias");
// printf("o_dep %d out_sz, %d in_si %d, in_dep %d, kers %d\n",o_dep,o_sz,in_sz, in_dep, k_sz );
float* f4_squ = con(o_dep,o_sz,pool3,in_sz,   in_dep,k_sz, ker, bias, str,same);
free(ker);free(bias);
free(pool3);
////////expand 1
in_sz=o_sz;in_dep=o_dep;
k_sz=1;k_d=in_dep,k_n=128;
o_dep=k_n;o_sz=in_sz;
str = 1;same=0;
ker = getwei(k_n*k_sz*k_sz*k_d,"wei/fire4_expand1x1_ker");
bias = getwei(k_n,"wei/fire4_expand1x1_bias");
float* f4_exp1 = con(o_dep,o_sz,f4_squ,in_sz,   in_dep,k_sz, ker, bias, str,same);
free(ker);free(bias);
///////////expand 3
k_sz=3;k_n=128;
o_dep=k_n;o_sz=in_sz;
str = 1;same=1;
ker = getwei(k_n*k_sz*k_sz*k_d,"wei/fire4_expand3x3_ker");
bias = getwei(k_n,"wei/fire4_expand3x3_bias");
float* f4_exp3 = con(o_dep,o_sz,f4_squ,in_sz,   in_dep,k_sz, ker, bias, str,same);
free(ker);free(bias);

float* f4 = concat(f4_exp1,f4_exp3,o_sz,o_dep);
o_dep*=2;
free(f4_squ);free(f4_exp3);free(f4_exp1);
//printf("fire4 out : %d %d %d\n", o_sz,o_sz,o_dep);

////////// fire 5////////////////////////////////////////
//////squeze
in_sz=o_sz;in_dep=o_dep;
k_sz=1;k_d=in_dep,k_n=32;
o_dep=k_n;o_sz=in_sz;
str = 1;same=0;
ker = getwei(k_n*k_sz*k_sz*k_d,"wei/fire5_squeeze1x1_ker");
bias = getwei(k_n,"wei/fire5_squeeze1x1_bias");
float* f5_squ = con(o_dep,o_sz,f4,in_sz,   in_dep,k_sz, ker, bias, str,same);
free(ker);free(bias);
free(f4);
////////expand 1
in_sz=o_sz;in_dep=o_dep;
k_sz=1;k_d=in_dep,k_n=128;
o_dep=k_n;o_sz=in_sz;
str = 1;same=0;
ker = getwei(k_n*k_sz*k_sz*k_d,"wei/fire5_expand1x1_ker");
bias = getwei(k_n,"wei/fire5_expand1x1_bias");
float* f5_exp1 = con(o_dep,o_sz,f5_squ,in_sz,   in_dep,k_sz, ker, bias, str,same);
free(ker);free(bias);
///////////expand 3
k_sz=3;k_n=128;
o_dep=k_n;o_sz=in_sz;
str = 1;same=1;
ker = getwei(k_n*k_sz*k_sz*k_d,"wei/fire5_expand3x3_ker");
bias = getwei(k_n,"wei/fire5_expand3x3_bias");
float* f5_exp3 = con(o_dep,o_sz,f5_squ,in_sz,   in_dep,k_sz, ker, bias, str,same);
free(ker);free(bias);

float* f5 = concat(f5_exp1,f5_exp3,o_sz,o_dep);
o_dep*=2;
free(f5_squ);free(f5_exp3);free(f5_exp1);
//printf("fire5 out : %d %d %d\n", o_sz,o_sz,o_dep);

//////////pool3/////////////////
in_sz=o_sz;in_dep=o_dep;
k_sz=3;k_d=in_dep;
o_dep=in_dep;o_sz=13;
str = 2;
float*  pool5= maxpool(f5, in_sz, in_dep, k_sz, o_dep, str);
free(f5);

////////// fire 6////////////////////////////////////////
//////squeze
in_sz=o_sz;in_dep=o_dep;
k_sz=1;k_d=in_dep,k_n=48;
o_dep=k_n;o_sz=in_sz;
str = 1;same=0;
ker = getwei(k_n*k_sz*k_sz*k_d,"wei/fire6_squeeze1x1_ker");
bias = getwei(k_n,"wei/fire6_squeeze1x1_bias");
float* f6_squ = con(o_dep,o_sz,pool5,in_sz,   in_dep,k_sz, ker, bias, str,same);
free(ker);free(bias);
free(pool5);
////////expand 1
in_sz=o_sz;in_dep=o_dep;
k_sz=1;k_d=in_dep,k_n=192;
o_dep=k_n;o_sz=in_sz;
str = 1;same=0;
ker = getwei(k_n*k_sz*k_sz*k_d,"wei/fire6_expand1x1_ker");
bias = getwei(k_n,"wei/fire6_expand1x1_bias");
float* f6_exp1 = con(o_dep,o_sz,f6_squ,in_sz,   in_dep,k_sz, ker, bias, str,same);
free(ker);free(bias);
///////////expand 3
k_sz=3;k_n=192;
o_dep=k_n;o_sz=in_sz;
str = 1;same=1;
ker = getwei(k_n*k_sz*k_sz*k_d,"wei/fire6_expand3x3_ker");
bias = getwei(k_n,"wei/fire6_expand3x3_bias");
float* f6_exp3 = con(o_dep,o_sz,f6_squ,in_sz,   in_dep,k_sz, ker, bias, str,same);
free(ker);free(bias);

float* f6 = concat(f6_exp1,f6_exp3,o_sz,o_dep);
o_dep*=2;
free(f6_squ);free(f6_exp3);free(f6_exp1);
//printf("fire6 out : %d %d %d\n", o_sz,o_sz,o_dep);

////////// fire 7////////////////////////////////////////
//////squeze
in_sz=o_sz;in_dep=o_dep;
k_sz=1;k_d=in_dep,k_n=48;
o_dep=k_n;o_sz=in_sz;
str = 1;same=0;
ker = getwei(k_n*k_sz*k_sz*k_d,"wei/fire7_squeeze1x1_ker");
bias = getwei(k_n,"wei/fire7_squeeze1x1_bias");
float* f7_squ = con(o_dep,o_sz,f6,in_sz,   in_dep,k_sz, ker, bias, str,same);
free(ker);free(bias);
free(f6);
////////expand 1
in_sz=o_sz;in_dep=o_dep;
k_sz=1;k_d=in_dep,k_n=192;
o_dep=k_n;o_sz=in_sz;
str = 1;same=0;
ker = getwei(k_n*k_sz*k_sz*k_d,"wei/fire7_expand1x1_ker");
bias = getwei(k_n,"wei/fire7_expand1x1_bias");
float* f7_exp1 = con(o_dep,o_sz,f7_squ,in_sz,   in_dep,k_sz, ker, bias, str,same);
free(ker);free(bias);
///////////expand 3
k_sz=3;k_n=192;
o_dep=k_n;o_sz=in_sz;
str = 1;same=1;
ker = getwei(k_n*k_sz*k_sz*k_d,"wei/fire7_expand3x3_ker");
bias = getwei(k_n,"wei/fire7_expand3x3_bias");
float* f7_exp3 = con(o_dep,o_sz,f7_squ,in_sz,   in_dep,k_sz, ker, bias, str,same);
free(ker);free(bias);

float* f7 = concat(f7_exp1,f7_exp3,o_sz,o_dep);
o_dep*=2;
free(f7_squ);free(f7_exp3);free(f7_exp1);
//printf("fire7 out : %d %d %d\n", o_sz,o_sz,o_dep);

////////// fire 8////////////////////////////////////////
//////squeze
in_sz=o_sz;in_dep=o_dep;
k_sz=1;k_d=in_dep,k_n=64;
o_dep=k_n;o_sz=in_sz;
str = 1;same=0;
ker = getwei(k_n*k_sz*k_sz*k_d,"wei/fire8_squeeze1x1_ker");
bias = getwei(k_n,"wei/fire8_squeeze1x1_bias");
float* f8_squ = con(o_dep,o_sz,f7,in_sz,   in_dep,k_sz, ker, bias, str,same);
free(ker);free(bias);
free(f7);
////////expand 1
in_sz=o_sz;in_dep=o_dep;
k_sz=1;k_d=in_dep,k_n=256;
o_dep=k_n;o_sz=in_sz;
str = 1;same=0;
ker = getwei(k_n*k_sz*k_sz*k_d,"wei/fire8_expand1x1_ker");
bias = getwei(k_n,"wei/fire8_expand1x1_bias");
float* f8_exp1 = con(o_dep,o_sz,f8_squ,in_sz,   in_dep,k_sz, ker, bias, str,same);
free(ker);free(bias);
///////////expand 3
k_sz=3;k_n=256;
o_dep=k_n;o_sz=in_sz;
str = 1;same=1;
ker = getwei(k_n*k_sz*k_sz*k_d,"wei/fire8_expand3x3_ker");
bias = getwei(k_n,"wei/fire8_expand3x3_bias");
float* f8_exp3 = con(o_dep,o_sz,f8_squ,in_sz,   in_dep,k_sz, ker, bias, str,same);
free(ker);free(bias);

float* f8 = concat(f8_exp1,f8_exp3,o_sz,o_dep);
o_dep*=2;
free(f8_squ);free(f8_exp3);free(f8_exp1);
//printf("fire8 out : %d %d %d\n", o_sz,o_sz,o_dep);

////////// fire 9////////////////////////////////////////
//////squeze
in_sz=o_sz;in_dep=o_dep;
k_sz=1;k_d=in_dep,k_n=64;
o_dep=k_n;o_sz=in_sz;
str = 1;same=0;
ker = getwei(k_n*k_sz*k_sz*k_d,"wei/fire9_squeeze1x1_ker");
bias = getwei(k_n,"wei/fire9_squeeze1x1_bias");
float* f9_squ = con(o_dep,o_sz,f8,in_sz,   in_dep,k_sz, ker, bias, str,same);
free(ker);free(bias);
free(f8);
////////expand 1
in_sz=o_sz;in_dep=o_dep;
k_sz=1;k_d=in_dep,k_n=256;
o_dep=k_n;o_sz=in_sz;
str = 1;same=0;
ker = getwei(k_n*k_sz*k_sz*k_d,"wei/fire9_expand1x1_ker");
bias = getwei(k_n,"wei/fire9_expand1x1_bias");
float* f9_exp1 = con(o_dep,o_sz,f9_squ,in_sz,   in_dep,k_sz, ker, bias, str,same);
free(ker);free(bias);
///////////expand 3
k_sz=3;k_n=256;
o_dep=k_n;o_sz=in_sz;
str = 1;same=1;
ker = getwei(k_n*k_sz*k_sz*k_d,"wei/fire9_expand3x3_ker");
bias = getwei(k_n,"wei/fire9_expand3x3_bias");
float* f9_exp3 = con(o_dep,o_sz,f9_squ,in_sz,   in_dep,k_sz, ker, bias, str,same);
free(ker);free(bias);

float* f9 = concat(f9_exp1,f9_exp3,o_sz,o_dep);
o_dep*=2;
free(f9_squ);free(f9_exp3);free(f9_exp1);
//printf("fire9 out : %d %d %d\n", o_sz,o_sz,o_dep);


/////////////conv 10/////////////////////////////
in_sz=o_sz;in_dep=o_dep;
k_sz=1;k_d=in_dep,k_n=1000;
o_dep=k_n;o_sz=in_sz;
str = 1;same=0;
ker = getwei(k_n*k_sz*k_sz*k_d,"wei/conv10_ker");
bias = getwei(k_n,"wei/conv10_bias");
float* conv10 = con(o_dep,o_sz,f9,in_sz,   in_dep,k_sz, ker, bias, str,same);
free(ker);free(bias);
free(f9);
//////////pool10

in_sz=o_sz;in_dep=o_dep;
k_sz=13;k_d=in_dep;
o_dep=in_dep;o_sz=1;
str = 1;
float*  pool10= avgpool(conv10, in_sz, in_dep, k_sz, o_dep, str);
free(conv10);


 // for (int i = 0; i < 1000; ++i) {    printf("%.2f \n", pool10[i]);    }

/////////get index of heights probability
float tot =0;
int idx = 0;
float max =0;
for (int i = 0; i < 1000; ++i) {
    tot+=pool10[i];
    if(pool10[i]>max){
        max = pool10[i];
        idx = i;
    }
}

printf("idx %d ", idx);

}