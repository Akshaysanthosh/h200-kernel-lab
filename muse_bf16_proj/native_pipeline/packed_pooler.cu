// Original CUDA producer/consumer implementation, using public CUDA pipeline/WMMA APIs.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cuda/barrier>
#include <cooperative_groups.h>
namespace cg=cooperative_groups;
namespace wm=nvcuda::wmma;

template<int BN,int BK,int PROD,int STAGES,bool PACKED>
__global__ void packed_pooler(const __nv_bfloat16* X,const __nv_bfloat16* W,
    const unsigned char* S,const unsigned char* E,const unsigned char* B,float* P,int split) {
    constexpr int M=16,N=4096,K=8192,LD=BK+8;
    extern __shared__ __align__(32) unsigned char memory[];
    auto buf=reinterpret_cast<__nv_bfloat16*>(memory);
    __shared__ cuda::barrier<cuda::thread_scope_block> ready[STAGES],freebuf[STAGES];
    if(threadIdx.x<STAGES) {
        init(&ready[threadIdx.x],PROD+128);
        init(&freebuf[threadIdx.x],PROD+128);
    }
    __syncthreads();
    int t=threadIdx.x,part=blockIdx.y,n0=blockIdx.x*BN;
    int iterations=K/split/BK;
    if(t<PROD) {
        for(int step=0;step<iterations;++step) {
            freebuf[step%STAGES].arrive_and_wait();
            auto a=buf+(step%STAGES)*(M+BN)*LD;
            auto b=a+M*LD;
            int k0=part*(K/split)+step*BK;
            // One aligned 16-byte activation/weight vector per producer lane.
            for(int v=t;v<M*BK/8;v+=PROD) {
                int i=v*8;
                *reinterpret_cast<uint4*>(a+(i/BK)*LD+i%BK)=
                    *reinterpret_cast<const uint4*>(X+(i/BK)*K+k0+i%BK);
            }
            for(int v=t;v<BN*BK/8;v+=PROD) {
                int i=v*8,row=n0+i/BK,k=k0+i%BK,idx=row*K+k;
                uint4 value;
                if constexpr(PACKED) {
                    unsigned base=B[row*(K/256)+k/256];
                    if(base==255) value=*reinterpret_cast<const uint4*>(W+idx);
                    else {
                        unsigned long long sm=*reinterpret_cast<const unsigned long long*>(S+idx);
                        unsigned ep=*reinterpret_cast<const unsigned*>(E+idx/2);
                        unsigned words[4];
                        #pragma unroll
                        for(int pair=0;pair<4;++pair) {
                            unsigned s0=(sm>>(pair*16))&255,s1=(sm>>(pair*16+8))&255;
                            unsigned b0=(s0&127)|((s0&128)<<8)|((base+((ep>>(pair*8))&15))<<7);
                            unsigned b1=(s1&127)|((s1&128)<<8)|((base+((ep>>(pair*8+4))&15))<<7);
                            words[pair]=b0|(b1<<16);
                        }
                        value=make_uint4(words[0],words[1],words[2],words[3]);
                    }
                } else value=*reinterpret_cast<const uint4*>(W+idx);
                *reinterpret_cast<uint4*>(b+(i/BK)*LD+i%BK)=value;
            }
            (void)ready[step%STAGES].arrive();
        }
    } else {
        for(int s=0;s<STAGES;++s) (void)freebuf[s].arrive();
        int warp=(t-PROD)/32;
        wm::fragment<wm::accumulator,16,16,16,float> acc[BN/64];
        #pragma unroll
        for(int j=0;j<BN/64;++j) wm::fill_fragment(acc[j],0.f);
        for(int step=0;step<iterations;++step) {
            ready[step%STAGES].arrive_and_wait();
            auto a=buf+(step%STAGES)*(M+BN)*LD;
            auto b=a+M*LD;
            #pragma unroll
            for(int kk=0;kk<BK;kk+=16) {
                wm::fragment<wm::matrix_a,16,16,16,__nv_bfloat16,wm::row_major> af;
                wm::load_matrix_sync(af,a+kk,LD);
                #pragma unroll
                for(int j=0;j<BN/64;++j) {
                    wm::fragment<wm::matrix_b,16,16,16,__nv_bfloat16,wm::col_major> bf;
                    wm::load_matrix_sync(bf,b+(warp*16+j*64)*LD+kk,LD);
                    wm::mma_sync(acc[j],af,bf,acc[j]);
                }
            }
            (void)freebuf[step%STAGES].arrive();
        }
        #pragma unroll
        for(int j=0;j<BN/64;++j) wm::store_matrix_sync(P+part*M*N+n0+warp*16+j*64,acc[j],N,wm::mem_row_major);
    }
}

template<int BN,int BK,int PROD,int STAGES,bool PACKED>
void launch(const torch::Tensor& x,const torch::Tensor& w,const torch::Tensor& s,const torch::Tensor& e,const torch::Tensor& b,torch::Tensor& p,int split,cudaStream_t stream) {
    constexpr int bytes=STAGES*(16+BN)*(BK+8)*2;
    auto kernel=packed_pooler<BN,BK,PROD,STAGES,PACKED>;
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));
    kernel<<<dim3(4096/BN,split),PROD+128,bytes,stream>>>(reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()),s.data_ptr<unsigned char>(),e.data_ptr<unsigned char>(),b.data_ptr<unsigned char>(),p.data_ptr<float>(),split);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<int BN,int BK,int PROD,int STAGES>
void dispatch(bool packed,const torch::Tensor& x,const torch::Tensor& w,const torch::Tensor& s,const torch::Tensor& e,const torch::Tensor& b,torch::Tensor& p,int split,cudaStream_t stream) {
    if(packed)launch<BN,BK,PROD,STAGES,true>(x,w,s,e,b,p,split,stream);
    else launch<BN,BK,PROD,STAGES,false>(x,w,s,e,b,p,split,stream);
}

torch::Tensor run(torch::Tensor x,torch::Tensor w,torch::Tensor s,torch::Tensor e,torch::Tensor b,int cfg,int split,bool packed) {
    TORCH_CHECK(x.is_cuda()&&x.is_contiguous()&&x.scalar_type()==at::kBFloat16&&x.size(0)==16&&x.size(1)==8192);
    TORCH_CHECK(w.is_cuda()&&w.is_contiguous()&&w.scalar_type()==at::kBFloat16&&w.size(0)==4096&&w.size(1)==8192);
    TORCH_CHECK(split==1||split==2||split==4||split==8||split==16);
    TORCH_CHECK(s.is_cuda()&&e.is_cuda()&&b.is_cuda()&&s.is_contiguous()&&e.is_contiguous()&&b.is_contiguous());
    TORCH_CHECK(s.scalar_type()==at::kByte&&e.scalar_type()==at::kByte&&b.scalar_type()==at::kByte);
    TORCH_CHECK(s.numel()==4096*8192&&e.numel()==4096*4096&&b.numel()==4096*32);
    TORCH_CHECK(x.device()==w.device()&&x.device()==s.device()&&x.device()==e.device()&&x.device()==b.device());
    c10::cuda::CUDAGuard guard(x.device());
    auto p=torch::empty({split,16,4096},x.options().dtype(at::kFloat));
    auto stream=at::cuda::getCurrentCUDAStream();
    switch(cfg) {
#define C(ID,N,K,P,S) case ID: dispatch<N,K,P,S>(packed,x,w,s,e,b,p,split,stream);break;
      C(0,64,64,64,1) C(1,64,64,64,2) C(2,64,128,64,1) C(3,64,128,64,2)
      C(4,64,128,128,2) C(5,128,64,128,2) C(6,128,128,128,2) C(7,64,128,64,3)
#undef C
      default: TORCH_CHECK(false,"invalid config");
    }
    return p;
}
