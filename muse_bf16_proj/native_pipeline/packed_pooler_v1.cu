// Original CUDA producer/consumer implementation, using public CUDA pipeline/WMMA APIs.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cuda/pipeline>
#include <cooperative_groups.h>
namespace cg=cooperative_groups;
namespace wm=nvcuda::wmma;

template<int BN,int BK,int PROD,int STAGES,bool PACKED>
__global__ void packed_pooler(const __nv_bfloat16* X,const __nv_bfloat16* W,
    const unsigned char* S,const unsigned char* E,const unsigned char* B,float* P,int split) {
    constexpr int M=16,N=4096,K=8192,LD=BK+8;
    extern __shared__ __align__(32) unsigned char memory[];
    auto buf=reinterpret_cast<__nv_bfloat16*>(memory);
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block,STAGES> ps;
    auto block=cg::this_thread_block();
    auto pipe=cuda::make_pipeline(block,&ps,PROD);
    int t=threadIdx.x,part=blockIdx.y,n0=blockIdx.x*BN;
    int iterations=K/split/BK;
    if(t<PROD) {
        for(int step=0;step<iterations;++step) {
            pipe.producer_acquire();
            auto a=buf+(step%STAGES)*(M+BN)*LD;
            auto b=a+M*LD;
            int k0=part*(K/split)+step*BK;
            for(int i=t;i<M*BK;i+=PROD) a[(i/BK)*LD+i%BK]=X[(i/BK)*K+k0+i%BK];
            for(int i=t;i<BN*BK;i+=PROD) {
                int row=n0+i/BK,k=k0+i%BK,idx=row*K+k;
                __nv_bfloat16 val;
                if constexpr(PACKED) {
                    unsigned base=B[row*(K/256)+k/256];
                    if(base==255) val=W[idx];
                    else {
                        unsigned sm=S[idx],ep=E[idx/2];
                        unsigned bits=(sm&127)|((sm&128)<<8)|((base+((ep>>((k&1)*4))&15))<<7);
                        val=__ushort_as_bfloat16(static_cast<unsigned short>(bits));
                    }
                } else val=W[idx];
                b[(i/BK)*LD+i%BK]=val;
            }
            pipe.producer_commit();
        }
    } else {
        int warp=(t-PROD)/32;
        wm::fragment<wm::accumulator,16,16,16,float> acc[BN/64];
        #pragma unroll
        for(int j=0;j<BN/64;++j) wm::fill_fragment(acc[j],0.f);
        for(int step=0;step<iterations;++step) {
            pipe.consumer_wait();
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
            pipe.consumer_release();
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
