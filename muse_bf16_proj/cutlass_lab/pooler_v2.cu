// Original PyTorch binding to public CUTLASS 3.9.2 collective-builder APIs.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
using namespace cute;

void ck(cutlass::Status s,const char* op) {TORCH_CHECK(s==cutlass::Status::kSuccess,op,": ",cutlassGetStatusString(s));}

template<int TM,int TN,int TK,int STAGES,bool DIRECT>
torch::Tensor invoke(torch::Tensor x,torch::Tensor w,int split) {
    using BF=cutlass::bfloat16_t;
    using Tile=Shape<Int<TM>,Int<TN>,Int<TK>>;
    using Cluster=Shape<_1,_1,_1>;
    using Epilogue=typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90,cutlass::arch::OpClassTensorOp,Tile,Cluster,
        cutlass::epilogue::collective::EpilogueTileAuto,float,float,
        float,cutlass::layout::ColumnMajor,4,float,cutlass::layout::ColumnMajor,4,
        cute::conditional_t<DIRECT,cutlass::epilogue::NoSmemWarpSpecialized,cutlass::epilogue::TmaWarpSpecialized>>::CollectiveOp;
    using Mainloop=typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90,cutlass::arch::OpClassTensorOp,
        BF,cutlass::layout::RowMajor,8,BF,cutlass::layout::ColumnMajor,8,float,Tile,Cluster,
        cutlass::gemm::collective::StageCount<STAGES>,cutlass::gemm::KernelTmaWarpSpecialized>::CollectiveOp;
    using Kernel=cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>,Mainloop,Epilogue,cutlass::gemm::PersistentScheduler>;
    using Gemm=cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
    auto out=torch::empty({split,16,4096},x.options().dtype(at::kFloat));
    typename Kernel::StrideA sa=make_stride(int64_t(8192),_1{},int64_t(8192/split));
    typename Kernel::StrideB sb=make_stride(int64_t(8192),_1{},int64_t(8192/split));
    typename Kernel::StrideC sc=make_stride(_1{},int64_t(4096),int64_t(16*4096));
    typename Kernel::StrideD sd=sc;
    cutlass::KernelHardwareInfo hw;
    hw.device_id=x.get_device();hw.sm_count=at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,
        {4096,16,8192/split,split},
        {reinterpret_cast<BF*>(w.data_ptr()),sa,reinterpret_cast<BF*>(x.data_ptr()),sb},
        {{1.f,0.f},out.data_ptr<float>(),sc,out.data_ptr<float>(),sd},hw};
    Gemm gemm;
    ck(gemm.can_implement(args),"can_implement");
    auto bytes=Gemm::get_workspace_size(args);
    auto workspace=torch::empty({static_cast<int64_t>(bytes)},x.options().dtype(at::kByte));
    auto stream=at::cuda::getCurrentCUDAStream();
    ck(gemm.initialize(args,bytes?workspace.data_ptr():nullptr,stream),"initialize");
    ck(gemm.run(stream),"run");
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor run(torch::Tensor x,torch::Tensor w,int config,int split) {
    TORCH_CHECK(x.is_cuda()&&w.is_cuda()&&x.device()==w.device());
    TORCH_CHECK(x.is_contiguous()&&w.is_contiguous()&&x.scalar_type()==at::kBFloat16&&w.scalar_type()==at::kBFloat16);
    TORCH_CHECK(x.dim()==2&&x.size(0)==16&&x.size(1)==8192&&w.dim()==2&&w.size(0)==4096&&w.size(1)==8192);
    TORCH_CHECK(split==1||split==2||split==4||split==8||split==16);
    c10::cuda::CUDAGuard guard(x.device());
    switch(config) {
        case 0:return invoke<64,16,128,2,true>(x,w,split);
        case 1:return invoke<64,16,128,3,true>(x,w,split);
        case 2:return invoke<64,16,128,4,true>(x,w,split);
        case 3:return invoke<64,16,128,4,false>(x,w,split);
        case 4:return invoke<128,16,128,3,true>(x,w,split);
        case 5:return invoke<64,16,256,2,true>(x,w,split);
        default:TORCH_CHECK(false,"invalid config");
    }
}
