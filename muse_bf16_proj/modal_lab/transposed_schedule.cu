// Lab-only binding to NVIDIA CUTLASS 3.9.2 collective-builder APIs.
// Exact BF16 operands, FP32 accumulation, BF16 output. No decoding.
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
void ck(cutlass::Status s,const char* op){TORCH_CHECK(s==cutlass::Status::kSuccess,op,": ",cutlassGetStatusString(s));}
template<bool PACK,int TM,int TN,int ST,int CM,bool COOP,bool SK>
torch::Tensor invoke(torch::Tensor x,torch::Tensor w,int swizzle,int raster){
 using BF=cutlass::bfloat16_t;using Tile=Shape<Int<TM>,Int<TN>,_64>;using Cluster=Shape<Int<CM>,_1,_1>;
 using EpiSchedule=cute::conditional_t<COOP,cutlass::epilogue::TmaWarpSpecializedCooperative,cutlass::epilogue::TmaWarpSpecialized>;
 using Schedule=cute::conditional_t<COOP,cutlass::gemm::KernelTmaWarpSpecializedCooperative,cutlass::gemm::KernelTmaWarpSpecializedPingpong>;
 using Epi=typename cutlass::epilogue::collective::CollectiveBuilder<cutlass::arch::Sm90,cutlass::arch::OpClassTensorOp,Tile,Cluster,cutlass::epilogue::collective::EpilogueTileAuto,float,float,BF,cutlass::layout::ColumnMajor,8,BF,cutlass::layout::ColumnMajor,8,EpiSchedule>::CollectiveOp;
 using Stages=cute::conditional_t<ST==0,cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename Epi::SharedStorage))>,cutlass::gemm::collective::StageCount<ST>>;
 using Main=typename cutlass::gemm::collective::CollectiveBuilder<cutlass::arch::Sm90,cutlass::arch::OpClassTensorOp,BF,cute::conditional_t<PACK,cutlass::layout::ColumnMajor,cutlass::layout::RowMajor>,8,BF,cutlass::layout::ColumnMajor,8,float,Tile,Cluster,Stages,Schedule>::CollectiveOp;
 using Scheduler=cute::conditional_t<SK,cutlass::gemm::StreamKScheduler,cutlass::gemm::PersistentScheduler>;
 using Kernel=cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>,Main,Epi,Scheduler>;
 using Gemm=cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
 int m=x.size(0),n=PACK?w.size(1):w.size(0),k=PACK?w.size(0):w.size(1);auto out=torch::empty({m,n},x.options());
 typename Kernel::StrideA sa;
 if constexpr(PACK) sa=make_stride(_1{},int64_t(n),int64_t(n)*k);
 else sa=make_stride(int64_t(k),_1{},int64_t(n)*k);
 typename Kernel::StrideB sb=make_stride(int64_t(k),_1{},int64_t(m)*k);
 typename Kernel::StrideC sc=make_stride(_1{},int64_t(n),int64_t(m)*n);
 typename Kernel::StrideD sd=sc;
 cutlass::KernelHardwareInfo hw;hw.device_id=x.get_device();hw.sm_count=at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
 typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,{n,m,k,1},{reinterpret_cast<BF*>(w.data_ptr()),sa,reinterpret_cast<BF*>(x.data_ptr()),sb},{{1.f,0.f},reinterpret_cast<BF*>(out.data_ptr()),sc,reinterpret_cast<BF*>(out.data_ptr()),sd},hw};
 args.scheduler.max_swizzle_size=swizzle;
 using RO=decltype(args.scheduler.raster_order);args.scheduler.raster_order=raster==0?RO::AlongM:RO::AlongN;
 Gemm gemm;ck(gemm.can_implement(args),"can_implement");auto bytes=Gemm::get_workspace_size(args);
 auto workspace=torch::empty({static_cast<int64_t>(bytes)},x.options().dtype(at::kByte));auto stream=at::cuda::getCurrentCUDAStream();
 // initialize includes required workspace clears; graph captures them every call.
 ck(gemm.initialize(args,bytes?workspace.data_ptr():nullptr,stream),"initialize");ck(gemm.run(stream),"run");C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
torch::Tensor run(torch::Tensor x,torch::Tensor w,int config,int swizzle,int raster){
 TORCH_CHECK(x.is_cuda()&&w.is_cuda()&&x.device()==w.device());
 TORCH_CHECK(x.is_contiguous()&&w.is_contiguous()&&x.scalar_type()==at::kBFloat16&&w.scalar_type()==at::kBFloat16);
 TORCH_CHECK(x.dim()==2&&w.dim()==2&&x.size(1)==w.size(config<8?0:1));c10::cuda::CUDAGuard guard(x.device());
 switch(config){
 case 0:return invoke<true,64,64,3,1,false,false>(x,w,swizzle,raster);
 case 1:return invoke<true,64,64,0,1,false,false>(x,w,swizzle,raster);
 case 2:return invoke<true,128,64,3,1,true,false>(x,w,swizzle,raster);
 case 3:return invoke<true,128,64,0,1,true,false>(x,w,swizzle,raster);
 case 4:return invoke<true,128,128,0,1,true,false>(x,w,swizzle,raster);
 case 5:return invoke<true,128,64,0,2,true,false>(x,w,swizzle,raster);
 case 6:return invoke<true,128,64,0,1,true,true>(x,w,swizzle,raster);
 case 7:return invoke<true,128,128,0,1,true,true>(x,w,swizzle,raster);
 case 8:return invoke<false,128,64,0,1,true,false>(x,w,swizzle,raster);
 case 9:return invoke<false,128,128,0,1,true,false>(x,w,swizzle,raster);
 default:TORCH_CHECK(false,"config");
 }
}
