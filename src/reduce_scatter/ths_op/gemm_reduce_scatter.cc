//===- gemm_reduce_scatter.cc ------------------------------------- C++ ---===//
//
// Copyright 2023 ByteDance Ltd. and/or its affiliates. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#include "c10/cuda/CUDAGuard.h"
#include "flux/gemm_hparams.h"
#include "flux/gemm_meta.h"
#include "flux/runtime_config.h"
#include "flux/ths_op/ths_op.h"
#include "flux/cuda/cuda_common.h"
#include "flux/flux.h"
#include "flux/op_registry.h"
#include "flux/ths_op/util.h"
#include "flux/args/reduce_scatter.h"
#include <ATen/core/List.h>
#include <ATen/ops/empty.h>
#include <c10/core/DeviceType.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/util/Optional.h>
#include <c10/util/intrusive_ptr.h>
#include <c10/util/logging_is_not_google_glog.h>
#include <cuda_runtime_api.h>
#include <ATen/core/jit_type.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAEvent.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/python.h>
#include "flux/utils.h"
#include "reduce_scatter/ths_op/helper_ops.h"
#include "reduce_scatter/reduce_scatter_barrier_struct.hpp"
#include "flux/ths_op/topo_utils.h"
#include "torch/serialize.h"
// 编译时使用宏FLUX_REDUCE_SCATTERT_WITH_NCCL会启用nccl.h头文件
// e.g: g++ -DFLUX_REDUCE_SCATTERT_WITH_NCCL -c gemm_reduce_scatter.cc
#ifdef FLUX_REDUCE_SCATTERT_WITH_NCCL
#include "nccl.h"
#endif

#ifdef FLUX_SHM_USE_NVSHMEM
#include "nvshmemx.h"
#endif
namespace bytedance::flux::ths_op {
using torch::Tensor;
// torch::CustomClassHolder是pytorch c++扩展中提供的一个类，充当基类，通过继承它，可以将c艹类注册为pytorch的一个自定义类
// 并通过torch::jit::script模块从python调用此类的实例和方法
// 类GemmRS继承torch::CustomClassHolder，GemmRS（Gemm Reduce Scatter）
class GemmRS : public torch::CustomClassHolder {
 private:
 // 基本配置变量
  c10::intrusive_ptr<c10d::ProcessGroup> tp_group; //管理分布式计算中的进程组，用于多GPU和多节点通信
  const int32_t nnodes; // 参与计算的节点数
  const int32_t max_m; // 矩阵乘最大可能的m维度，用于控制矩阵大小
  const int32_t n_dim; // gemm操作的n维度，定义输出矩阵的列数
  const c10::ScalarType input_dtype; // 输入的张量数据类型，float16 or float32
  const c10::ScalarType output_dtype; // 输出的张量数据类型
  const bool transpose_weight; // 布尔，表示是否对权重矩阵进行转置
  const bool fuse_reduction; // 布尔，是否在gemm和reduce-scatter操作中进行融合优化，来减少内存传输开销

 private:
 // 集群和拓扑结构相关变量
  const int32_t rank; // 当前设备在分布式环境中的唯一编号
  const int32_t world_size; // 总的参与计算的设备数目
  const int32_t local_world_size; // 当前节点中设备的数量
  const int32_t local_rank; // 当前设备在本地节点中的编号
  const int32_t node_idx; // 当前设备所在的节点编号

 private:
  // Symmetrically distributed tensor
  // 张量缓冲区
  std::vector<torch::Tensor> output_buffers; // 存储gemm的输出结果的分布式缓冲区
  std::vector<torch::Tensor> reduce_buffers; // 存储reduce-scatter过程中各个部分的数据
  std::vector<torch::Tensor> barrier_buffers; // 用于同步barrier的张量缓冲区
#ifndef FLUX_SHM_USE_NVSHMEM
  // used for the cuda-ipc-barrier
  std::vector<torch::Tensor> sync_buffers; // 未使用NVSHMEM，提供cuda设备间同步的缓冲区（用于cuda-ipc-barrier同步）
#endif
  /**
   * @brief torch::Tensor类型的具体缓冲区，存储单一的矩阵数据和gemm的计算结果
   */
  torch::Tensor output_buffer; // 存储gemm的输出结果的单个张量
  torch::Tensor reduce_buffer; // 存储reduce-scatter过程中各个部分的数据的单个张量
  torch::Tensor barrier_buffer; 
  torch::Tensor gemm_buffer;
  /**
   * @brief std::vector<void *>类型的指针数组，指向不同节点或进程内的缓冲区，便于reduce-scatter操作中的数据传递
   */
  std::vector<void *> output_scatter_ptrs;
  std::vector<void *> barrier_ptrs;
  bool no_nvlink; // 布尔，表示是否不使用nvlink
  int sub_world_size; // 分布式计算中的分区大小，用于优化通信
  // 计算流和事件
  c10::cuda::CUDAStream rs_stream_; // 一个cuda流，用于处理reduce-scatter操作，避免与其他计算流冲突
  cudaEvent_t event_; // cuda事件，用于记录和协调异步操作，确保操作顺序正确
  bool use_1d_ring; // 是否使用1D环形拓扑结构，通常用于多节点间的数据传输优化
  bool use_p2p_read; // 是否启用点对点读取通信方式，某些硬件上可能加速数据传输
  const bool is_fp8_gemm; // 是否用fp8精度的gemm操作

#ifdef FLUX_REDUCE_SCATTERT_WITH_NCCL
  ncclComm_t nccl_comm; 
#endif

/**
 * @brief init_output_buffer函数初始化output_buffer和reduce_buffer
 * 初始化分布式计算过程中所需的输出和中间缓冲区，确保在不同架构、节点数量和数据类型设置下，正确分配内存和指针，以支持后续的 GEMM 和 reduce-scatter 操作
 * 提供了后续操作所需的内存分配和同步配置，output_buffer和reduce_buffers缓冲区存储gemm和reduce-scatter操作的输出和中间结果
 * 根据不同gpu架构、节点数量和NVLINK支持情况动态调整内存大小
 * 设置分布式数据指针，output_scatter_ptrs指向分布式数据片段的位置，便于reduce-scatter操作的数据传递
 * 分布式同步，sync_buffers（在未启用NVSHMEM时）提供了这种同步机制，确保cuda进程间的通信一致性，如果没有sync_buffers，分布式计算可能出现不同步或死锁等问题
 * 1. 分配reduce buffer
 * 为reduce_buffers分配合适大小的内存，用于存储reduce-scatter操作的中间结果，函数根据GPU架构、节点数量和NVLink支持情况动态计算reduce_buffers的尺寸
 * 2. 分配output buffer
 * 分配output_buffers的内存，存储gemm计算的输出。对于Sm80架构且输入类型为bfloat16，为确保兼容性，输出缓冲区大小翻倍(max_m * 2)
 * 3. 初始化output_scatter_ptrs
 * 设置output_scatter_ptrs指针数组，指向每个进程的output_buffers数据指针，便于在reduce-scatter操作中高效访问这些数据
 * 只有属于同一节点的进程指针会被设置，跨节点的进程则置为nullptr，以优化内存访问
 * 4. 同步缓冲区（sync_buffers）初始化
 * 如果未启用NVSHMEM（共享内存加速库），分配sync_buffers用于进程间通信中的同步操作，确保各进程在CUDA上的IPC（进程间通信）同步
 */
  void
  init_output_buffer() {
    // update max_m and allocate buffer
    if (get_arch() == _Sm90{} || no_nvlink || (get_arch() == _Sm80{} && nnodes > 1)) {
      int reduce_m_dim = (get_arch() == _Sm90{})
                             ? (max_m + world_size - 1) / world_size * nnodes * nnodes
                             : max_m;
      this->reduce_buffers =
          flux_create_tensor_list({reduce_m_dim, n_dim}, output_dtype, this->tp_group);
      this->reduce_buffer = this->reduce_buffers[this->local_rank];
    }
    if (get_arch() == _Sm80{} && nnodes > 1 && from_torch_dtype(this->input_dtype) == _BF16{}) {
      // SM80 does not support the fuse reduction for the bfloat16 data type
      // we have to use the float32 global_red instruction when SM80 && nnodes>1 && input_type=bf16
      // Therefore, in this case, here double the size of the output_buffer.
      this->output_buffers =
          flux_create_tensor_list({max_m * 2, n_dim}, output_dtype, this->tp_group);
    } else {
      this->output_buffers = flux_create_tensor_list({max_m, n_dim}, output_dtype, this->tp_group);
    }
    this->output_buffer = this->output_buffers[this->local_rank];
    for (int i = 0; i < world_size; ++i) {
      if (i / this->local_world_size == rank / this->local_world_size) {
        output_scatter_ptrs[i] = this->output_buffers[i % this->local_world_size].data_ptr();
        // only check for ranks on the same node
        TORCH_CHECK(
            output_scatter_ptrs[i] != nullptr, "nullptr buffr of rank " + std::to_string(i));
      } else {
        output_scatter_ptrs[i] = nullptr;
      }
    }
#ifndef FLUX_SHM_USE_NVSHMEM
    this->sync_buffers =
        flux_create_tensor_list({this->world_size}, c10::ScalarType::Int, this->tp_group);
    this->sync_buffers[this->rank].zero_();  // zeros the sync buffer for cuda ipc at the start
#endif
  }

/**
 * @brief 延迟初始化用于同步的barrier缓冲区，分布式计算中，barrier用于确保所有进程或设备在执行到某一步之前都已完成之前的操作，以保证数据一致性和同步
 * 该函数按需创建barrier_buffer，并设置指针以支持分布式环境下的同步
 * 1. 延迟初始化检查
 * 如果buffer_size是0或者barrier_buffer已经定义足够大，无需初始化缓冲区，避免重复申请内存
 * 2. 分配barrier buffers
 * 使用flux_create_tensor_list函数创建barrier_buffers，类型为c10::ScalarType::Byte，大小为buffer_size，并与当前进程组(tp_group)关联
 * 然后将当前进程的barrier_buffer设置为barrier_buffers中对应的缓冲区
 * 3. 设置barrier pointers（barrier_ptrs）
 * 循环遍历所有进程world_size，为barrier_ptrs设置指针，每个进程会检查当前进程与其他进程是否位于同一节点
 *  如果在同一节点，将barrier_ptrs[i]指向该进程的barrier_buffer数据指针，以便同节点的进程之间可以直接访问共享的 barrier_buffers
 *  对于不同节点的进程，将 barrier_ptrs[i] 设置为 nullptr，避免跨节点不必要的资源访问
 * 4. 错误检查
 * TORCH_CHECK 用于确保 barrier_ptrs[i] 的指针非空（nullptr），否则抛出错误。这是一个安全检查，确保同步缓冲区初始化后可以正确访问数据
 * @param buffer_size 
 */
  void
  lazy_init_barrier_buffer(int64_t buffer_size) {
    if ((buffer_size == 0) ||
        (barrier_buffer.defined() && buffer_size <= barrier_buffer.numel())) {
      return;
    }
    this->barrier_buffers =
        flux_create_tensor_list({buffer_size}, c10::ScalarType::Byte, this->tp_group);
    this->barrier_buffer = this->barrier_buffers[this->local_rank];
    for (int i = 0; i < world_size; ++i) {
      if (i / this->local_world_size == rank / this->local_world_size) {
        // 将barrier_buffers中某个数据指针赋值给barrier_ptrs[i]，分布式计算中的一种指针分配方式，便于进程之间共享或访问数据
        barrier_ptrs[i] = this->barrier_buffers[i % this->local_world_size].data_ptr();
        // only check for ranks on the same node
        TORCH_CHECK(barrier_ptrs[i] != nullptr, "nullptr buffr of rank " + std::to_string(i));
      } else {
        barrier_ptrs[i] = nullptr;
      }
    }
  }

  bool
  has_nvlink() {
    return true;
  }

/**
 * @brief 确定分布式环境中是否采用1D环形拓扑结构进行通信优化
 * 通过检测当前GPU的设备名称和进程数（world_size）决定是否适合使用1D环形拓扑
 * @return true 
 * @return false 
 */
  bool
  use_1d_ring_or_not() {
    ensure_nvml_init();
    int devid = at::cuda::current_device();
    std::string devname(get_gpu_device_name(devid));
    if (devname != "NVIDIA L20" && world_size == 8) {
      return false;
    }
    return true;
  }

  bool
  use_p2p_read_or_not() {
    ensure_nvml_init();
    int devid = at::cuda::current_device();
    std::string devname(get_gpu_device_name(devid));
    if (devname != "NVIDIA L20") {
      return true;
    }
    return false;
  }

/**
 * @brief 延迟初始化gemm_buffer，并确保其大小满足需求，执行gemm操作时，需要一个用于储存中间计算结果的缓冲区gemm_buffer
 * 函数按需创建或调整gemm_buffer的大小
 * 
 * @param input 
 * @param buffer_size 
 */
  void
  lazy_init_gemm_buffer(torch::Tensor input, int64_t buffer_size) {
    if (buffer_size <= 0) {
      return;
    }
    buffer_size = (buffer_size + 127) / 128 * 128;
    if (!this->gemm_buffer.defined() || buffer_size > this->gemm_buffer.numel()) {
      auto options = input.options().dtype(c10::ScalarType::Byte);
      this->gemm_buffer = torch::empty({buffer_size}, options);
    }
  }

/**
 * @brief Create a Reduce Scatter Stream object
 * 创建一个cuda流，并指定该流的优先级最高，以便用于reduce-scatter操作
 * cuda流可以让不同的gpu操作独立执行，实现并行化操作
 * 设置高优先级可以确保该流中的任务尽可能快地被执行
 * @return c10::cuda::CUDAStream 
 */
  c10::cuda::CUDAStream
  CreateReduceScatterStream() {
    at::cuda::CUDAGuard guard(at::cuda::current_device());
    cudaStream_t rs_stream = nullptr;
    int least_priority, greatest_priority;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority));
    CUDA_CHECK(cudaStreamCreateWithPriority(&rs_stream, cudaStreamNonBlocking, greatest_priority));
    return at::cuda::getStreamFromExternal(rs_stream, at::cuda::current_device());
  }

 public:
 /**
  * @brief Construct a new Gemm R S object
  * 初始化GemmRS类的对象，配置和准备分布式GEMM操作所需的各项资源
  * 配置了该类所需的各类缓冲区，设备，分布式进程信息，以及并行计算和通信的相关参数
  * @param tp_group_ // 分布式进程组，管理多GPU或多节点环境中的进程
  * @param nnodes  // 节点数量
  * @param max_m //gemm操作中矩阵的最大行数，用于设置缓冲区大小
  * @param n_dim //矩阵的列数，决定输出矩阵的列数
  * @param input_dtype //输入张量的数据类型（如float16 or float32）
  * @param output_dtype //输出张量的数据类型
  * @param transpose_weight //布尔值，是否对权重矩阵进行转置
  * @param fuse_reduction //是否将reduce操作与gemm操作融合，提升计算效率（计算-通信融合）
  */
  GemmRS(
      c10::intrusive_ptr<c10d::ProcessGroup> tp_group_,
      int32_t nnodes,
      int32_t max_m,
      int32_t n_dim,
      c10::ScalarType input_dtype,
      c10::ScalarType output_dtype,
      bool transpose_weight,
      bool fuse_reduction)
      : tp_group(tp_group_),//通过tp_group获取当前进程的rank和world_size，rank是当前设备在分布式环境中的唯一编号，world_size是总的参与计算的设备数目
        nnodes(nnodes),
        max_m(max_m),
        n_dim(n_dim),
        input_dtype(input_dtype),
        output_dtype(output_dtype),
        transpose_weight(transpose_weight),
        fuse_reduction(fuse_reduction),
        rank(tp_group->getRank()),
        world_size(tp_group->getSize()),
        local_world_size(world_size / nnodes),
        local_rank(rank % local_world_size),
        node_idx(rank / local_world_size),
        //初始化指向output_scatter_ptrs和barrier_ptrs的指针数组，数组大小是world_size（总的参与计算的设备数目）
        //所有数组中存放的都有一个指针指向每一个设备，初始值是nullptr，这些指针在后续的分布式操作中用于跨进程的张量通信
        output_scatter_ptrs(world_size, nullptr),
        barrier_ptrs(world_size, nullptr),
        no_nvlink(!has_nvlink()),
        rs_stream_(CreateReduceScatterStream()),  // private stream. never dup with gemm stream
        use_1d_ring(use_1d_ring_or_not()),
        use_p2p_read(use_p2p_read_or_not()),
        is_fp8_gemm(is_fp8_torch_dtype(input_dtype)) {
    TORCH_CHECK(
        rank >= 0 && rank < world_size,
        "invalid rank: " + std::to_string(rank) +
            " and world_size: " + std::to_string(world_size));
    TORCH_CHECK(
        world_size % nnodes == 0,
        "invalid nnodes: world_size[" + std::to_string(world_size) + "] % nnodes[" +
            std::to_string(nnodes) + "] != 0");
    //检查如果启用了fuse_reduction，必须使用float16类型，这是SM80架构的指令限制
    TORCH_CHECK(
        !fuse_reduction || input_dtype == at::ScalarType::Half,
        "Fuse reduction only support float16 type on SM80 due to instruction limitation.");
    this->init_output_buffer();
    CUDA_CHECK(cudaEventCreate(&event_));
#if defined(FLUX_DEBUG)
    if (no_nvlink) {
      LOG(WARNING) << "NvLink is not supported, seems running on a PCI-e machine.";
      ensure_nvml_init();
      int devid = at::cuda::current_device();
      std::string devname(get_gpu_device_name(devid));
      if (devname != "NVIDIA A100 80GB PCIe" && devname != "NVIDIA A800 80GB PCIe") {
        LOG(WARNING) << "Only NVIDIA A100/A800 80GB PCIe is tuned for. got " << devname;
      }
      if (world_size > 4 && world_size != 8) {
        LOG(WARNING) << "Only TensorParallel = 4 or 8 is tuned for. got " << world_size;
      }
      unsigned int gen = get_pcie_gen(devid);
      if (gen != 4) {
        LOG(WARNING) << "only PCI-e 4 version is tuned for. got PCI-e " << gen;
      }
    }
#endif
#ifdef FLUX_REDUCE_SCATTERT_WITH_NCCL
    if (nnodes > 1 && no_nvlink) {
      nccl_comm = topo_utils::create_nccl_comm_with_processgroup(tp_group);
    } else {
      nccl_comm = nullptr;
    }
#endif
  }

/**
 * @brief Destroy the Gemm R S object
 * 释放资源，确保GemmRS对象在销毁时不留下任何GPU资源或通信资源，避免内存泄漏和不必要的资源占用
 */
  ~GemmRS() {
    CUDA_CHECK(cudaEventDestroy(event_));
    CUDA_CHECK(cudaStreamDestroy(rs_stream_));
#ifdef FLUX_REDUCE_SCATTERT_WITH_NCCL
    if (nccl_comm) {
      NCCL_CHECK(ncclCommDestroy(nccl_comm));
    }
#endif
  }

/**
 * @brief Get the gemm meta object
 * 生成并返回一个gemm原数据（meta）配置对象
 * 包含用于执行gemm操作所需的各项参数和设置
 * 该元数据配置包括了数据类型、设备架构、拓扑结构以及计算布局等，目的是根据硬件特性和当前的计算需求调整gemm操作的执行方式
 * @param has_bias 
 * @param fast_accum 
 * @return auto 
 */
  auto
  get_gemm_meta(bool has_bias, bool fast_accum = false) {
    ArchEnum arch = get_arch();
    auto gemm_layout = transpose_weight ? _RRR{}() : _RCR{}();
    //将输入和输出数据类型从pytorch格式转换为内部格式，并创建dt_conf配置对象
    auto input_dtype = from_torch_dtype(this->input_dtype);
    auto output_dtype = from_torch_dtype(this->output_dtype);
    auto dt_conf = make_gemm_dtype_config(
        input_dtype, input_dtype, has_bias ? output_dtype : _Void{}(), output_dtype);

    fast_accum = fast_accum & dt_conf.is_input_fp8();
    bool is_gemm_v2 = ((int)arch < (int)_Sm90{}());
    //创建gemm元数据对象
    auto meta = make_gemm_meta(
        dt_conf,
        arch,
        _ReduceScatter{},
        gemm_layout,
        is_gemm_v2 ? _GemmV2{}() : _GemmV3{}(),
        is_gemm_v2 ? UnifiedImplMeta(make_gemm_v2_meta(fast_accum))
                   : UnifiedImplMeta(make_gemm_v3_meta(fast_accum)),
        make_reduce_scatter_meta(
            this->fuse_reduction,
            nnodes > 1        ? _AcrossNode{}()
            : this->no_nvlink ? _IntraNodePcie{}()
                              : _IntraNode{}()));
    return meta;
  }

/**
 * @brief Get the rt conf object
 * 生成并返回一个运行时配置（runtimeconfig），用于执行gemm操作时的配置验证和参数设置
 * 函数主要检查输入、权重和偏置张量的形状是否符合预期，并根据当前的矩阵参数（行数、列数等）以及配置，创建并返回一个 RuntimeConfig 对象，用于指导 GEMM 计算过程
 * @param input 
 * @param weight 
 * @param bias 
 * @return RuntimeConfig 
 */
  RuntimeConfig
  get_rt_conf(torch::Tensor input, torch::Tensor weight, c10::optional<torch::Tensor> bias) {
    CHECK_INPUT(input, this->input_dtype);
    CHECK_INPUT(weight, this->input_dtype);
    TORCH_CHECK(input.dim() == 2, "input dim is not 2");
    TORCH_CHECK(weight.dim() == 2, "weight dim is not 2");
    int32_t m = input.size(0);
    int32_t k = input.size(1);
    int32_t n = transpose_weight ? weight.size(1) : weight.size(0);

    if (bias.has_value()) {
      CHECK_INPUT(bias.value(), this->output_dtype);
      TORCH_CHECK(bias->dim() == 2, "bias dim is not 2");
      TORCH_CHECK(
          m == bias->size(0),
          "bias dim0 != m: " + std::to_string(bias->size(0)) + " vs " + std::to_string(m));
      TORCH_CHECK(
          n == bias->size(1),
          "bias dim1 != n: " + std::to_string(bias->size(1)) + " vs " + std::to_string(n));
    }

    // row major for streamk, todo: make weight layout an option
    int32_t wk = transpose_weight ? weight.size(0) : weight.size(1);
    FLUX_CHECK_LE(m, this->max_m) << "m-dim greater than maximum possible value";
    FLUX_CHECK_EQ(n, this->n_dim) << "n-dim != expected n_dim";
    FLUX_CHECK_EQ(wk, k) << "weight k-dim mismatch";
    return make_runtime_config(m, n, k, make_reduce_scatter_runtime_config(world_size, nnodes));
  }

/**
 * @brief 函数作用是执行带有可选reduce-scatter操作的gemm计算，函数根据输入的张量、配置参数和硬件特性，选择合适的操作配置和算法（如标准浮点计算或者fp8精度）
 * 并在分布式或本地多gpu环境下高效的运算
 * @param input 
 * @param weight 
 * @param bias 
 * @param input_scale 
 * @param weight_scale 
 * @param output_scale 
 * @param fast_accum 
 * @param hparams 
 */
  void
  forward_gemm_impl(
      torch::Tensor input, //输入矩阵张量（二维的张量，其实就是矩阵）
      torch::Tensor weight,//权重矩阵张量
      c10::optional<torch::Tensor> bias,//可选的偏置项张量
      c10::optional<torch::Tensor> input_scale,//可选的输入缩放因子，用于fp8精度下的缩放操作
      c10::optional<torch::Tensor> weight_scale,
      c10::optional<torch::Tensor> output_scale,
      bool fast_accum,//是否启用快速积累模式
      //可选的gemm参数配置
      //UnifiedGemmHParams是一个与gemm配置相关的参数类
      //c10是pytorch中底层核心库的一部分，是Core Ten的缩写
      //c10提供了跨平台的支持和一组通用的工具，可以在不同编译器、系统架构中更高效的执行pytorch操作
      c10::optional<UnifiedGemmHParams> const &hparams) {
    auto meta = get_gemm_meta(/*has_bias=*/bias.has_value(), fast_accum);//获取用于执行gemm的元数据对象meta，包含数据类型、布局等信息
    auto rt_conf = get_rt_conf(input, weight, bias);//获取运行时配置rt_conf，包含矩阵尺寸和分布式配置等信息
    // get cutlass op
    //获取并初始化cutlass操作对象cutlass_op，cutlass_op是cutlass操作对象的指针
    OpRegistry::OpPtr cutlass_op;
    if (hparams.has_value()) {
      cutlass_op = OpRegistry::instance().get_op(meta, hparams.value());
    } else {
      cutlass_op = OpRegistry::instance().get_op(meta, rt_conf);
    }

    // TODO(houqi.1993) using args instead of envs
    //配置环境变量参数
    //通过环境变量设置一些可调参数，例如块数量（num_blocks），是否使用屏障队列（use_barrier_queue）等，以便根据不同的硬件配置优化性能
    static int num_blocks = get_int_from_env("FLUX_RS_BLOCKS", 12);
    static bool use_barrier_queue = get_bool_from_env("FLUX_RS_USE_BARRIER_QUEUE", false);
    static bool use_gemmk = get_bool_from_env("FLUX_RS_USE_GEMMK", no_nvlink);
    static bool use_cudaMemcpyAsync = get_bool_from_env("FLUX_RS_USE_CUDA_MEMCPY_ASYNC", false);
    static int n_split = get_int_from_env("FLUX_RS_N_SPLIT", 1);
    static bool per_tile_flags = get_bool_from_env("FLUX_RS_PER_TILE_FLAGS", no_nvlink);
    //配置reduce_scatter_args
    ReduceScatterArguments reduce_scatter_args{
        .reduce_scatter_num_blocks = num_blocks,
        .rs_stream = rs_stream_,
        .event = event_,
        .use_barrier_queue = use_barrier_queue,
        .use_gemmk = use_gemmk,
        .per_tile_flags = per_tile_flags,
        .use_cudaMemcpyAsync = use_cudaMemcpyAsync,
        .n_split = n_split,
        .sub_world_size = this->sub_world_size,
#ifdef FLUX_REDUCE_SCATTERT_WITH_NCCL
        .opaque = nccl_comm,
#else
        .opaque = nullptr,
#endif
        .use_1d_ring = use_1d_ring,
        .use_p2p_read = use_p2p_read,
    };
    //设置cuda流
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    //标准浮点计算逻辑（非fp8）
    if (!is_fp8_gemm) {
      FLUX_CHECK(!input_scale.has_value());
      FLUX_CHECK(!weight_scale.has_value());
      FLUX_CHECK(!output_scale.has_value());
      const GemmReduceScatterArguments args{
          .m = rt_conf.m(),
          .n = rt_conf.n(),
          .k = rt_conf.k(),
          .rank = static_cast<int>(this->rank),
          .world_size = static_cast<int>(this->world_size),
          .nnodes = static_cast<int>(this->nnodes),
          .alpha = 1.0f,
          .beta = bias.has_value() ? 1.0f : 0.0f,
          .input = input.data_ptr(),
          .weight = weight.data_ptr(),
          .bias = bias.has_value() ? bias->data_ptr() : nullptr,
          .output_scatter_ptrs = this->output_scatter_ptrs.data(),
          .local_reduce_buffer =
              this->reduce_buffer.defined() ? this->reduce_buffer.data_ptr() : nullptr,
          .barrier_ptrs = this->barrier_ptrs.data(),
          .avail_sms = no_nvlink ? 1 : -1,
          .reduce_scatter_args = reduce_scatter_args};

      // initialize workspace
      int64_t workspace_size = cutlass_op->get_workspace_size(args);
      this->lazy_init_gemm_buffer(input, workspace_size);
      void *workspace = this->gemm_buffer.defined() ? this->gemm_buffer.data_ptr() : nullptr;

      // initialize barrier workspace
      int64_t barrier_workspace_size = cutlass_op->get_barrier_workspace_size(args);
      // * 8 is for corner case reduce_scatter tiles. never mind this won't be a large memory
      barrier_workspace_size = barrier_workspace_size / sizeof(int) * sizeof(PerTileFlags) * 8;
      this->lazy_init_barrier_buffer(barrier_workspace_size);

      if ((fuse_reduction && !(meta.arch() == _Sm90{})) || this->no_nvlink) {
        // need to zero buffers;
        zero_buffers();
      }
      cutlass_op->run(args, workspace, stream);

    } else {
      GemmReduceScatterFp8Arguments fp8_args{
          .m = rt_conf.m(),
          .n = rt_conf.n(),
          .k = rt_conf.k(),
          .rank = static_cast<int>(this->rank),
          .world_size = static_cast<int>(this->world_size),
          .nnodes = static_cast<int>(this->nnodes),
          .alpha = 1.0f,
          .beta = bias.has_value() ? 1.0f : 0.0f,
          .input = input.data_ptr(),
          .weight = weight.data_ptr(),
          .bias = bias.has_value() ? bias->data_ptr() : nullptr,
          .output_scatter_ptrs = this->output_scatter_ptrs.data(),
          .local_reduce_buffer =
              this->reduce_buffer.defined() ? this->reduce_buffer.data_ptr() : nullptr,
          .barrier_ptrs = this->barrier_ptrs.data(),
          .avail_sms = no_nvlink ? 1 : -1,
          .reduce_scatter_args = reduce_scatter_args,
          .Aux = nullptr,
          .Vector = bias.has_value() ? bias->data_ptr() : nullptr,
          .abs_max_Aux = nullptr,
          .abs_max_D = nullptr,
          .scaleA = (float *)(input_scale.has_value() ? input_scale->data_ptr() : nullptr),
          .scaleB = (float *)(weight_scale.has_value() ? weight_scale->data_ptr() : nullptr),
          .scaleC = nullptr,
          .scaleD = (float *)(output_scale.has_value() ? output_scale->data_ptr() : nullptr),
          .scaleAux = nullptr};

      // initialize workspace
      int64_t workspace_size = cutlass_op->get_workspace_size(fp8_args);
      this->lazy_init_gemm_buffer(input, workspace_size);
      void *workspace = this->gemm_buffer.defined() ? this->gemm_buffer.data_ptr() : nullptr;

      // initialize barrier workspace
      int64_t barrier_workspace_size = cutlass_op->get_barrier_workspace_size(fp8_args);
      // * 8 is for corner case reduce_scatter tiles. never mind this won't be a large memory
      barrier_workspace_size = barrier_workspace_size / sizeof(int) * sizeof(PerTileFlags) * 8;
      this->lazy_init_barrier_buffer(barrier_workspace_size);

      if ((fuse_reduction && !(meta.arch() == _Sm90{})) || this->no_nvlink) {
        // need to zero buffers;
        zero_buffers();
      }
      cutlass_op->run(fp8_args, workspace, stream);
    }

  }  // namespace ths_op

/**
 * @brief 执行gemm操作的reduce-scatter阶段，将计算结果按照不同节点进行分配和汇总
 * 
 * @param input 
 * @param weight 
 * @param bias 
 * @param hparams 
 * @return torch::Tensor 
 */
  torch::Tensor
  forward_reduce_scatter_impl(
      torch::Tensor input,
      torch::Tensor weight,
      c10::optional<torch::Tensor> bias,
      c10::optional<UnifiedGemmHParams> hparams) {
    auto meta = get_gemm_meta(/*has_bias=*/bias.has_value());  // fast_accum doesn't matter
    auto rt_conf = get_rt_conf(input, weight, bias);

    // get cutlass op
    OpRegistry::OpPtr cutlass_op;
    if (hparams.has_value()) {
      cutlass_op = OpRegistry::instance().get_op(meta, hparams.value());
    } else {
      cutlass_op = OpRegistry::instance().get_op(meta, rt_conf);
    }

    int m = rt_conf.m();
    int n = rt_conf.n();

    if (((int)get_arch() < (int)_Sm90{}())) {
      auto full_output = this->output_buffer.slice(0, 0, m);
      if (nnodes > 1 && !no_nvlink) {
        // printf("fuse_reduction:%d \n\n", fuse_reduction);
        auto unified_hparams = cutlass_op->get_runtime_gemm_hparams();
        auto tile_shape = unified_hparams.tile_shape();
        auto [tile_M, tile_N, tile_K] = tile_shape;
        int m_rank = m / world_size;
        auto result = torch::empty({m_rank, n}, this->output_buffer.options());
        auto output_to_reduce =
            this->reduce_buffer.slice(0, 0, nnodes * m_rank).view({nnodes, m_rank, this->n_dim});
        bsr_reduce(output_to_reduce, result, tile_M, tile_N);
        return result;
        // return full_output;
      } else if (no_nvlink) {
        int m_per_rank = m / this->world_size;
        auto output_2d =
            output_buffer.slice(0, m_per_rank * this->rank, m_per_rank * (this->rank + 1));
        constexpr int kNumaWorldSize = 4;
        constexpr int kNumaNodes = 2;
        int local_world_size = world_size / nnodes;
        int local_rank = rank % local_world_size;
        int node_id = rank / local_world_size;
        int numa_id = local_rank / kNumaWorldSize;
        int rank_numa_local = local_rank % kNumaWorldSize;
        int rank_prev = (rank_numa_local - 1 + kNumaWorldSize) % kNumaWorldSize;
        rank_prev += numa_id * kNumaWorldSize + node_id * local_world_size;
        int rank_next = (rank_numa_local + 1) % kNumaWorldSize;
        rank_next += numa_id * kNumaWorldSize + node_id * local_world_size;
        int rank_from = numa_id == 0 ? rank_next : rank_prev;
        for (int i = 1; i < nnodes; i++) {
          int reduce_unused_segment = (rank_from + kNumaNodes + i * local_world_size) % world_size;
          auto segment_other_node = reduce_buffer.slice(
              0, m_per_rank * reduce_unused_segment, m_per_rank * (reduce_unused_segment + 1));
          output_2d.add_(segment_other_node);
        }
        return output_2d;
      } else {
        int local_world_size = world_size / nnodes;
        if (fuse_reduction) {
          auto length = m / world_size;
          // return this->output_buffer.slice(0, rank * length, (rank + 1) * length).unsqueeze(0);
          return this->output_buffer.slice(0, 0, length).unsqueeze(0);
        } else {
          auto output_4d = full_output.view({nnodes, local_world_size, m / world_size, n});
          auto output = output_4d.sum(1);  // (nnodes,m_rank,n)
          return output;
        }
      }
    } else if (meta.arch() == _Sm90{}) {
      int reduce_m_dim = m / world_size * nnodes * nnodes;
      auto full_output = this->reduce_buffer.slice(0, 0, reduce_m_dim);
      auto output_4d = full_output.view({nnodes, nnodes, m / world_size, n});
      if (nnodes == 1) {
        auto output = output_4d[node_idx].sum(0);  // (m_rank,n)
        return output;
      } else {
        int m_rank = m / world_size;
        auto output = torch::empty({m_rank, n}, output_buffer.options());
        auto unified_hparams = cutlass_op->get_runtime_gemm_hparams();
        auto tile_shape = unified_hparams.tile_shape();
        auto [tile_M, tile_N, tile_K] = tile_shape;
        bsr_reduce(output_4d[node_idx], output, tile_M, tile_N);
        return output;
      }
    } else {
      TORCH_CHECK(false, "unsupported arch:" + std::string(enum_to_string(meta.arch())));
    }
  }

/**
 * @brief 封装了gemm操作、同步屏障和reduce-scatter操作的完整执行流程
 * 接收输入张量、权重张量、偏置张量、缩放因子等参数，并调用一系列函数来完成矩阵乘法操作
 * 同时确保在分布式环境下的同步和结果聚合
 * 
 * @param input 
 * @param weight 
 * @param bias 
 * @param input_scale 
 * @param weight_scale 
 * @param output_scale 
 * @param fast_accum 
 * @param hparams 
 * @return torch::Tensor 
 */
  torch::Tensor
  forward_impl(
      torch::Tensor input,
      torch::Tensor weight,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      c10::optional<torch::Tensor> output_scale,
      bool fast_accum,
      c10::optional<UnifiedGemmHParams> const &hparams) {
    forward_gemm_impl(
        input, weight, bias, input_scale, weight_scale, output_scale, fast_accum, hparams);
    forward_barrier(input, weight, bias);
    return forward_reduce_scatter_impl(input, weight, bias, hparams);
  }

/**
 * @brief 执行gemm操作
 * 对forward_gemm_impl的一个简单封装，将输入张量、权重张量和可选的偏置和缩放因子传递给forward_gemm_impl，完成矩阵乘法操作
 * forward_gemm作用是简化接口，将可选的hparams参数设置为nullopt，默认情况下不使用特定的gemm参数配置
 * 
 * @param input 
 * @param weight 
 * @param bias 
 * @param input_scale 
 * @param weight_scale 
 * @param output_scale 
 * @param fast_accum 
 */
  void
  forward_gemm(
      torch::Tensor input,
      torch::Tensor weight,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      c10::optional<torch::Tensor> output_scale,
      bool fast_accum) {
    return forward_gemm_impl(
        std::move(input),
        std::move(weight),
        std::move(bias),
        input_scale,
        weight_scale,
        output_scale,
        fast_accum,
        c10::nullopt);
  }

/**
 * @brief 分布式或多gpu环境下同步各个计算节点，确保每个节点在进行下一步操作之前都完成了当前的计算
 * 这是一个同步屏障函数，用于在gemm计算后确保所有节点的数据状态一致
 * 
 * @param input 
 * @param weight 
 * @param bias 
 */
  void
  forward_barrier(torch::Tensor input, torch::Tensor weight, c10::optional<torch::Tensor> bias) {
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    if (get_arch() == _Sm90{} and nnodes == 1) {
      // only local reduce, skip nvshmem barrier
    } else {
#ifdef FLUX_SHM_USE_NVSHMEM
      flux_barrier_all_on_stream(stream);
#else
      flux_barrier_all_on_stream(stream, this->sync_buffers, this->rank);
#endif
    }
  }

/**
 * @brief 执行reduce-scatter操作
 * 调用forward_reduce_scatter_impl（简化外部调用时的参数传递）
 * @param input 
 * @param weight 
 * @param bias 
 * @return torch::Tensor 
 */
  torch::Tensor
  forward_reduce_scatter(
      torch::Tensor input, torch::Tensor weight, c10::optional<torch::Tensor> bias) {
    return forward_reduce_scatter_impl(
        std::move(input), std::move(weight), std::move(bias), c10::nullopt);
  }

/**
 * @brief 执行完整的gemm计算的封装
 * 调用forward_impl函数，并返回计算结果
 * @param input 
 * @param weight 
 * @param bias 
 * @param input_scale 
 * @param weight_scale 
 * @param output_scale 
 * @param fast_accum 
 * @return torch::Tensor 
 */
  torch::Tensor
  forward(
      torch::Tensor input,
      torch::Tensor weight,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      c10::optional<torch::Tensor> output_scale,
      bool fast_accum) {
    return forward_impl(
        std::move(input),
        std::move(weight),
        std::move(bias),
        std::move(input_scale),
        std::move(weight_scale),
        std::move(output_scale),
        fast_accum,
        c10::nullopt);
  }

/**
 * @brief 将类中的几个缓存区（output_buffer, barrier_buffer, reduce_buffer）置为0
 * 并在分布式环境中同步各节点，确保在新的周期中不会受到之前数据的干扰
 * 
 */
  void
  zero_buffers() {
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    if (this->output_buffer.defined()) {
      this->output_buffer.zero_();
    }
    if (this->barrier_buffer.defined()) {
      this->barrier_buffer.zero_();
    }
    if (this->reduce_buffer.defined()) {
      this->reduce_buffer.zero_();
    }
#ifdef FLUX_SHM_USE_NVSHMEM
    flux_barrier_all_on_stream(stream);
#else
    flux_barrier_all_on_stream(stream, this->sync_buffers, this->rank);
#endif
    if (!no_nvlink) {
      c10::cuda::stream_synchronize(stream);
    }
  }

/**
 * @brief profiling函数用于在启用了nvshmem库的环境中对gemm操作进行性能分析
 * 该函数根据多个参数配置执行gemm操作，测量计算时间，并通过对不同配置的比较来记录最佳配置
 * profiling函数目标是确定执行gemm操作的最优参数配置，提高性能
 * 
 * @param input 
 * @param weight 
 * @param bias 
 * @param input_scale 
 * @param weight_scale 
 * @param output_scale 
 * @param fast_accum 
 * @param opt_ctx 
 * @return torch::Tensor 
 */
  torch::Tensor
  profiling(
      torch::Tensor input,
      torch::Tensor weight,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      c10::optional<torch::Tensor> output_scale,
      bool fast_accum,
      c10::intrusive_ptr<ProfilingContext> opt_ctx) {
#ifdef FLUX_SHM_USE_NVSHMEM
    auto meta = unify_type(this->get_gemm_meta(/*has_bias=*/bias.has_value(), fast_accum));
    auto rt_conf = this->get_rt_conf(input, weight, bias);
    ProfilingContext tmp_ctx("__tmp__");
    ProfilingContext *ctx = opt_ctx == nullptr ? &tmp_ctx : opt_ctx.get();

    auto filter_hparams = [&](UnifiedGemmHParams const &hparams) { return true; };

    auto elapsed_tensor = torch::empty({}, weight.options().dtype(c10::ScalarType::Float));
    auto reduced_elapsed_tensor = elapsed_tensor.clone();

    OpRegistry::instance().visit_hparams(
        [&](UnifiedGemmHParams const &hparams) {
          if (not filter_hparams(hparams)) {
            return;
          }
          // filter non-consistent hparams
          constexpr int warm_iters = 5;
          constexpr int iters = 10;
          float total_elapsed = 0;

          auto stream = c10::cuda::getCurrentCUDAStream();
          flux_barrier_all_on_stream(stream);
          c10::cuda::stream_synchronize(stream);
          for (int iter = 0; iter < warm_iters + iters; ++iter) {
            GpuTimer timer;
            timer.start(stream);
            auto output [[maybe_unused]] = this->forward_impl(
                input, weight, bias, input_scale, weight_scale, output_scale, fast_accum, hparams);
            timer.stop();
            if (iter >= warm_iters) {
              total_elapsed += timer.elapsed_millis();
            }
          }

          // Avoid GPU frequency adjustment
          flux_barrier_all_on_stream(stream);
          c10::cuda::stream_synchronize(stream);
          sleep(1);

          float avg_elapsed = int(total_elapsed / iters * 1000) / 1000.0;
          elapsed_tensor.copy_(torch::full({}, avg_elapsed));

          nvshmemx_float_max_reduce_on_stream(
              NVSHMEM_TEAM_WORLD,
              static_cast<float *>(reduced_elapsed_tensor.data_ptr()),
              static_cast<float const *>(elapsed_tensor.data_ptr()),
              1,
              stream);

          float reduce_elapsed = reduced_elapsed_tensor.item().toFloat();
          ctx->add(meta, rt_conf, hparams, reduce_elapsed);
        },
        meta);

    auto best_hparams = ctx->record_best(meta, rt_conf);
    return this->forward_impl(
        std::move(input),
        std::move(weight),
        std::move(bias),
        std::move(input_scale),
        std::move(weight_scale),
        std::move(output_scale),
        fast_accum,
        std::move(best_hparams));
#else
    FLUX_CHECK(false) << "only support profiling when nvshmem is enabled";
    return torch::Tensor();
#endif
  }

/**
 * @brief 确保分布式计算的拓扑结构（topology）已被初始化
 * 通过调用拓扑工具topo_utils中的检查和初始化函数，确认当前拓扑配置是否已就绪，从而保证通信和数据传输的正确性
 * 
 */
  void
  _ensure_topo_initialized() {
    if (!topo_utils::is_topo_initialized()) {
      topo_utils::initialize_topo(this->tp_group);
    }
  }
};  // namespace flux

namespace py = pybind11;//通过pybind11将c++类GemmRS注册到python中，python可以直接调用该类及其方法，引用pybind11的命名空间，简化后续代码书写

/**
 * @brief 静态的lambda表达式，用于自动将GemmRS类绑定到python
 * 
 */
static int _register_gemm_rs_ops [[maybe_unused]] = []() {
  ThsOpsInitRegistry::instance().register_one("gemm_reduce_scatter", [](py::module &m) {
    py::class_<GemmRS>(m, "GemmRS")
        .def(
            py::init([](c10::intrusive_ptr<c10d::ProcessGroup> tp_group,
                        int32_t nnodes,
                        int32_t max_m,
                        int32_t n_dim,
                        py::object py_input_dtype,
                        py::object py_output_dtype,
                        bool transpose_weight,
                        bool fuse_reduction) {
              auto input_dtype = torch::python::detail::py_object_to_dtype(py_input_dtype);
              auto output_dtype = py_output_dtype.is(py::none())
                                      ? input_dtype
                                      : torch::python::detail::py_object_to_dtype(py_output_dtype);

              return new GemmRS(
                  tp_group,
                  nnodes,
                  max_m,
                  n_dim,
                  input_dtype,
                  output_dtype,
                  transpose_weight,
                  fuse_reduction);
            }),
            py::arg("tp_group"),
            py::arg("nnodes"),
            py::arg("max_m"),
            py::arg("n_dim"),
            py::arg("input_dtype"),
            py::arg("output_dtype") = py::none(),
            py::arg("transpose_weight") = false,
            py::arg("fuse_reduction") = false)
        .def("zero_buffers", &GemmRS::zero_buffers)
        .def(
            "forward",
            &GemmRS::forward,
            py::arg("input"),
            py::arg("weight"),
            py::arg("bias") = py::none(),
            py::arg("input_scale") = py::none(),
            py::arg("weight_scale") = py::none(),
            py::arg("output_scale") = py::none(),
            py::arg("fast_accum") = false)
        .def(
            "forward_gemm",
            &GemmRS::forward_gemm,
            py::arg("input"),
            py::arg("weight"),
            py::arg("bias") = py::none(),
            py::arg("input_scale") = py::none(),
            py::arg("weight_scale") = py::none(),
            py::arg("output_scale") = py::none(),
            py::arg("fast_accum") = false)
        .def(
            "forward_barrier",
            &GemmRS::forward_barrier,
            py::arg("input"),
            py::arg("weight"),
            py::arg("bias") = py::none())
        .def(
            "forward_reduce_scatter",
            &GemmRS::forward_reduce_scatter,
            py::arg("input"),
            py::arg("weight"),
            py::arg("bias") = py::none())
        .def(
            "profiling",
            &GemmRS::profiling,
            py::arg("input"),
            py::arg("weight"),
            py::arg("bias") = py::none(),
            py::arg("input_scale") = py::none(),
            py::arg("weight_scale") = py::none(),
            py::arg("output_scale") = py::none(),
            py::arg("fast_accum") = false,
            py::arg("prof_ctx") = nullptr);
  });
  return 0;
}();

}  // namespace bytedance::flux::ths_op
