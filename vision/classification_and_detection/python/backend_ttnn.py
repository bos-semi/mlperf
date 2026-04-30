"""
TTNN backend for MLPerf Inference
Supports: resnet50, vit, yolov8s

This file implements a modular backend for MLPerf inference using TTNN.
Each model type (resnet50, vit, yolov8s) is handled by a dedicated runner class.
The BackendTTNN class acts as a unified interface, delegating all model-specific logic to the appropriate runner.
"""

import os
import sys

os.environ.setdefault("LOGURU_LEVEL", "ERROR")
os.environ.setdefault("TT_LOGGER_LEVEL", "ERROR")
_TT_METAL_ROOT = os.environ.get("TT_METAL_HOME", "/home/bos_docker/work/tt-metal")
if _TT_METAL_ROOT not in sys.path:
    sys.path.insert(0, _TT_METAL_ROOT)

import importlib
import threading
from abc import ABC, abstractmethod

import numpy as np
import torch
import ttnn

import backend


# ---------------------------------------------------------------------------------------------------------------------------
# Dummy outputs
# ---------------------------------------------------------------------------------------------------------------------------
def _dummy_classification(batch: int):
    return [np.zeros((batch, 1000), dtype=np.float32)]


def _dummy_detection():
    return [
        np.array([0]),
        np.zeros((1, 1, 4), dtype=np.float32),
        np.zeros((1, 1), dtype=np.float32),
        np.zeros((1, 1), dtype=np.float32),
    ]


# ---------------------------------------------------------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------------------------------------------------------
class TTNNDatasetWrapper:
    """
    Wraps the dataset to allow per-sample preprocessing and caching for TTNN models.
    This enables efficient sample preparation and batch assembly for inference.
    """
    def __init__(self, base_dataset, backend_obj):
        self._base = base_dataset
        self._backend = backend_obj
        self._cache = {}

    def __getattr__(self, name):
        return getattr(self._base, name)

    def load_query_samples(self, sample_list):
        self._base.load_query_samples(sample_list)
        for s in sample_list:
            numpy_img = self._base.image_list_inmemory[s]
            self._cache[s] = self._backend.prepare_sample(numpy_img)

    def unload_query_samples(self, sample_list):
        self._base.unload_query_samples(sample_list)
        if sample_list:
            for s in sample_list:
                self._cache.pop(s, None)
        else:
            self._cache.clear()

    def get_samples(self, id_list):
        items = [self._cache[i] for i in id_list]
        labels = self._base.label_list[id_list]
        return self._backend.make_batch_input(items), labels


# ---------------------------------------------------------------------------------------------------------------------------
# Model runner base
# ---------------------------------------------------------------------------------------------------------------------------
class BaseTTNNModelRunner(ABC):
    """
    Abstract base class for all TTNN model runners.
    Defines the required interface for model-specific runners (load, prepare_sample, make_batch_input, predict, etc).
    """
    def __init__(self, device, batch_size, use_trace=True, use_2cq=False):
        self.device = device
        self.batch_size = batch_size
        self.use_trace = use_trace
        self.use_2cq = use_2cq

    @property
    @abstractmethod
    def model_type(self):
        pass

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def prepare_sample(self, numpy_img: np.ndarray):
        pass

    @abstractmethod
    def make_batch_input(self, items: list):
        pass

    @abstractmethod
    def predict(self, x):
        pass

    @abstractmethod
    def dummy_output(self):
        pass

    def release(self):
        pass


class ClassificationModelRunner(BaseTTNNModelRunner):
    """
    Base class for classification models (e.g., ResNet, ViT).
    Provides a default dummy output for classification tasks.
    """
    def dummy_output(self):
        return _dummy_classification(1)


class DetectionModelRunner(BaseTTNNModelRunner):
    """
    Base class for detection models (e.g., YOLO).
    Provides a default dummy output for detection tasks.
    """
    def dummy_output(self):
        return _dummy_detection()


# ---------------------------------------------------------------------------------------------------------------------------
# ResNet50 runner
# ---------------------------------------------------------------------------------------------------------------------------
class ResNet50Runner(ClassificationModelRunner):
    """
    Model runner for ResNet50.
    Handles model loading, sample preparation, batch assembly, and inference for ResNet50.
    """
    model_type = "resnet50"

    def __init__(self, device, batch_size, use_trace=True, use_2cq=False):
        super().__init__(device, batch_size, use_trace, use_2cq)
        self._model = None

    def load(self):
        # Load the ResNet50 model using the provided utility function.
        from models.bos_model.demo.model_task.classification.model_utils import get_model

        self._model = get_model(
            self.device,
            "microsoft/resnet-50",
            self.batch_size,
            self.use_trace,
            self.use_2cq,
            [True, False],
        )
        if self.use_trace:
            self._model.trace_capture()

    def prepare_sample(self, numpy_img: np.ndarray):
        # Convert numpy image to folded TTNN host tensor for ResNet50.
        from models.bos_model.resnet50.tt.Utils import fold_like_ttnn

        sample = torch.from_numpy(numpy_img).to(torch.bfloat16).unsqueeze(0)
        tt_host = fold_like_ttnn(
            sample, stride_h=2, stride_w=2, pad_c=1, pad_h=3, pad_w=3
        )
        return tt_host

    def make_batch_input(self, items: list):
        # Assemble a batch of folded tensors into a TTNN host tensor.
        if len(items) != self.batch_size:
            return None
        batched = torch.cat(items, dim=0)
        return ttnn.from_torch(
            batched, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
        )

    def predict(self, tt_host):
        # Run inference on the device and return the output as numpy array.
        if self.use_trace:
            if self.use_2cq:
                # 2CQ synchronous path: submit async then sync for correctness.
                host_out = self.predict_async(tt_host)
                ttnn.synchronize_device(self.device)
                return self._collect(host_out)
            else:
                # 1CQ trace path: direct copy to L1 then blocking execute.
                ttnn.copy_host_to_device_tensor(tt_host, self._model.tt_image_res)
                ttnn.execute_trace(self.device, self._model.tid, cq_id=0, blocking=False)
                tt_out = ttnn.reshape(self._model.tt_output_res, (self.batch_size, 1, 1000))
                out = ttnn.to_torch(ttnn.from_device(tt_out, blocking=True))
                return [out[:, 0, :].to(torch.float32).numpy()]
        else:
            # TODO: Implement non-trace path if needed. 1CQ and 2CQ paths need separate handling.
            pass

    def predict_async(self, tt_host):
        # Submit 2CQ inference (non-blocking).
        # CQ1 DMA of tt_host to DRAM overlaps with previous trace on CQ0.
        # Immediately queues a cpu-read on CQ0 (after trace).
        # Returns the host tensor handle; fill completes after synchronize_device().
        self._model._inference(tt_host)
        tt_out = ttnn.reshape(self._model.tt_output_res, (self.batch_size, 1, 1000))
        return tt_out.cpu(blocking=False, cq_id=0)

    def _collect(self, host_out):
        # Convert a host tensor handle (from predict_async) to numpy result.
        out = ttnn.to_torch(host_out)
        return [out[:, 0, :].to(torch.float32).numpy()]

    def release(self):
        # Release trace resources if used.
        if self.use_trace and self._model is not None and hasattr(self._model, "tid"):
            ttnn.release_trace(self.device, self._model.tid)


# ---------------------------------------------------------------------------------------------------------------------------
# ViT runner
# ---------------------------------------------------------------------------------------------------------------------------
class ViTRunner(ClassificationModelRunner):
    """
    Model runner for ViT (Vision Transformer).
    Handles model loading, sample preparation, batch assembly, and inference for ViT.
    """
    model_type = "vit"

    def __init__(self, device, batch_size, use_trace=True, use_2cq=False):
        super().__init__(device, batch_size, use_trace, use_2cq)
        self._vit = None
        self._input_l1 = None
        self._output_l1 = None
        self._trace_id = None
        self._spec = None
        # 2CQ 5-event pipeline state
        self._dram_input = None
        self._first_op_event = None      # CQ0 event after reshard (gates CQ1 upload of next batch)
        self._read_event = None          # CQ1 event after DtoH (gates CQ0 L1�DRAM reuse)
        self._pending_last_op_event = None  # CQ0 event after L1�DRAM (triggers CQ1 DtoH)
        self._pending_output_dram = None    # DRAM tensor waiting for DtoH
        self._pending_output_drams = []     # keep-alive until synchronize

    def load(self):
        # Load the ViT model and perform warmup and trace capture.
        from models.bos_model.demo.model_task.classification.model_utils import get_model

        self._vit = get_model(
            self.device,
            "google/vit-base-patch16-224",
            self.batch_size,
            self.use_trace,
            self.use_2cq,
            [True, False],
        )

        self._warmup()
        self._trace_capture()
        if self.use_2cq:
            ttnn.synchronize_device(self.device)
            self._setup_2cq()

    def prepare_sample(self, numpy_img: np.ndarray):
        # Pre-apply permute+pad+reshape at load time so make_batch_input only needs
        # torch.cat + ttnn.from_torch (eliminating ~0.5-1ms CPU cost from the hot path).
        patch_size = self._vit.patch_size  # 16
        sample = torch.from_numpy(numpy_img).to(torch.bfloat16).unsqueeze(0)  # [1, 3, H, W]
        sample = torch.permute(sample, (0, 2, 3, 1))                            # [1, H, W, 3]
        sample = torch.nn.functional.pad(sample, (0, 1, 0, 0, 0, 0, 0, 0))    # [1, H, W, 4]
        _, img_h, img_w, _ = sample.shape
        sample = sample.reshape(1, img_h, img_w // patch_size, 4 * patch_size)  # [1, H, 14, 64]
        return sample

    def make_batch_input(self, items: list):
        # Assemble a batch of pre-processed tensors; only ttnn.from_torch remains on the hot path.
        if len(items) != self.batch_size:
            return None
        batched = torch.cat(items, dim=0)  # [batch, H, 14, 64]
        return ttnn.from_torch(batched, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)

    def predict(self, tt_host):
        # Run inference using the captured trace and return the output as numpy array.
        if self.use_trace:
            if self.use_2cq:
                # 2CQ synchronous path: predict_async returns None (1-batch delay),
                # flush_pipeline issues DtoH and returns the actual host tensor.
                _ = self.predict_async(tt_host)
                host_out = self.flush_pipeline()
                ttnn.synchronize_device(self.device)
                self.post_synchronize()
                return self._collect(host_out)
            else:
                # 1CQ path: direct host�L1 copy + blocking trace.
                ttnn.copy_host_to_device_tensor(tt_host, self._input_l1)
                ttnn.execute_trace(self.device, self._trace_id, cq_id=0, blocking=False)
                result = ttnn.from_device(self._output_l1, blocking=True)
                return [ttnn.to_torch(result)[:, 0, :1000].to(torch.float32).numpy()]
        else:
            # TODO: Implement non-trace path if needed. 1CQ and 2CQ paths need separate handling.
            pass

    def _warmup(self):
        # Run dummy inference to initialize kernels and memory configs.
        dummy = torch.rand([self.batch_size, 3, 224, 224], dtype=torch.bfloat16)
        tt_inputs_host = self._vit._prepare_inputs(dummy)

        self._input_l1 = tt_inputs_host.to(self.device, self._vit.input_l1_mem_config)
        self._spec = self._input_l1.spec
        out_l1 = self._vit._vit_device_func(self._input_l1)
        _ = ttnn.from_device(out_l1, blocking=True)
        out_l1.deallocate(force=True)

        self._input_l1 = tt_inputs_host.to(self.device, self._vit.input_l1_mem_config)
        out_l1 = self._vit._vit_device_func(self._input_l1)
        _ = ttnn.from_device(out_l1, blocking=True)

    def _trace_capture(self):
        # Capture a trace for fast repeated inference.
        dummy = torch.rand([self.batch_size, 3, 224, 224], dtype=torch.bfloat16)
        tt_inputs_host = self._vit._prepare_inputs(dummy)

        self._input_l1 = tt_inputs_host.to(self.device, self._vit.input_l1_mem_config)
        trace_addr = self._input_l1.buffer_address()

        if self._output_l1 is not None:
            self._output_l1.deallocate(force=True)

        self._trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)
        self._output_l1 = self._vit._vit_device_func(self._input_l1)
        self._input_l1 = ttnn.allocate_tensor_on_device(self._spec, self.device)
        ttnn.end_trace_capture(self.device, self._trace_id, cq_id=0)

        if trace_addr != self._input_l1.buffer_address():
            raise RuntimeError(
                f"ViT trace capture failed: address mismatch "
                f"(expected {trace_addr}, got {self._input_l1.buffer_address()})"
            )

    def _setup_2cq(self):
        # Create a DRAM HEIGHT_SHARDED staging buffer for 2CQ pipeline.
        # ViT host input shape after _prepare_inputs: [batch, 224, 14, 64], ROW_MAJOR, bfloat16.
        dummy_torch = torch.zeros([self.batch_size, 224, 14, 64], dtype=torch.bfloat16)
        dummy_host = ttnn.from_torch(dummy_torch, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

        dram_grid = self.device.dram_grid_size()
        total_height = dummy_host.volume() // dummy_host.shape[-1]  # batch * 224 * 14
        last_dim = dummy_host.shape[-1]  # 64

        # Find the largest divisor of total_height that is <= dram_grid.x for even sharding.
        dram_cores = dram_grid.x
        while dram_cores > 1 and total_height % dram_cores != 0:
            dram_cores -= 1

        dram_shard_spec = ttnn.ShardSpec(
            ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_cores - 1, 0))}
            ),
            [total_height // dram_cores, last_dim],
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        dram_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, dram_shard_spec
        )
        self._dram_input = ttnn.allocate_tensor_on_device(
            dummy_host.shape, dummy_host.dtype, dummy_host.layout, self.device, dram_mem_config
        )
        # JIT warmup: compile the L1�DRAM and tile�ROW_MAJOR kernels before the live pipeline runs.
        _warmup_dram_tile = ttnn.to_memory_config(self._output_l1, ttnn.DRAM_MEMORY_CONFIG)
        _warmup_dram_rm = ttnn.to_layout(_warmup_dram_tile, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _warmup_dram_tile.deallocate(force=True)
        _warmup_dram_rm.deallocate(force=True)
        ttnn.synchronize_device(self.device)
        # Initialize pipeline events (pre-fired so the first predict_async proceeds immediately).
        self._first_op_event = ttnn.record_event(self.device, 0)
        self._read_event = ttnn.record_event(self.device, 1)
        self._pending_last_op_event = None

    def predict_async(self, tt_host):
        # 5-event full pipeline matching test_e2e_trace2cq (1-batch delay).
        # Returns: host tensor for the PREVIOUS batch (None on the first call).
        # After the final batch, call flush_pipeline() to retrieve the last result.
        #
        # CQ1 per call i+1: [wait(foe_i)][up_i+1][we_i+1][wait(loe_i)][DtoH_i][re_i]
        # CQ0 per call i+1: [wait(we_i+1)][reshard_i+1][foe_i+1][trace_i+1][wait(re_i)][L1�DRAM_TILE_i+1][untile_i+1][loe_i+1]
        # � DtoH_i (ROW_MAJOR) overlaps with trace_i+1; untile runs on device (fast) instead of CPU (slow).

        # CQ1: upload current batch (overlaps previous trace on CQ0)
        ttnn.wait_for_event(1, self._first_op_event)
        ttnn.copy_host_to_device_tensor(tt_host, self._dram_input, cq_id=1)
        write_event = ttnn.record_event(self.device, 1)

        # CQ1: DtoH previous batch (queued AFTER upload so upload runs first,
        #      allowing it to overlap with the current trace on CQ0)
        host_prev = None
        if self._pending_last_op_event is not None:
            ttnn.wait_for_event(1, self._pending_last_op_event)
            host_prev = ttnn.from_device(self._pending_output_dram, blocking=False, cq_id=1)
            self._read_event = ttnn.record_event(self.device, 1)

        # CQ0: reshard + trace + L1�DRAM
        ttnn.wait_for_event(0, write_event)
        self._input_l1 = ttnn.reshard(self._dram_input, self._vit.input_l1_mem_config, self._input_l1)
        self._first_op_event = ttnn.record_event(self.device, 0)
        ttnn.execute_trace(self.device, self._trace_id, cq_id=0, blocking=False)
        ttnn.wait_for_event(0, self._read_event)  # ensure prev output_dram is safe to reuse
        # Move L1 TILE � DRAM TILE � DRAM ROW_MAJOR on device.
        # Device-side untile is ~0.1ms vs CPU-side ttnn.to_torch untile ~1.6ms.
        output_dram_tile = ttnn.to_memory_config(self._output_l1, ttnn.DRAM_MEMORY_CONFIG)
        output_dram = ttnn.to_layout(output_dram_tile, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self._pending_last_op_event = ttnn.record_event(self.device, 0)

        self._pending_output_dram = output_dram
        # Keep refs to both tensors; post_synchronize() releases them after device sync.
        self._pending_output_drams.append(output_dram_tile)
        self._pending_output_drams.append(output_dram)
        return host_prev

    def flush_pipeline(self):
        # Issue DtoH for the last batch and return its host tensor.
        # Must be called once after the final predict_async() in a pipeline run.
        if self._pending_last_op_event is None:
            return None
        ttnn.wait_for_event(1, self._pending_last_op_event)
        host_last = ttnn.from_device(self._pending_output_dram, blocking=False, cq_id=1)
        self._read_event = ttnn.record_event(self.device, 1)
        self._pending_last_op_event = None
        return host_last

    def post_synchronize(self):
        # Release DRAM output tensors after synchronize_device() confirms all DtoH is done.
        self._pending_output_drams.clear()

    def _collect(self, host_out):
        # Convert a host tensor handle (from predict_async) to numpy result.
        result = ttnn.to_torch(host_out)
        return [result[:, 0, :1000].to(torch.float32).numpy()]

    def release(self):
        # Release trace resources if used.
        if self.use_trace and self._trace_id is not None:
            ttnn.release_trace(self.device, self._trace_id)


# ---------------------------------------------------------------------------------------------------------------------------
# YOLOv8s runner
# ---------------------------------------------------------------------------------------------------------------------------
class YoloV8sRunner(DetectionModelRunner):
    """
    Model runner for YOLOv8s object detection.
    Handles model loading, sample preparation, batch assembly, and inference for YOLOv8s.
    """
    model_type = "yolov8s"

    def __init__(self, device, batch_size, use_trace=True, use_2cq=False):
        super().__init__(device, batch_size, use_trace, use_2cq)
        self._image_size = 256
        self._golden_shapes = [
            [1, 144, self._image_size // 8, self._image_size // 8],
            [1, 144, self._image_size // 16, self._image_size // 16],
            [1, 144, self._image_size // 32, self._image_size // 32],
        ]
        self._class_map = [
             1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 13, 14, 15, 16, 17, 18, 19,
            20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
            41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
            60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
            82, 84, 85, 86, 87, 88, 89, 90,
        ]
        self._model = None
        self._decode = None
        self._nms = None
        self._trace_id = None
        self._image_res = None

    def load(self):
        # Load the YOLOv8s model, decoder, and NMS utility. Capture trace if enabled.
        from models.bos_model.yolov8s.yolov8s import YoloV8, Detect
        from models.bos_model.yolov8s.configs.yolov8s_256x256 import layer_configs

        self._model = YoloV8(
            device=self.device,
            image_shape=(self._image_size, self._image_size),
            in_channels=3,
            num_classes=80,
            layer_configs=layer_configs,
        )
        self._model.eval()

        self._decode = Detect(nc=80, ch=[128, 256, 512])
        self._decode.eval()

        try:
            from ultralytics.utils.nms import non_max_suppression
        except ImportError:
            from ultralytics.utils import ops
            non_max_suppression = ops.non_max_suppression
        self._nms = non_max_suppression

        if self.use_trace:
            self._trace_capture()

    def prepare_sample(self, numpy_img: np.ndarray):
        # Convert numpy image to TTNN host tensor for YOLOv8s, including resizing and normalization.
        from models.bos_model.yolov8s.utilities.utility_functions import setup_l1_sharded_input

        sample = torch.from_numpy(numpy_img).to(torch.bfloat16).unsqueeze(0)

        tf = sample.to(torch.float32)
        if tf.max() > 1.0:
            tf = tf / 255.0
        if tf.shape[-2:] != (self._image_size, self._image_size):
            tf = torch.nn.functional.interpolate(
                tf,
                size=(self._image_size, self._image_size),
                mode="bilinear",
                align_corners=False,
            )
        tt_host, _ = setup_l1_sharded_input(self.device, tf)
        return tt_host

    def make_batch_input(self, items: list):
        # Assemble a batch for YOLOv8s (single tensor for batch inference).
        if len(items) != self.batch_size:
            return None
        return items[0]

    def predict(self, tt_host):
        # Run inference and post-processing (NMS) for YOLOv8s.
        ttnn.copy_host_to_device_tensor(tt_host, self._image_res, cq_id=0)
        ttnn.execute_trace(self.device, self._trace_id, cq_id=0, blocking=False)

        heads = []
        for output, shape in zip(
            [self._model.output0, self._model.output1, self._model.output2],
            self._golden_shapes,
        ):
            heads.append(
                ttnn.to_torch(ttnn.from_device(output, blocking=True))
                .permute(0, 3, 1, 2)
                .reshape(shape)
                .float()
            )

        with torch.no_grad():
            final_output, _ = self._decode.post_detect_inference(heads)

        nms_out = self._nms(final_output, conf_thres=0.25, iou_thres=0.45)
        detections = nms_out[0]

        if detections is None or len(detections) == 0:
            return _dummy_detection()

        det = detections.detach().cpu().numpy()
        num_det = len(det)

        boxes = np.clip(det[:, :4] / self._image_size, 0.0, 1.0)
        boxes_yxyx = boxes[:, [1, 0, 3, 2]]
        scores = det[:, 4]
        classes = np.array(
            [self._class_map[int(c)] for c in det[:, 5]],
            dtype=np.float32,
        )

        return [
            np.array([num_det]),
            boxes_yxyx[np.newaxis],
            scores[np.newaxis],
            classes[np.newaxis],
        ]

    def _trace_capture(self):
        # Capture a trace for YOLOv8s inference.
        from models.bos_model.yolov8s.utilities.utility_functions import setup_l1_sharded_input

        dummy = torch.zeros([self.batch_size, 3, self._image_size, self._image_size], dtype=torch.float32)
        tt_host, mem_cfg = setup_l1_sharded_input(self.device, dummy)
        self._model.input_tensor = tt_host.to(self.device, mem_cfg)
        _ = self._model()

        self._trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)
        _ = self._model()
        ttnn.end_trace_capture(self.device, self._trace_id, cq_id=0)
        self._image_res = self._model.input_tensor

    def release(self):
        # Release trace resources if used.
        if self.use_trace and self._trace_id is not None:
            ttnn.release_trace(self.device, self._trace_id)


# ---------------------------------------------------------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------------------------------------------------------
def create_model_runner(model_type, device, batch_size, use_trace=True, use_2cq=False):
    """
    Factory function to create the appropriate model runner instance based on model_type.
    """
    registry = {
        "resnet50": ResNet50Runner,
        "vit": ViTRunner,
        "yolov8s": YoloV8sRunner,
    }

    if model_type not in registry:
        raise ValueError(f"Unknown model_type: {model_type}")

    return registry[model_type](
        device=device,
        batch_size=batch_size,
        use_trace=use_trace,
        use_2cq=use_2cq,
    )


# ---------------------------------------------------------------------------------------------------------------------------
# BackendTTNN
# ---------------------------------------------------------------------------------------------------------------------------
class BackendTTNN(backend.Backend):
    """
    Unified backend for TTNN models.
    Delegates all model-specific logic to the appropriate model runner instance.
    Provides a consistent interface for MLPerf inference code.
    """
    def __init__(self, model_type, batch_size, device_id=0, use_trace=True, use_2cq=False):
        super().__init__()
        self.model_type = model_type
        self.batch_size = batch_size
        self.device_id = device_id
        self.use_trace = use_trace
        self.use_2cq = use_2cq
        self.device = None
        self.model_runner = None
        self._lock = threading.Lock()

    def name(self):
        return f"ttnn-{self.model_type}"

    def version(self):
        try:
            return ttnn.__version__
        except Exception:
            return importlib.metadata.version("ttnn")

    def image_format(self):
        return "NCHW"

    def wrap_dataset(self, dataset):
        return TTNNDatasetWrapper(dataset, self)

    def load(self, model_path, inputs=None, outputs=None):
        # Create TTNN device and instantiate the model runner for the selected model type.
        self.device = ttnn.CreateDevice(
            device_id=self.device_id,
            l1_small_size=32768,
            trace_region_size=1605632 if self.use_trace else ttnn._ttnn.device.DEFAULT_TRACE_REGION_SIZE,
            num_command_queues=2 if self.use_2cq else 1,
        )
        self.device.enable_program_cache()

        self.model_runner = create_model_runner(
            model_type=self.model_type,
            device=self.device,
            batch_size=self.batch_size,
            use_trace=self.use_trace,
            use_2cq=self.use_2cq,
        )
        self.model_runner.load()

        self.inputs = inputs or ["image"]
        self.outputs = outputs or ["output"]
        return self

    def prepare_sample(self, numpy_img: np.ndarray):
        # Delegate sample preparation to the model runner.
        return self.model_runner.prepare_sample(numpy_img)

    def make_batch_input(self, items: list):
        # Delegate batch assembly to the model runner.
        return self.model_runner.make_batch_input(items)

    def predict(self, feed: dict):
        # Delegate inference to the model runner. Returns dummy output if input is None.
        x = feed[self.inputs[0]]
        if x is None:
            return self.model_runner.dummy_output()

        with self._lock:
            return self.model_runner.predict(x)

    def supports_async_predict(self):
        # Returns True when the backend supports pipelined 2CQ async predict.
        return (
            self.use_2cq
            and self.model_runner is not None
            and hasattr(self.model_runner, "predict_async")
        )

    def uses_pipeline_delay(self):
        # Returns True when predict_async uses 1-batch delay (5-event ViT pipeline).
        # In this mode, predict_async(batch_i) returns batch_i-1's result;
        # flush_pipeline() must be called after the final batch.
        return (
            self.use_2cq
            and self.model_runner is not None
            and hasattr(self.model_runner, "flush_pipeline")
        )

    def predict_async(self, feed: dict):
        # Submit inference non-blocking (2CQ pipeline).
        # Returns a host tensor handle; data is valid after synchronize().
        x = feed[self.inputs[0]]
        if x is None:
            return None
        with self._lock:
            return self.model_runner.predict_async(x)

    def flush_pipeline(self):
        # Issue DtoH for the last batch in the 5-event pipeline.
        # Returns the last host tensor (valid after synchronize()).
        if self.model_runner is not None and hasattr(self.model_runner, "flush_pipeline"):
            return self.model_runner.flush_pipeline()
        return None

    def synchronize(self):
        # Wait for all pending device operations to complete.
        ttnn.synchronize_device(self.device)
        if self.model_runner is not None and hasattr(self.model_runner, "post_synchronize"):
            self.model_runner.post_synchronize()

    def collect(self, host_out):
        # Convert a host tensor handle (from predict_async) to an inference result.
        if host_out is None:
            return self.model_runner.dummy_output()
        return self.model_runner._collect(host_out)

    def __del__(self):
        # Release model runner and device resources on destruction.
        if self.device is None:
            return
        try:
            if self.model_runner is not None:
                self.model_runner.release()
        except Exception:
            pass

        try:
            ttnn.CloseDevice(self.device)
        except Exception:
            pass
