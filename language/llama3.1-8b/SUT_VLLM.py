import asyncio
import os
import time
import numpy as np
import array
import torch
from torch.nn.functional import pad
from vllm import LLM, AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from vllm.inputs import TokensPrompt

import pickle
import time
import threading
import tqdm
import queue

import logging
from typing import TYPE_CHECKING, Optional, List
from pathlib import Path

import mlperf_loadgen as lg
from dataset import Dataset

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Llama-8B-SUT")


class SUT:
    def __init__(
        self,
        model_path=None,
        dtype="bfloat16",
        batch_size=None,
        total_sample_count=13368,
        dataset_path=None,
        use_cached_outputs=False,
        # Set this to True *only for test accuracy runs* in case your prior
        # session was killed partway through
        workers=1,
        tensor_parallel_size=8
    ):

        self.model_path = model_path or f"meta-llama/Meta-Llama-3.1-8B-Instruct"

        if not batch_size:
            batch_size = 1
        self.batch_size = batch_size

        self.dtype = dtype
        self.tensor_parallel_size = tensor_parallel_size

        # if not torch.cuda.is_available():
        #     assert False, "torch gpu is not available, exiting..."

        self.dataset_path = dataset_path
        self.data_object = Dataset(
            self.model_path,
            dataset_path=self.dataset_path,
            total_sample_count=total_sample_count,
            dtype=dtype
        )
        self.qsl = lg.ConstructQSL(
            self.data_object.total_sample_count,
            self.data_object.perf_count,
            self.data_object.LoadSamplesToRam,
            self.data_object.UnloadSamplesFromRam,
        )

        self.load_model()
        gen_kwargs = {
            "temperature": 0.0,
            "top_p": 1,
            "top_k": 1,
            "seed": 42,
            "max_tokens": 128,
            "min_tokens": 1
        }
        self.sampling_params = SamplingParams(**gen_kwargs)
        # self.sampling_params.all_stop_token_ids.add(self.model.get_tokenizer().eos_token_id)

        self.num_workers = workers
        self.worker_threads = [None] * self.num_workers
        self.query_queue = queue.Queue()

        self.use_cached_outputs = use_cached_outputs
        self.sample_counter = 0
        self.sample_counter_lock = threading.Lock()

    def start(self):
        # Create worker threads
        for j in range(self.num_workers):
            worker = threading.Thread(target=self.process_queries)
            worker.start()
            self.worker_threads[j] = worker

    def stop(self):
        for _ in range(self.num_workers):
            self.query_queue.put(None)

        for worker in self.worker_threads:
            worker.join()

    def process_queries(self):
        """Processor of the queued queries. User may choose to add batching logic"""
        while True:
            qitem = self.query_queue.get()
            if qitem is None:
                break

            query_ids = [q.index for q in qitem]

            tik1 = time.time()

            input_ids_tensor = [
                token_id for q in qitem for token_id in self.data_object.input_ids[q.index]]
            # input_text_tensor = [
            #     self.data_object.input[q.index] for q in qitem]
            # for in_text in input_text_tensor:
            #     log.info(f"Input: {in_text}")

            tik2 = time.time()
            outputs = self.model.generate(
                {"prompt_token_ids": input_ids_tensor}, sampling_params=self.sampling_params
            )
            pred_output_tokens = []
            for output in outputs:
                pred_output_tokens.append(list(output.outputs[0].token_ids))
                # log.info(f"Output: {output.outputs[0].text}")
            tik3 = time.time()

            processed_output = self.data_object.postProcess(
                pred_output_tokens,
                query_id_list=query_ids,
            )
            for i in range(len(qitem)):
                n_tokens = processed_output[i].shape[0]
                response_array = array.array(
                    "B", processed_output[i].tobytes())
                bi = response_array.buffer_info()
                response = [
                    lg.QuerySampleResponse(
                        qitem[i].id,
                        bi[0],
                        bi[1],
                        n_tokens)]
                lg.QuerySamplesComplete(response)

            tok = time.time()

            with self.sample_counter_lock:
                self.sample_counter += len(qitem)
                log.info(f"Samples run: {self.sample_counter}")
                if tik1:
                    log.info(f"\tBatchMaker time: {tik2 - tik1}")
                    log.info(f"\tInference time: {tik3 - tik2}")
                    log.info(f"\tPostprocess time: {tok - tik3}")
                    log.info(f"\t==== Total time: {tok - tik1}")

    def load_model(self):
        log.info("Loading model...")
        self.model = LLM(
            self.model_path,
            dtype=self.dtype,
            tensor_parallel_size=self.tensor_parallel_size,
            max_num_seqs=1,  # max_batch_size
            block_size=64,  # KV cache block size
            override_tt_config={"enable_model_warmup": False}
        )
        log.info("Loaded model")

    def get_sut(self):
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        return self.sut

    def get_qsl(self):
        return self.qsl

    def predict(self, **kwargs):
        raise NotImplementedError

    def issue_queries(self, query_samples):
        """Receives samples from loadgen and adds them to queue. Users may choose to batch here"""

        list_prompts_tokens = []
        list_prompts_attn_masks = []

        log.info(f"IssueQuery started with {len(query_samples)} samples")
        while len(query_samples) > 0:
            self.query_queue.put(query_samples[: self.batch_size])
            query_samples = query_samples[self.batch_size:]
        log.info(f"IssueQuery done")

    def flush_queries(self):
        pass

    def __del__(self):
        pass


class SUTTTNN(SUT):
    def __init__(
        self,
        model_path=None,
        dtype="bfloat16",
        batch_size=None,
        total_sample_count=13368,
        dataset_path=None,
        use_cached_outputs=False,
        # Set this to True *only for test accuracy runs* in case your prior
        # session was killed partway through
        workers=1,
        tensor_parallel_size=8,
        first_token_tracking=True,
    ):
        super().__init__(
            model_path=model_path,
            dtype=dtype,
            batch_size=batch_size,
            total_sample_count=total_sample_count,
            dataset_path=dataset_path,
            use_cached_outputs=use_cached_outputs,
            workers=workers,
            tensor_parallel_size=tensor_parallel_size,
        )
        self.first_token_tracking = first_token_tracking

    def _generate_with_first_token(self, qitem, input_ids):
        """Step through the engine one step at a time and call FirstTokenComplete as soon as the first token is produced."""
        import uuid as _uuid
        request_id = str(_uuid.uuid4())
        prompt = TokensPrompt(prompt_token_ids=list(input_ids))
        engine = self.model.llm_engine
        engine.add_request(request_id, prompt, self.sampling_params)

        first_token_sent = False
        final_output = None
        t_start = time.time()
        while engine.has_unfinished_requests():
            step_outputs = engine.step()
            for out in step_outputs:
                if out.request_id != request_id:
                    continue
                if (not first_token_sent
                        and out.outputs
                        and len(out.outputs[0].token_ids) > 0):
                    ft_ids = list(out.outputs[0].token_ids[:1])
                    ft_arr = array.array("B", np.array(ft_ids, np.int32).tobytes())
                    bi = ft_arr.buffer_info()
                    lg.FirstTokenComplete(
                        [lg.QuerySampleResponse(qitem.id, bi[0], bi[1])])
                    first_token_sent = True
                if out.finished:
                    final_output = out

        # Fallback: for backends (e.g. TT-Metal) that do not expose intermediate tokens
        # during step(), FirstTokenComplete may not have been called before the loop ends.
        # In that case TTFT equals total generation time, so emit a warning.
        if not first_token_sent:
            if final_output and final_output.outputs and final_output.outputs[0].token_ids:
                ft_ids = list(final_output.outputs[0].token_ids[:1])
            else:
                ft_ids = []
            ft_arr = array.array("B", np.array(ft_ids, np.int32).tobytes())
            bi = ft_arr.buffer_info()
            lg.FirstTokenComplete(
                [lg.QuerySampleResponse(qitem.id, bi[0], bi[1])])
            log.warning(
                f"[TTFT] Fallback: engine did not expose tokens mid-step. "
                f"FirstTokenComplete called after full generation "
                f"({(time.time() - t_start)*1000:.2f} ms). "
                f"TTFT equals total decoding time — streaming not available in this backend."
            )

        return final_output

    def process_queries(self):
        """Processor of the queued queries. User may choose to add batching logic"""
        while True:
            qitem = self.query_queue.get()
            if qitem is None:
                break

            query_ids = [q.index for q in qitem]

            tik1 = time.time()

            if self.first_token_tracking and len(qitem) == 1:
                tik2 = time.time()
                output = self._generate_with_first_token(
                    qitem[0], self.data_object.input_ids[qitem[0].index])
                pred_output_tokens = [
                    list(output.outputs[0].token_ids) if output else []
                ]
            else:
                input_ids_tensor = [
                    token_id for q in qitem for token_id in self.data_object.input_ids[q.index]]
                tik2 = time.time()
                outputs = self.model.generate(
                    {"prompt_token_ids": input_ids_tensor}, sampling_params=self.sampling_params
                )
                pred_output_tokens = []
                for output in outputs:
                    pred_output_tokens.append(list(output.outputs[0].token_ids))
            tik3 = time.time()

            processed_output = self.data_object.postProcess(
                pred_output_tokens,
                query_id_list=query_ids,
            )
            for i in range(len(qitem)):
                n_tokens = processed_output[i].shape[0]
                response_array = array.array(
                    "B", processed_output[i].tobytes())
                bi = response_array.buffer_info()
                response = [
                    lg.QuerySampleResponse(
                        qitem[i].id,
                        bi[0],
                        bi[1],
                        n_tokens)]
                lg.QuerySamplesComplete(response)

            tok = time.time()

            with self.sample_counter_lock:
                self.sample_counter += len(qitem)
                log.info(f"Samples run: {self.sample_counter}")
                if tik1:
                    log.info(f"\tBatchMaker time: {tik2 - tik1}")
                    log.info(f"\tInference time: {tik3 - tik2}")
                    log.info(f"\tPostprocess time: {tok - tik3}")
                    log.info(f"\t==== Total time: {tok - tik1}")

    def load_model(self):
        log.info("Loading model...")
        log.info(
            "NOTE: trace_mode='none' (trace disabled). "
            "To enable TT-Metal graph capture, set trace_mode='light' or 'full' "
            "and enable_model_warmup=True."
        )
        self.model = LLM(
            self.model_path,
            dtype=self.dtype,
            tensor_parallel_size=self.tensor_parallel_size,
            max_num_seqs=1,  # max_batch_size
            block_size=64,  # KV cache block size
            override_tt_config={"enable_model_warmup": False},
        )
        log.info("Loaded model")
        self._warmup_model()

    def _warmup_model(self):
        """Run a dummy inference to trigger JIT compilation ahead of time,
        preventing compile overhead from appearing in the first real sample.
        Uses the step() path to compile the same code path as _generate_with_first_token.
        """
        import uuid as _uuid
        log.info("Warming up model (dummy inference to trigger JIT compile)...")
        t_warmup = time.time()

        tokenizer = self.model.get_tokenizer()
        dummy_ids = tokenizer.encode("warmup", add_special_tokens=False)[:8] or [1, 2, 3, 4]
        prompt = TokensPrompt(prompt_token_ids=dummy_ids)
        engine = self.model.llm_engine

        dummy_sp = SamplingParams(temperature=0.0, top_p=1, top_k=1, seed=42, max_tokens=2, min_tokens=1)
        request_id = str(_uuid.uuid4())
        engine.add_request(request_id, prompt, dummy_sp)
        while engine.has_unfinished_requests():
            engine.step()

        log.info(f"Warmup done in {(time.time() - t_warmup)*1000:.2f} ms")


class SUTServer(SUT):
    def __init__(
        self,
        model_path=None,
        dtype="bfloat16",
        total_sample_count=13368,
        dataset_path=None,
        batch_size=None,
        workers=1,
        tensor_parallel_size=1
    ):

        super().__init__(
            model_path=model_path,
            dtype=dtype,
            total_sample_count=total_sample_count,
            dataset_path=dataset_path,
            workers=workers,
            tensor_parallel_size=tensor_parallel_size,
        )
        self.request_id = 0

        self.first_token_queue = queue.Queue()

    def start(self):
        # Create worker threads
        for j in range(self.num_workers):
            worker = threading.Thread(target=self.process_queries)
            worker.start()
            self.worker_threads[j] = worker

    async def stream_output(self, qitem, results_generator):
        first = True
        async for request_output in results_generator:
            output_response = request_output
            if first:
                first_tokens = list(output_response.outputs[0].token_ids)
                response_data = array.array(
                    "B", np.array(first_tokens, np.int32).tobytes())
                bi = response_data.buffer_info()
                response = [lg.QuerySampleResponse(qitem.id, bi[0], bi[1])]
                lg.FirstTokenComplete(response)
                first = False

        outputs = output_response
        pred_output_tokens = list(output_response.outputs[0].token_ids)
        n_tokens = len(pred_output_tokens)
        response_array = array.array(
            "B", np.array(pred_output_tokens, np.int32).tobytes()
        )
        bi = response_array.buffer_info()
        response = [
            lg.QuerySampleResponse(
                qitem.id,
                bi[0],
                bi[1],
                n_tokens)]
        lg.QuerySamplesComplete(response)

    def process_queries(self):
        """Processor of the queued queries. User may choose to add batching logic"""
        while True:

            qitem = self.query_queue.get()
            if qitem is None:
                break

            input_ids_tensor = TokensPrompt(
                prompt_token_ids=self.data_object.input_ids[qitem.index])

            # TODO: This PoC is super slow with significant overhead. Best to
            # create a patch to `generate`
            results_generator = self.model.generate(
                prompt=input_ids_tensor, sampling_params=self.sampling_params, request_id=str(
                    self.request_id)
            )
            self.request_id += 1
            asyncio.run(self.stream_output(qitem, results_generator))

    def issue_queries(self, query_samples):
        self.query_queue.put(query_samples[0])

    def stop(self):
        for _ in range(self.num_workers):
            self.query_queue.put(None)

        for worker in self.worker_threads:
            worker.join()

        self.first_token_queue.put(None)
        self.ft_response_thread.join()

    def load_model(self):
        log.info("Loading model")
        self.engine_args = AsyncEngineArgs(
            self.model_path,
            dtype=self.dtype,
            tensor_parallel_size=self.tensor_parallel_size,
            max_num_seqs=1,  # max_batch_size
            block_size=64,  # KV cache block size
            override_tt_config={"enable_model_warmup": False},
        )
        self.model = AsyncLLMEngine.from_engine_args(self.engine_args)
        log.info("Loaded model")
