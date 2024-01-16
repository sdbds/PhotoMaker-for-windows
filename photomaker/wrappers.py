"""
"""
from __future__ import annotations

import multiprocessing
import os
import signal
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from datetime import timedelta
from functools import partial
from functools import wraps
from pickle import PicklingError
from queue import Empty
from queue import Queue as ThreadQueue
from threading import Thread
from typing import TYPE_CHECKING
from typing import Callable
from typing import Generator
from typing import Generic
from typing_extensions import assert_never

import gradio as gr
import psutil

from ..utils import debug
from ..utils import drop_params
from ..utils import gradio_request_var
from ..utils import SimpleQueue as Queue
from . import client
from . import torch
from .gradio import GradioPartialContext
from .gradio import patch_gradio_queue
from .gradio import try_process_queue_event
from .types import * # TODO: Please don't do that


GENERATOR_GLOBAL_TIMEOUT = 20 * 60


Process = multiprocessing.get_context('spawn').Process
forked = False


class Worker(Generic[Res]):
    process: Process
    arg_queue: Queue[tuple[Params, GradioPartialContext]]
    res_queue: Queue[Res]
    _sentinel: Thread

    def __init__(
        self,
        target: Callable[[Queue[tuple[Params, GradioPartialContext]], Queue[Res], NvidiaUUID, list[int]], None],
        nvidia_uuid: str,
    ):
        self._sentinel = Thread(target=self._close_on_exit)
        self.arg_queue = Queue()
        self.res_queue = Queue()
        fds = [c.fd for c in psutil.Process().connections()]
        args = self.arg_queue, self.res_queue, nvidia_uuid, fds
        if TYPE_CHECKING:
            target(*args)
        self.process = Process(
            target=target,
            args=args,
            daemon=True,
        )
        self.process.start()
        self._sentinel.start()

    def _close_on_exit(self):
        self.process.join()
        self.res_queue.close()


def regular_function_wrapper(
    task: Callable[Param, Res],
    duration: timedelta | None,
    enable_queue: bool,
) -> Callable[Param, Res]:

    request_var = gradio_request_var()
    workers: dict[NvidiaIndex, Worker[RegularResQueueResult[Res]]] = {}
    task_id = id(task)

    @wraps(task)
    def gradio_handler(*args: Param.args, **kwargs: Param.kwargs) -> Res:

        if forked:
            return task(*args, **kwargs)

        request = request_var.get()
        schedule_response = client.schedule(task_id, request, duration, enable_queue)
        nvidia_index = schedule_response.nvidiaIndex
        nvidia_uuid = schedule_response.nvidiaUUID
        release = partial(client.release, task_id=task_id, nvidia_index=nvidia_index)

        worker = workers.get(nvidia_index)
        if worker is None or not worker.process.is_alive():
            worker = Worker(thread_wrapper, nvidia_uuid)
            workers[nvidia_index] = worker

        try:
            worker.arg_queue.put(((args, kwargs), GradioPartialContext.get()))
        except PicklingError:
            release(fail=True)
            raise

        while True:
            try:
                res = worker.res_queue.get()
            except EOFError:
                release(fail=True, allow_404=True)
                raise gr.Error("GPU task aborted")
            if isinstance(res, ExceptionResult):
                release(fail=True)
                raise res.value
            if isinstance(res, OkResult):
                release()
                return res.value
            if isinstance(res, GradioQueueEvent):
                try_process_queue_event(res.method_name, *res.args, **res.kwargs)
                continue
            assert_never(res)


    def thread_wrapper(
        arg_queue: Queue[tuple[Params, GradioPartialContext]],
        res_queue: Queue[RegularResQueueResult[Res]],
        nvidia_uuid: str,
        fds: list[int],
    ):
        global forked
        forked = True
        torch.unpatch()
        try:
            torch.move(nvidia_uuid)
        except Exception as e: # pragma: no cover
            traceback.print_exc()
            res_queue.put(ExceptionResult(e))
            return
        patch_gradio_queue(res_queue)
        for fd in fds:
            try:
                os.close(fd)
            except Exception as e: # pragma: no cover
                if isinstance(e, OSError) and e.errno == 9:
                    continue
                traceback.print_exc()
                res_queue.put(ExceptionResult(e))
                return
        signal.signal(signal.SIGTERM, drop_params(arg_queue.close))
        while True:
            try:
                (args, kwargs), gradio_context = arg_queue.get()
            except OSError:
                break
            GradioPartialContext.apply(gradio_context)
            context = copy_context()
            with ThreadPoolExecutor() as executor:
                future = executor.submit(context.run, task, *args, **kwargs) # type: ignore
            try:
                res = future.result()
            except Exception as e:
                traceback.print_exc()
                res = ExceptionResult(e)
            else:
                res = OkResult(res)
            try:
                res_queue.put(res)
            except PicklingError as e:
                res_queue.put(ExceptionResult(e))


    return gradio_handler


def generator_function_wrapper(
    task: Callable[Param, Generator[Res, None, None]],
    duration: timedelta | None,
    enable_queue: bool,
) -> Callable[Param, Generator[Res, None, None]]:

    request_var = gradio_request_var()
    workers: dict[NvidiaIndex, Worker[GeneratorResQueueResult[Res]]] = {}
    task_id = id(task)

    @wraps(task)
    def gradio_handler(*args: Param.args, **kwargs: Param.kwargs) -> Generator[Res, None, None]:

        if forked:
            yield from task(*args, **kwargs)
            return

        request = request_var.get()
        schedule_response = client.schedule(task_id, request, duration, enable_queue)
        nvidia_index = schedule_response.nvidiaIndex
        nvidia_uuid = schedule_response.nvidiaUUID
        release = partial(client.release, task_id=task_id, nvidia_index=nvidia_index)

        worker = workers.get(nvidia_index)
        if worker is None or not worker.process.is_alive():
            worker = Worker(thread_wrapper, nvidia_uuid)
            workers[nvidia_index] = worker

        try:
            worker.arg_queue.put(((args, kwargs), GradioPartialContext.get()))
        except PicklingError:
            release(fail=True)
            raise

        yield_queue: ThreadQueue[YieldQueueResult[Res]] = ThreadQueue()
        def fill_yield_queue(worker: Worker[GeneratorResQueueResult[Res]]):
            while True:
                try:
                    res = worker.res_queue.get()
                except Exception:
                    release(fail=True, allow_404=True)
                    yield_queue.put(AbortedResult())
                    return
                if isinstance(res, ExceptionResult):
                    release(fail=True)
                    yield_queue.put(ExceptionResult(res.value))
                    return
                if isinstance(res, EndResult):
                    release()
                    yield_queue.put(EndResult())
                    return
                if isinstance(res, OkResult):
                    yield_queue.put(OkResult(res.value))
                    continue
                if isinstance(res, GradioQueueEvent): # pragma: no cover (not working properly on Gradio side)
                    try_process_queue_event(res.method_name, *res.args, **res.kwargs)
                    continue
                debug(f"fill_yield_queue: assert_never({res=})")
                assert_never(res)
        from typing_extensions import assert_never
        with ThreadPoolExecutor() as e:
            f = e.submit(fill_yield_queue, worker)
            f.add_done_callback(lambda _: debug("fill_yield_queue DONE"))
            while True:
                try:
                    res = yield_queue.get(timeout=GENERATOR_GLOBAL_TIMEOUT)
                except Empty: # pragma: no cover
                    debug(f"yield_queue TIMEOUT ({GENERATOR_GLOBAL_TIMEOUT=})")
                    raise
                if isinstance(res, AbortedResult):
                    raise gr.Error("GPU task aborted")
                if isinstance(res, ExceptionResult):
                    raise res.value
                if isinstance(res, EndResult):
                    break
                if isinstance(res, OkResult):
                    yield res.value
                    continue
                debug(f"gradio_handler: assert_never({res=})")
                assert_never(res)


    def thread_wrapper(
        arg_queue: Queue[tuple[Params, GradioPartialContext]],
        res_queue: Queue[GeneratorResQueueResult[Res]],
        nvidia_uuid: str,
        fds: list[int],
    ):
        global forked
        forked = True
        torch.unpatch()
        try:
            torch.move(nvidia_uuid)
        except Exception as e: # pragma: no cover
            traceback.print_exc()
            res_queue.put(ExceptionResult(e))
            return
        patch_gradio_queue(res_queue)
        for fd in fds:
            try:
                os.close(fd)
            except Exception as e: # pragma: no cover
                if isinstance(e, OSError) and e.errno == 9:
                    continue
                traceback.print_exc()
                res_queue.put(ExceptionResult(e))
                return
        signal.signal(signal.SIGTERM, drop_params(arg_queue.close))
        while True:
            try:
                (args, kwargs), gradio_context = arg_queue.get()
            except OSError:
                break
            def iterate():
                gen = task(*args, **kwargs) # type: ignore
                while True:
                    try:
                        res = next(gen)
                    except StopIteration:
                        break
                    except Exception as e:
                        res_queue.put(ExceptionResult(e))
                        break
                    try:
                        res_queue.put(OkResult(res))
                    except PicklingError as e:
                        res_queue.put(ExceptionResult(e))
                        break
                    else:
                        continue
            GradioPartialContext.apply(gradio_context)
            context = copy_context()
            with ThreadPoolExecutor() as executor:
                executor.submit(context.run, iterate)
            res_queue.put(EndResult())

    return gradio_handler
